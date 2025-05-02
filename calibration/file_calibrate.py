#!/usr/bin/env python
# file_calibrate.py - Calibration script for BeforeIT model using file-based client

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time
import argparse
import sys
import os

from black_it.calibrator import Calibrator
from black_it.loss_functions.msm import MethodOfMomentsLoss
from black_it.plot.plot_results import (
    plot_convergence,
    plot_losses,
    plot_sampling,
)
from black_it.samplers.best_batch import BestBatchSampler
from black_it.samplers.halton import HaltonSampler
from black_it.samplers.random_forest import RandomForestSampler

# Import the updated client with ServerMonitor
from file_based_client import BeforeITFileClient, ServerMonitor

def get_parameter_bounds_for_version(version_c):
    """
    Return the parameter bounds and precisions for a specific version.
    
    Parameters:
    -----------
    version_c: int
        The version of the consumption heuristic (1-8)
        
    Returns:
    --------
    bounds: list of lists
        Lower and upper bounds for parameters
    precisions: list
        Precision for each parameter
    """
    # Version 1: c = κ + γy
    if version_c == 1:
        # γ (MPC for income)
        bounds = [[0.5], [0.99]]
        precisions = [0.01]
    
    # Version 2: c = κ + α * y^β
    elif version_c == 2:
        # α, β
        bounds = [[0.0, 0.0], [2.0, 0.7]]
        precisions = [0.01, 0.01]
    
    # Version 3: c = κ + γy + δw
    elif version_c == 3:
        # γ (MPC for income), δ (MPC for wealth)
        bounds = [[0.5, 0.0], [0.99, 0.1]]
        precisions = [0.01, 0.005]
    
    # Version 4: c = κ + αy^β + δw
    elif version_c == 4:
        # α, β, δ
        bounds = [[0.0, 0.0, 0.0], [2.0, 0.7, 0.1]]
        precisions = [0.01, 0.01, 0.005]
    
    # Version 5: c = κ + αy^β + θw^ρ
    elif version_c == 5:
        # α, β, θ, ρ
        bounds = [[0.0, 0.0, 0.0, 0.0], [2.0, 0.7, 0.5, 0.7]]
        precisions = [0.01, 0.01, 0.01, 0.01]
    
    # Version 6: c = κ + γy + αy^β
    elif version_c == 6:
        # γ, α, β
        bounds = [[0.0, 0.0, 0.0], [0.9, 2.0, 0.7]]
        precisions = [0.01, 0.01, 0.01]
    
    # Version 7: c = κ + γy + αy^β + δw
    elif version_c == 7:
        # γ, α, β, δ
        bounds = [[0.0, 0.0, 0.0, 0.0], [0.9, 2.0, 0.7, 0.1]]
        precisions = [0.01, 0.01, 0.01, 0.005]
    
    # Version 8: c = κ + γy + αy^β + δw + θw^ρ
    elif version_c == 8:
        # γ, α, β, δ, θ, ρ
        bounds = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9, 2.0, 0.7, 0.1, 0.5, 0.7]]
        precisions = [0.01, 0.01, 0.01, 0.005, 0.01, 0.01]
    else:
        raise ValueError(f"Invalid version number: {version_c}. Must be between 1 and 8.")
    
    return bounds, precisions

def calibrate_model(version_c, monitor, real_data, start_date, end_date, saving_folder,
batch_size=4, max_iterations=10, num_servers=4):
    """
    Calibrate a specific version of the consumption heuristic.
    
    Parameters:
    -----------
    version_c: int
        The version of the consumption heuristic (1-8)
    monitor: ServerMonitor
        Server monitor to distribute jobs across multiple servers
    real_data: numpy.ndarray
        Real data for comparison
    start_date, end_date: str
        Date range for the calibration
    batch_size: int
        Batch size for the samplers
    max_iterations: int
        Maximum number of calibration iterations
    num_servers: int
        Number of servers to use for parallel processing
        
    Returns:
    --------
    params: list
        Best parameters found
    losses: list
        Loss history
    """
    print(f"\n{'='*60}")
    print(f"Calibrating Version {version_c}")
    print(f"{'='*60}")
    
    # Get parameter bounds for this version
    bounds, precisions = get_parameter_bounds_for_version(version_c)
    
    # Loss function
    # Create a mask for padded zeros and calculate weights
    print("Calculating weights for loss function...")
    mask = (real_data == 0.0)
    masked_data = np.ma.array(real_data, mask=mask)

    # Calculate proper standard errors on non-padded data only
    std_errors = np.ma.std(masked_data, axis=0).data
    weights = 1.0 / (std_errors + 1e-10)  # Add small constant to avoid division by zero
    weights /= np.sum(weights)  # Normalize weights to sum to 1

    # Create Method of Moments loss with these weights
    loss = MethodOfMomentsLoss(
        covariance_mat='identity',  # Already weights moments by inverse variance
        coordinate_weights=weights  # Weights per variable based on inverse std errors
    )
    
    # Samplers
    halton_sampler = HaltonSampler(batch_size=batch_size)
    random_forest_sampler = RandomForestSampler(batch_size=batch_size)
    best_batch_sampler = BestBatchSampler(batch_size=batch_size)
    
    samplers = [halton_sampler, random_forest_sampler, best_batch_sampler]
    
    # Define wrapper for this specific version that uses the monitor
    def abm_wrapper(params, sim_length=None, seed=None):
        """
        Wrapper for the BeforeIT simulation function using ServerMonitor.
        
        Args:
            params: Parameters to simulate with
            sim_length: Length of simulation (ignored, handled by server)
            seed: Random seed (ignored, handled by server)
        
        Returns:
            NumPy array with simulation results
        """
        # Convert NumPy arrays to Python lists for JSON serialization
        if isinstance(params, np.ndarray):
            params = params.tolist()
        
        # Use the monitor to distribute work to the next available server
        return monitor.run_simulation(
            params=params, 
            start_date=start_date, 
            end_date=end_date,
            version_c=version_c,
            num_simulations=10,  # Number of simulations to run
        )
    
    # Initialize a Calibrator object
    cal = Calibrator(
        samplers=samplers,
        real_data=real_data,
        model=abm_wrapper,
        parameters_bounds=bounds,
        parameters_precision=precisions,
        ensemble_size=1,
        loss_function=loss,
        saving_folder=saving_folder,
        n_jobs=num_servers,  # Use as many jobs as we have servers
    )
    
    # Run calibration
    start_time = time.time()
    params, losses = cal.calibrate(max_iterations)
    end_time = time.time()
    
    print(f"Calibration for version {version_c} took {end_time - start_time:.2f} seconds")
    print(f"Best parameters: {params}")
    print(f"Final loss: {losses[-1]}")
    
    return params, losses

# Helper function to convert NumPy arrays to Python lists for JSON serialization
def numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    else:
        return obj

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calibrate BeforeIT model for a specific version')
    parser.add_argument('--version', type=int, required=True, choices=range(1, 9), 
                      help='Version number to calibrate (1-8)')
    parser.add_argument('--start-date', type=str, default="2010-03-31", 
                      help='Start date for calibration (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default="2013-12-31", 
                      help='End date for calibration (YYYY-MM-DD)')
    parser.add_argument('--batch-size', type=int, default=4, 
                      help='Batch size for samplers')
    parser.add_argument('--max-iterations', type=int, default=10, 
                      help='Maximum number of calibration iterations')
    parser.add_argument('--num-servers', type=int, default=4,
                      help='Number of servers to use')
    parser.add_argument('--base-dir', type=str, default="./shared_data_",
                      help='Base directory for server data (will append server number)')
    
    args = parser.parse_args()
    
    # Get the version to calibrate
    version_c = args.version
    
    # Configuration
    start_date = args.start_date
    end_date = args.end_date
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_servers = args.num_servers
    base_dir = args.base_dir
    
    print(f"Starting calibration for Version {version_c}")
    print(f"Start date: {start_date}, End date: {end_date}")
    print(f"Batch size: {batch_size}, Max iterations: {max_iterations}")
    print(f"Using {num_servers} servers with base directory: {base_dir}")
    
    # Create ServerMonitor
    monitor = ServerMonitor(num_servers=num_servers, base_dir=base_dir)
    
    # Initialize all servers
    print("Initializing all servers...")
    monitor.initialize_all_servers(start_date=start_date, end_date=end_date)
    
    # Get real data from the first server
    print("Getting real data...")
    # Use the first client to get real data
    client = monitor.clients[0]
    real_data = np.array(client.get_data(start_date=start_date, end_date=end_date))
    print(f"Real data shape: {real_data.shape}")
    
        # Create a mask for padded zeros and calculate weights
    print("Calculating weights for loss function...")
    mask = (real_data == 0.0)
    masked_data = np.ma.array(real_data, mask=mask)

    # Calculate proper standard errors on non-padded data only
    std_errors = np.ma.std(masked_data, axis=0).data
    weights = 1.0 / (std_errors + 1e-10)  # Add small constant to avoid division by zero
    weights /= np.sum(weights)  # Normalize weights to sum to 1

    # Create Method of Moments loss with these weights
    loss = MethodOfMomentsLoss(
        covariance_mat='identity',  # Already weights moments by inverse variance
        coordinate_weights=weights  # Weights per variable based on inverse std errors
    )
    # Create output folder
    saving_folder = f'extended_heuristic/v{version_c}'
    Path(saving_folder).mkdir(parents=True, exist_ok=True)

    # Calibrate the specified version
    params, losses = calibrate_model(
        version_c, 
        monitor, 
        real_data, 
        start_date, 
        end_date, 
        batch_size=batch_size,
        max_iterations=max_iterations,
        num_servers=num_servers,
        saving_folder=saving_folder
    )
    
    # Save results for this version
    Path("extended_heuristic").mkdir(parents=True, exist_ok=True)
    
    # Convert NumPy arrays to Python lists for JSON serialization
    results = {
        "version": version_c,
        "params": numpy_to_python(params),
        "final_loss": float(losses[-1]) if isinstance(losses[-1], (np.float32, np.float64)) else losses[-1],
        "loss_history": numpy_to_python(losses),
        "config": {
            "start_date": start_date,
            "end_date": end_date,
            "batch_size": batch_size,
            "max_iterations": max_iterations,
            "num_servers": num_servers
        }
    }
    
    with open(f"extended_heuristic/v{version_c}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCalibration for Version {version_c} completed")
    print(f"Best parameters: {numpy_to_python(params)}")
    print(f"Final loss: {float(losses[-1]) if isinstance(losses[-1], (np.float32, np.float64)) else losses[-1]}")
    print(f"Results saved to: extended_heuristic/v{version_c}_results.json")

if __name__ == "__main__":
    main()