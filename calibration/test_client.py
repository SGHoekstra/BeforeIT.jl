import sys
import os
import time
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from black_it.calibrator import Calibrator
from black_it.loss_functions.msm import MethodOfMomentsLoss
from black_it.loss_functions.minkowski import MinkowskiLoss
from black_it.samplers.best_batch import BestBatchSampler
from black_it.samplers.halton import HaltonSampler
from black_it.samplers.random_forest import RandomForestSampler

# Import the simplified ServerMonitor
from calibration.file_based_client import ServerMonitor

# Configuration
num_servers = 4  # Number of server instances
batch_size = 5
bounds = [[0.0], [1.1]]
precisions = [0.01]
start_date = "2010-03-31"
end_date = "2013-12-31"
num_iterations = 50  # Number of calibration iterations
saving_folder = f'calibration_union'
os.makedirs(saving_folder, exist_ok=True)

print(f"Starting calibration for union model")
print(f"Using {num_servers} servers")
print(f"Parameter bounds: {bounds}")
print(f"Start date: {start_date}, End date: {end_date}")

# Initialize the ServerMonitor
monitor = ServerMonitor(num_servers=num_servers)

# Initialize all servers with the same model/parameters
try:
    print("Initializing all servers...")
    monitor.initialize_all_servers(
        start_date=start_date,
        end_date=end_date
    )
    print("All servers initialized successfully")
except Exception as e:
    print(f"Error initializing servers: {e}")
    sys.exit(1)

# Get real data from the first server
print("Getting real data for calibration...")
try:
    # Get the real data from server 1 (index 0)
    client = monitor.clients[0]
    
    # Get the real data
    real_data = np.array(client.get_data(
        start_date=start_date,
        end_date=end_date,
        repetitions=10
    ))
    real_data = real_data.T
    print(f"Real data shape: {real_data.shape}")
except Exception as e:
    print(f"Error getting real data: {e}")
    sys.exit(1)


def abm_wrapper(params, sim_length=None, seed=None):
    """Wrapper for BeforeIT simulations for Black-it"""
    try:
        # Convert NumPy arrays to lists for JSON serialization if needed
        if isinstance(params, np.ndarray):
            params_list = params.tolist()
        else:
            params_list = params

        # Use the ServerMonitor to run the simulation on the next available server
        result = monitor.run_simulation(
            params=params_list,
            start_date=start_date,
            end_date=end_date,
            num_simulations=10,
        )

        return result.T
    except Exception as e:
        print(f"Error in ABM wrapper: {e}")
        # Return dummy data in case of error to avoid crashing the calibration
        return np.zeros_like(real_data)
    
# Test the function with some dummy parameters
test_params = np.array([0.9])
result = abm_wrapper(test_params)
print("\nTest with valid parameters:")
print("Result shape:", result.shape)
print("Result type:", type(result))

# Setup samplers
print("Setting up samplers...")
halton_sampler = HaltonSampler(batch_size=batch_size)
random_forest_sampler = RandomForestSampler(batch_size=batch_size)
best_batch_sampler = BestBatchSampler(batch_size=batch_size)
samplers = [halton_sampler, random_forest_sampler, best_batch_sampler]

from black_it.loss_functions.minkowski import MinkowskiLoss

loss = MinkowskiLoss()

# Initialize the Calibrator
print("Initializing calibrator...")
cal = Calibrator(
    samplers=samplers,
    real_data=real_data,
    model=abm_wrapper,
    parameters_bounds=bounds,
    parameters_precision=precisions,
    ensemble_size=1,
    loss_function=custom_loss,
    saving_folder=saving_folder,
    n_jobs=4  # Set to 4 to allow the calibrator to send multiple sets of parameters at once
)

# Run calibration
print(f"Starting calibration with {num_iterations} iterations...")
start_time = time.time()

try:
    params, losses = cal.calibrate(num_iterations)
    end_time = time.time()
    
    # Log results
    print(f"Calibration completed in {end_time - start_time:.2f} seconds")
    
    # Find best parameters
    min_loss = np.min(cal.losses_samp)
    idxmin = np.argmin(cal.losses_samp)
    param_min = cal.params_samp[idxmin]
    
    print(f"Best parameters: {param_min}")
    print(f"Minimum loss: {min_loss}")
    
except Exception as e:
    end_time = time.time()
    print(f"Error during calibration: {e}")
    print(f"Calibration failed after {end_time - start_time:.2f} seconds")

# Generate plots
try:
    print("Generating plots...")
    
    # Plot convergence
    calibration_results_file = Path(saving_folder) / "calibration_results.csv"
    data_frame = pd.read_csv(calibration_results_file)
    
    # Convergence plot
    losses_cummin = data_frame.groupby("batch_num_samp").min()["losses_samp"].cummin()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=data_frame,
        x="batch_num_samp",
        y="losses_samp",
        hue="method_samp",
        palette="tab10",
    )
    
    sns.lineplot(
        x=np.arange(max(data_frame["batch_num_samp"]) + 1),
        y=losses_cummin,
        color="black",
        ls="--",
        marker="o",
        label="min loss",
    )
    
    plt.title("Convergence of Loss Function")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss Value")
    plt.savefig(f"{saving_folder}/convergence_plot.png", dpi=300)
    
    # Parameter plots
    num_params = sum("params_samp_" in c_str for c_str in data_frame.columns)
    variables = ["params_samp_" + str(i) for i in range(num_params)]
    
    # Loss plot
    g = sns.pairplot(
        data_frame,
        hue="losses_samp",
        vars=variables,
        diag_kind="hist",
        corner=True,
        palette="viridis",
        plot_kws={"markers": "o", "linewidth": 1, "alpha": 0.8},
        diag_kws={
            "fill": False,
            "hue": None,
        },
    )
    
    g.legend.set_bbox_to_anchor((0.8, 0.5))
    g.fig.suptitle("Parameter Values Colored by Loss")
    plt.savefig(f"{saving_folder}/parameters_by_loss.png", dpi=300)
    
    # Sampling method plot
    g = sns.pairplot(
        data_frame,
        hue="method_samp",
        vars=variables,
        diag_kind="hist",
        corner=True,
        palette="tab10",
        plot_kws={"markers": "o", "linewidth": 1, "alpha": 0.8},
        diag_kws={
            "fill": False,
        },
    )
    
    # Update legend
    handles, _ = g.axes[-1][0].get_legend_handles_labels()
    g.legend.remove()
    plt.legend(loc=2, handles=handles, bbox_to_anchor=(0.0, 1.8))
    g.fig.suptitle("Parameter Values by Sampling Method")
    plt.savefig(f"{saving_folder}/parameters_by_method.png", dpi=300)
    
    print("Plots generated successfully")
    
except Exception as e:
    print(f"Error generating plots: {e}")

print("Calibration process completed")

import numpy as np
from black_it.loss_functions.base import BaseLoss

class EconomicTimeSeriesRMSE(BaseLoss):
    def __init__(
        self,
        variable_weights=None,
        period_weights=None,
        horizon_weights=None,
        coordinate_weights=None,
        coordinate_filters=None,
        num_variables=5,
        num_simulations=10
    ):
        """
        RMSE loss function for economic time series with Monte Carlo simulations.
        
        Args:
            variable_weights: Weights for each variable [GDP, Inflation, Consumption, Investment, Interest]
            period_weights: Weights for different starting periods
            horizon_weights: Weights for different forecast horizons
            coordinate_weights: Required by BaseLoss
            coordinate_filters: Required by BaseLoss
            num_variables: Number of economic variables (default: 5)
            num_simulations: Number of Monte Carlo simulations (default: 10)
        """
        super().__init__(coordinate_weights, coordinate_filters)
        
        self.variable_weights = variable_weights if variable_weights else [1.0] * num_variables
        self.period_weights = period_weights if period_weights else [1.0] * 25
        self.horizon_weights = horizon_weights if horizon_weights else [1.0] * 12
        self.num_variables = num_variables
        self.num_simulations = num_simulations
        
    def compute_loss_1d(self, sim_data_ensemble, real_data):
        """Implementation for Black-it to use directly with its ensemble approach"""
        # Skip first time step (initial condition)
        sim_data = sim_data_ensemble[:, 1:]
        real_data = real_data[1:]
        
        # Average across ensemble
        mean_sim = np.mean(sim_data, axis=0)
        
        # Handle NaN values
        valid_indices = ~(np.isnan(mean_sim) | np.isnan(real_data))
        if not np.any(valid_indices):
            return np.inf
        
        # Compute MSE
        squared_errors = (mean_sim[valid_indices] - real_data[valid_indices]) ** 2
        mse = np.mean(squared_errors)
        
        # Apply 100x scaling factor as in paper
        return mse * 100**2
    
    def compute_loss(self, sim_data, real_data):
        """
        Calculate RMSE with time dimension first, handling repeated real data.
        
        Args:
            sim_data: shape (T, N*M*S) - T time steps, N periods, M variables, S simulations
            real_data: shape (T, N*M*R) - T time steps, N periods, M variables, R repetitions
                where R is usually smaller than S and contains repeated values
            
        Returns:
            Weighted RMSE
        """
        # Check for and remove extra first dimension if size is 1
        if len(sim_data.shape) == 3 and sim_data.shape[0] == 1:
            print("Removing extra first dimension from sim_data")
            sim_data = sim_data[0]
        
        if len(real_data.shape) == 3 and real_data.shape[0] == 1:
            print("Removing extra first dimension from real_data")
            real_data = real_data[0]

        # Extract dimensions
        T, NMS = sim_data.shape
        _, NMR = real_data.shape
        
        M = self.num_variables  # Number of variables (5)
        S = self.num_simulations  # Number of simulations (10)
        
        # Calculate N from simulation data
        N = NMS // (M * S)
        
        # Calculate R (repetitions) from real data
        R = NMR // (N * M)
        
        #print(f"Dimensions - T: {T}, N: {N}, M: {M}, S: {S}, R: {R}")
        
        # Check if real data has repetitions as expected
        if R > 1:
            # If real data has repetitions, extract only the first repetition for each variable
            real_data_base = np.zeros((T, N * M))
            for n in range(N):
                for m in range(M):
                    base_idx = n * M + m
                    rep_idx = base_idx * R + 1  # Get the first repetition
                    real_data_base[:, base_idx] = real_data[:, rep_idx-1]  # -1 for 0-indexing
            real_data = real_data_base
        
        # Reshape simulated data to (T, N, M, S) to isolate simulations
        try:
            reshaped_sim = sim_data.reshape(T, N, M, S)
        except ValueError as e:
            print(f"Reshape error. Data size: {sim_data.size}, Target shape: {(T, N, M, S)}")
            print(f"Expected elements: {T*N*M*S}")
            # Try to adjust S if reshape fails
            actual_S = sim_data.size // (T * N * M)
            print(f"Attempting with adjusted S = {actual_S}")
            S = actual_S
            reshaped_sim = sim_data.reshape(T, N, M, S)
        
        # Average over simulations (axis 3)
        avg_sim = np.mean(reshaped_sim, axis=3)  # Shape: (T, N, M)
        
        # Reshape to (T, N*M) to match real_data
        avg_sim_flat = avg_sim.reshape(T, N*M)
        
        # Calculate weighted RMSE
        total_weighted_squared_error = 0.0
        total_weight = 0.0
        
        # For each period
        for n in range(N):
            period_weight = self.period_weights[min(n, len(self.period_weights)-1)]
            
            # For each variable
            for m in range(M):
                var_weight = self.variable_weights[m]
                var_idx = n * M + m
                
                # For each time step (skip t=0 as initial condition)
                for t in range(1, T):
                    horizon_weight = self.horizon_weights[min(t-1, len(self.horizon_weights)-1)]
                    
                    # Get values
                    sim_value = avg_sim_flat[t, var_idx]
                    real_value = real_data[t, var_idx]
                    
                    # Skip NaNs
                    if np.isnan(sim_value) or np.isnan(real_value):
                        continue
                    
                    # Calculate squared error
                    squared_error = (sim_value - real_value) ** 2
                    
                    # Apply weights
                    weight = var_weight * period_weight * horizon_weight
                    total_weighted_squared_error += squared_error * weight
                    total_weight += weight
        
        # Calculate RMSE and apply 100x scaling
        if total_weight > 0:
            rmse = np.sqrt(total_weighted_squared_error / total_weight) * 100
        else:
            rmse = np.inf
            
        return rmse
    
# Variable weights based on the RMSE values in the table
variable_weights = [
    1.0,    # GDP
    3.5,    # Inflation
    1.0,    # Household Consumption
    0.6,    # Investment
    12.0    # Euribor
]

# Initialize the custom loss function
custom_loss = EconomicTimeSeriesRMSE(
    variable_weights=variable_weights,
    period_weights=[1.0] * 25,  # Equal weights for periods
    horizon_weights=[1.0] *5  # Decreasing weights for longer horizons
)

# Calculate the RMSE
rmse_value = custom_loss.compute_loss(result,real_data)