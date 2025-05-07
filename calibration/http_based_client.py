import json
import numpy as np
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from beforeit_http_client import BeforeITHttpClient, ServerMonitor

class CalibrationRunner:
    """Runs calibration using the BeforeIT HTTP API"""
    
    def __init__(self, config_file=None):
        """Initialize with a config file or default values"""
        # Default configuration
        self.config = {
            "server_urls": ["http://localhost:8080", "http://localhost:8081", 
                          "http://localhost:8082", "http://localhost:8083"],
            "version": 2,
            "num_calibration": 10,
            "start_date": "2010-03-31",
            "end_date": "2013-12-31"
        }
        
        # Override with config file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self.config.update(file_config)
        
        # Initialize server monitor
        self.monitor = ServerMonitor(server_urls=self.config["server_urls"])
        
        # Set up directories
        os.makedirs("results", exist_ok=True)
        
        print(f"Initialized CalibrationRunner with {len(self.config['server_urls'])} servers")
        print(f"Version: {self.config['version']}")
        print(f"Calibration iterations: {self.config['num_calibration']}")
    
    def initialize_all_servers(self):
        """Initialize all servers with the same parameters"""
        print("Initializing all servers...")
        
        self.monitor.initialize_all_servers(
            start_date=self.config["start_date"],
            end_date=self.config["end_date"]
        )
        
        print("All servers initialized.")
    
    def run_calibration(self):
        """Run calibration for the specified parameters"""
        version = self.config["version"]
        num_calibration = self.config["num_calibration"]
        
        start_time = time.time()
        
        # Generate parameter combinations based on version
        if version == 1:
            # Version 1 calibration parameters
            param_set = self._generate_version1_params(num_calibration)
        else:
            # Version 2 calibration parameters (default)
            param_set = self._generate_version2_params(num_calibration)
        
        print(f"Running {len(param_set)} parameter combinations...")
        
        # Run simulations in parallel using ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor(max_workers=len(self.config["server_urls"])) as executor:
            futures = []
            
            for params in param_set:
                # Submit task to thread pool
                future = executor.submit(
                    self._run_simulation_with_retry,
                    params
                )
                futures.append((params, future))
            
            # Collect results as they complete
            for params, future in futures:
                try:
                    result = future.result()
                    results.append({
                        "params": params.tolist() if isinstance(params, np.ndarray) else params,
                        "result": result.tolist() if isinstance(result, np.ndarray) else result
                    })
                    print(f"Completed simulation for params: {params}")
                except Exception as e:
                    print(f"Error running simulation for params {params}: {e}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Calibration completed in {elapsed_time:.2f} seconds")
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = f"results/calibration_v{version}_{timestamp}.json"
        
        with open(result_file, 'w') as f:
            json.dump({
                "version": version,
                "num_calibration": num_calibration,
                "elapsed_time": elapsed_time,
                "results": results
            }, f, indent=2)
        
        print(f"Results saved to {result_file}")
        
        return results
    
    def _run_simulation_with_retry(self, params, max_retries=3, retry_delay=5):
        """Run a simulation with retry logic"""
        retries = 0
        while retries <= max_retries:
            try:
                result = self.monitor.run_simulation(
                    params=params,
                    start_date=self.config["start_date"],
                    end_date=self.config["end_date"]
                )
                return result
            except Exception as e:
                retries += 1
                if retries <= max_retries:
                    print(f"Retry {retries}/{max_retries} for params {params} after error: {e}")
                    time.sleep(retry_delay)
                else:
                    print(f"Max retries reached for params {params}")
                    raise
    
    def _generate_version1_params(self, num_samples):
        """Generate parameters for version 1 calibration"""
        # Simple parameter generation - single parameter varying from 0.5 to 1.0
        return np.linspace(0.5, 1.0, num_samples)
    
    def _generate_version2_params(self, num_samples):
        """Generate parameters for version 2 calibration"""
        # More complex parameter space with two dimensions
        # First parameter varies from 0.7 to 0.95
        # Second parameter varies from 0.3 to 0.7
        params = []
        
        # Create a grid of parameters
        p1_values = np.linspace(0.7, 0.95, int(np.sqrt(num_samples)))
        p2_values = np.linspace(0.3, 0.7, int(np.sqrt(num_samples)))
        
        for p1 in p1_values:
            for p2 in p2_values:
                params.append([p1, p2])
        
        return params[:num_samples]  # Limit to requested number of samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BeforeIT calibration with HTTP servers")
    parser.add_argument("--config", default=os.environ.get("CONFIG_FILE", "beforeit_config.json"),
                      help="Path to configuration file")
    args = parser.parse_args()
    
    # Create and run calibration
    runner = CalibrationRunner(config_file=args.config)
    runner.initialize_all_servers()
    runner.run_calibration()