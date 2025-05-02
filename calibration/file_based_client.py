import json
import os
import time
import uuid
from datetime import datetime
import numpy as np

class BeforeITFileClient:
    """Client for communicating with BeforeIT Julia server through files"""
    
    def __init__(self, input_dir="./shared/input", output_dir="./shared/output", timeout=300):
        """Initialize with input/output directories and timeout"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.timeout = timeout
        
        # Ensure directories exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store server ID for logs
        self.server_id = os.path.basename(os.path.dirname(self.input_dir)) if '/' in self.input_dir else "unknown"
    
    def _send_command(self, command_data):
        """Send a command to the Julia server via file system"""
        # Generate a unique ID for this command
        cmd_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create input file path
        input_file = os.path.join(self.input_dir, f"{cmd_id}.json")
        
        # Write command to input file
        with open(input_file, "w") as f:
            json.dump(command_data, f)
        
        print(f"Sent command to {input_file}")
        
        # Wait for completion file
        complete_file = os.path.join(self.output_dir, f"{cmd_id}_complete")
        result_file = os.path.join(self.output_dir, f"{cmd_id}_result.json")
        
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            if os.path.exists(complete_file):
                # Read result
                with open(result_file, "r") as f:
                    result = json.load(f)
                
                # Check for error
                if "error" in result:
                    raise RuntimeError(f"Server error: {result['error']}")
                
                return result
            
            # Sleep a bit to avoid busy waiting
            time.sleep(0.5)
        
        raise TimeoutError(f"Timeout waiting for response after {self.timeout} seconds")
    
    def initialize(self, start_date="2010-03-31", end_date="2013-12-31", 
                 model_type="base", empirical_distribution=False, conditional_forecasts=False):
        """Initialize models in the server"""
        command_data = {
            "command": "initialize",
            "start_date": start_date,
            "end_date": end_date,
            "model_type": model_type,
            "empirical_distribution": empirical_distribution,
            "conditional_forecasts": conditional_forecasts
        }
        
        print(f"Initializing models on server {self.server_id}")
        return self._send_command(command_data)
    
    def get_data(self, start_date="2010-03-31", end_date="2013-12-31", repetitions=10):
        """Get real data for calibration"""
        command_data = {
            "command": "get_data",
            "start_date": start_date,
            "end_date": end_date,
            "repetitions": repetitions
        }
        
        print(f"Getting data from server {self.server_id}")
        result = self._send_command(command_data)
        
        if "data" not in result:
            raise RuntimeError("Server response missing 'data' field")
        
        return result["data"]
    
    def run_simulation(self, params, start_date="2010-03-31", end_date="2013-12-31", 
                     model_type="extended_heuristic", num_simulations=10, abmx=False, 
                     conditional_forecasts=False, version_c=1, multi_threading=True):
        """Run a simulation with given parameters"""
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(params, np.ndarray):
            params = params.tolist()
        
        command_data = {
            "command": "run_simulation",
            "params": params,
            "start_date": start_date,
            "end_date": end_date,
            "model_type": model_type,
            "num_simulations": num_simulations,
            "abmx": abmx,
            "conditional_forecasts": conditional_forecasts,
            "version_c": version_c,
            "multi_threading": multi_threading
        }
        
        print(f"Running simulation on server {self.server_id} with parameters: {params}")
        result = self._send_command(command_data)
        
        if "result" not in result:
            raise RuntimeError("Server response missing 'result' field")
        
        return result["result"]


class ServerMonitor:
    """Simple monitor for distributing work across multiple BeforeIT servers"""
    
    def __init__(self, num_servers, base_dir="./shared_data_"):
        """Initialize with the number of servers and base directory"""
        self.num_servers = num_servers
        self.base_dir = base_dir
        self.next_server = 0
        self.clients = []
        
        # Initialize clients for each server
        for i in range(num_servers):
            server_num = i + 1
            input_dir = f"{base_dir}{server_num}/input"
            output_dir = f"{base_dir}{server_num}/output"
            
            # Create directories if they don't exist
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create client
            client = BeforeITFileClient(input_dir=input_dir, output_dir=output_dir)
            self.clients.append(client)
        
        print(f"Initialized ServerMonitor with {num_servers} servers")
    
    def is_server_busy(self, server_idx):
        """Check if a server is busy by looking for input files"""
        server_num = server_idx + 1
        input_dir = f"{self.base_dir}{server_num}/input"
        
        # Look for any JSON files in the input directory
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        
        # If there are any input files, the server is busy
        is_busy = len(input_files) > 0
        
        if is_busy:
            print(f"Server {server_num} is busy")
        
        return is_busy
    
    def get_next_server(self):
        """Get the next available server using round-robin with input file check"""
        # Try each server in round-robin order
        for i in range(self.num_servers):
            # Calculate the server index in round-robin order
            server_idx = (self.next_server + i) % self.num_servers
            
            # Check if this server is busy
            if not self.is_server_busy(server_idx):
                # Server is available, update next server and return this one
                self.next_server = (server_idx + 1) % self.num_servers
                print(f"Selected server {server_idx + 1}")
                return server_idx
        
        # All servers are busy, just use the next one in sequence
        server_idx = self.next_server
        self.next_server = (server_idx + 1) % self.num_servers
        print(f"All servers are busy, using server {server_idx + 1}")
        return server_idx
    
    def initialize_all_servers(self, start_date, end_date, **kwargs):
        """Initialize all servers with the same parameters"""
        print(f"Initializing all servers from {start_date} to {end_date}")
        
        # Initialize each server
        for i in range(self.num_servers):
            server_num = i + 1
            print(f"Initializing server {server_num}")
            
            try:
                self.clients[i].initialize(
                    start_date=start_date,
                    end_date=end_date,
                    **kwargs
                )
                print(f"Server {server_num} initialized successfully")
            except Exception as e:
                print(f"Error initializing server {server_num}: {e}")
    
    def run_simulation(self, params, start_date, end_date, version_c=2, **kwargs):
        """Run a simulation on the next available server"""
        # Get the next available server
        server_idx = self.get_next_server()
        server_num = server_idx + 1
        
        # Convert params to list if it's a numpy array
        if isinstance(params, np.ndarray):
            params = params.tolist()
        
        print(f"Running simulation on server {server_num} with parameters {params}")
        
        # Run simulation on this server
        result = np.array(self.clients[server_idx].run_simulation(
            params,
            start_date=start_date,
            end_date=end_date,
            version_c=version_c,
            **kwargs
        ))
        
        return result


# Example usage
if __name__ == "__main__":
    # Test with a single server
    client = BeforeITFileClient(
        input_dir="./shared_data_1/input",
        output_dir="./shared_data_1/output"
    )
    
    # Initialize models
    client.initialize()
    
    # Get real data
    real_data = np.array(client.get_data(repetitions=10))
    print(f"Real data shape: {real_data.shape}")
    
    # Run a simulation
    result = client.run_simulation([0.9, 0.5], version_c=2)
    print(f"Simulation result shape: {np.array(result).shape}")
    
    # Test with multiple servers
    monitor = ServerMonitor(num_servers=4)
    
    # Initialize all servers
    monitor.initialize_all_servers(
        start_date="2010-03-31",
        end_date="2013-12-31"
    )
    
    # Run some simulations using different servers
    for i in range(8):
        params = [0.8 + i*0.1, 0.5 + i*0.05]
        result = monitor.run_simulation(
            params,
            start_date="2010-03-31",
            end_date="2013-12-31",
            version_c=2
        )
        print(f"Result shape: {result.shape}")