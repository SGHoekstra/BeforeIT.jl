import json
import time
import uuid
from datetime import datetime
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

class BeforeITHttpClient:
    """Client for communicating with BeforeIT Julia server through HTTP"""
    
    def __init__(self, server_url="http://localhost:8080", timeout=300):
        """Initialize with server URL and timeout"""
        self.server_url = server_url
        self.timeout = timeout
        
        # Store server ID for logs
        self.server_id = server_url.split("://")[1]
    
    def _send_command(self, endpoint, command_data):
        """Send a command to the Julia server via HTTP"""
        # Generate a unique ID for this command
        cmd_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Add command ID to the data
        command_data["cmd_id"] = cmd_id
        
        # Construct the full URL
        url = f"{self.server_url}/{endpoint}"
        
        print(f"Sending command to {url}")
        
        try:
            # Send the HTTP request
            response = requests.post(
                url,
                json=command_data,
                timeout=self.timeout
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the JSON response
            result = response.json()
            
            # Check for error
            if "error" in result:
                raise RuntimeError(f"Server error: {result['error']}")
            
            return result
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HTTP request failed: {str(e)}")
    
    def initialize(self, start_date="2010-03-31", end_date="2013-12-31", 
                  model_type="base", empirical_distribution=False, conditional_forecasts=False):
        """Initialize models in the server"""
        command_data = {
            "start_date": start_date,
            "end_date": end_date,
            "model_type": model_type,
            "empirical_distribution": empirical_distribution,
            "conditional_forecasts": conditional_forecasts
        }
        
        print(f"Initializing models on server {self.server_id}")
        return self._send_command("initialize", command_data)
    
    def get_data(self, start_date="2010-03-31", end_date="2013-12-31", repetitions=10):
        """Get real data for calibration"""
        command_data = {
            "start_date": start_date,
            "end_date": end_date,
            "repetitions": repetitions
        }
        
        print(f"Getting data from server {self.server_id}")
        result = self._send_command("get_data", command_data)
        
        if "data" not in result:
            raise RuntimeError("Server response missing 'data' field")
        
        return result["data"]
    
    def run_simulation(self, params, start_date="2010-03-31", end_date="2013-12-31", 
                       num_simulations=10, multi_threading=True):
        """Run a simulation with given parameters"""
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(params, np.ndarray):
            params = params.tolist()
        
        command_data = {
            "params": params,
            "start_date": start_date,
            "end_date": end_date,
            "num_simulations": num_simulations,
            "multi_threading": multi_threading
        }
        
        print(f"Running simulation on server {self.server_id} with parameters: {params}")
        result = self._send_command("run_simulation", command_data)
        
        if "result" not in result:
            raise RuntimeError("Server response missing 'result' field")
        
        return result["result"]

    def healthcheck(self):
        """Check if the server is alive and ready"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


class ServerMonitor:
    """Monitor for distributing work across multiple BeforeIT HTTP servers"""
    
    def __init__(self, server_urls=None):
        """Initialize with a list of server URLs"""
        if server_urls is None:
            # Default to 4 servers on localhost with different ports
            server_urls = [f"http://localhost:{8080 + i}" for i in range(4)]
        
        self.server_urls = server_urls
        self.num_servers = len(server_urls)
        self.next_server = 0
        self.clients = []
        
        # Initialize clients for each server
        for url in server_urls:
            client = BeforeITHttpClient(server_url=url)
            self.clients.append(client)
        
        print(f"Initialized ServerMonitor with {self.num_servers} servers")
    
    def is_server_healthy(self, server_idx):
        """Check if a server is healthy by pinging its health endpoint"""
        return self.clients[server_idx].healthcheck()
    
    def get_next_server(self):
        """Get the next available server using round-robin with health check"""
        # Try each server in round-robin order
        for i in range(self.num_servers):
            # Calculate the server index in round-robin order
            server_idx = (self.next_server + i) % self.num_servers
            
            # Check if this server is healthy
            if self.is_server_healthy(server_idx):
                # Server is available, update next server and return this one
                self.next_server = (server_idx + 1) % self.num_servers
                print(f"Selected server {server_idx} ({self.server_urls[server_idx]})")
                return server_idx
        
        # All servers are unavailable
        raise RuntimeError("No healthy servers available")
    
    def initialize_all_servers(self, start_date, end_date, **kwargs):
        """Initialize all servers with the same parameters"""
        print(f"Initializing all servers from {start_date} to {end_date}")
        
        # Initialize each server in parallel
        with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            futures = []
            
            for i in range(self.num_servers):
                print(f"Initializing server {i} ({self.server_urls[i]})")
                
                # Submit initialization task
                future = executor.submit(
                    self.clients[i].initialize,
                    start_date=start_date,
                    end_date=end_date,
                    **kwargs
                )
                
                futures.append((i, future))
            
            # Wait for all initializations and collect results
            for i, future in futures:
                try:
                    future.result()
                    print(f"Server {i} ({self.server_urls[i]}) initialized successfully")
                except Exception as e:
                    print(f"Error initializing server {i} ({self.server_urls[i]}): {e}")
    
    def run_simulation(self, params, start_date, end_date, **kwargs):
        """Run a simulation on the next available server"""
        # Get the next available server
        server_idx = self.get_next_server()
        
        # Convert params to list if it's a numpy array
        if isinstance(params, np.ndarray):
            params = params.tolist()
        
        # Ensure params is a list
        if not isinstance(params, list):
            params = [params]

        print(f"Running simulation on server {server_idx} ({self.server_urls[server_idx]}) with parameters {params}")
        
        # Run simulation on this server
        result = np.array(self.clients[server_idx].run_simulation(
            params,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        ))
        
        return result


# Example usage
if __name__ == "__main__":
    # Test with a single server
    client = BeforeITHttpClient(server_url="http://localhost:8080")
    
    # Check if server is available
    if not client.healthcheck():
        print("Server is not available")
        exit(1)
    
    # Initialize models
    client.initialize()
    
    # Get real data
    real_data = np.array(client.get_data(repetitions=10))
    print(f"Real data shape: {real_data.shape}")
    
    # Run a simulation
    result = client.run_simulation([0.9])
    print(f"Simulation result shape: {np.array(result).shape}")
    
    # Test with multiple servers
    monitor = ServerMonitor(server_urls=[
        "http://localhost:8080",
        "http://localhost:8081",
        "http://localhost:8082",
        "http://localhost:8083"
    ])
    
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
            end_date="2013-12-31"
        )
        print(f"Result shape: {result.shape}")