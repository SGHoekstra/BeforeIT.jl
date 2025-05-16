import os
import sys
import time
import numpy as np
import torch
from torch import nn
from pathlib import Path
import json
import pickle
import warnings

# sbi4abm imports
from sbi4abm.sbi import inference as sbi_inference
from sbi4abm.sbi import utils as sbi_utils
from sbi4abm.sbi.inference import prepare_for_sbi, simulate_for_sbi

# Local imports for BeforeIT interaction
# Ensure 'calibration' directory is in PYTHONPATH or in the same directory
try:
    from calibration.beforeit_http_client import ServerMonitor
except ImportError:
    # Try adding the parent directory of 'calibration' to sys.path
    # This assumes the script is run from a directory where 'calibration' is a subdir
    # or 'calibration' is in a location findable by this adjustment.
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir.parent))
    try:
        from calibration.beforeit_http_client import ServerMonitor
    except ImportError:
        raise ImportError("Could not import ServerMonitor. Ensure 'calibration.beforeit_http_client' is accessible.")


# Local imports for network definitions
from sbi4abm.networks import time_series

# User's utils for saving (from their Black-it example)
from sbi4abm.utils import io as user_io


# --- Configuration ---
num_servers = int(os.environ.get('NUM_SERVERS', 4))
batch_size_sbi_training = 50
simulation_batch_size_sbi_call = 5 # How many thetas one sbi worker processes before returning

bounds_np = np.array([[0.0, 0.0, 0.0], [1.1, 1.0, 1.0]])
start_date_sim = "2010-03-31"
end_date_sim = "2013-12-31"
num_simulations_sbi = int(os.environ.get('NUM_SIMULATIONS_SBI', 100)) # Total simulations for SBI
num_posterior_samples = 5000
saving_folder = f'sbi_calibration_union_http_output'
os.makedirs(saving_folder, exist_ok=True)

# Number of Monte Carlo runs per parameter set for the ABM
# This should match how real_data and simulated data are generated/processed
num_mc_sims_per_theta = 10

# Server URLs
server_urls = None
config_file_path = os.environ.get('CONFIG_FILE', 'beforeit_config.json')
if os.path.exists(config_file_path):
    with open(config_file_path, 'r') as f:
        config = json.load(f)
        if 'server_urls' in config:
            server_urls = config['server_urls']
            print(f"Loaded {len(server_urls)} server URLs from config file: {server_urls}")

if not server_urls:
    server_urls = [f"http://localhost:{8080 + i}" for i in range(num_servers)]
    print(f"Using default server URLs: {server_urls}")

print(f"Starting SBI calibration for BeforeIT.jl model (Union variant)")
print(f"Using {num_servers} servers via HTTP: {server_urls}")
print(f"Parameter bounds: {bounds_np.tolist()}")
print(f"Simulation period: {start_date_sim} to {end_date_sim}")
print(f"Number of SBI simulations: {num_simulations_sbi}")
print(f"Saving results to: {saving_folder}")

# --- Initialize ServerMonitor ---
monitor = ServerMonitor(server_urls=server_urls)

# Initialize all servers
try:
    print("Initializing all BeforeIT.jl servers...")
    monitor.initialize_all_servers(
        start_date=start_date_sim,
        end_date=end_date_sim
        # model_type="union" # If your initialize supports this
    )
    print("All servers initialized successfully.")
except Exception as e:
    print(f"Error initializing servers: {e}")
    sys.exit(1)

# --- Get Real Data (Observation x_o) ---
print("Getting real data for calibration...")
try:
    client = monitor.clients[0]
    real_data_MTR = np.array(client.get_data(
        start_date=start_date_sim,
        end_date=end_date_sim,
        repetitions=num_mc_sims_per_theta 
    ))
    # Transpose to (T, M) for SBI/RNN convention
    real_data_avg_TM = real_data_MTR.T 
    x_o = torch.from_numpy(real_data_avg_TM).float()
    print(f"Real data (x_o) shape after averaging repetitions and transposing: {x_o.shape}")
except Exception as e:
    print(f"Error getting real data: {e}")
    sys.exit(1)

# --- Define SBI Simulator Wrapper ---
def sbi_simulator_wrapper(params_tensor: torch.Tensor) -> torch.Tensor:
    """
    SBI-compatible simulator wrapper for BeforeIT.jl.
    Args:
        params_tensor: A batch of parameters (torch.Tensor, shape: [batch_dim, num_params]).
    Returns:
        A batch of simulation outputs (torch.Tensor, shape: [batch_dim, num_timesteps, num_variables]).
    """
    sim_results_list = []
    for i in range(params_tensor.shape[0]):
        single_param_set_np = params_tensor[i,:].numpy()
        try:
            sim_output_MTS = np.array(monitor.run_simulation(
                params=single_param_set_np.tolist(),
                start_date=start_date_sim,
                end_date=end_date_sim,
                num_simulations=num_mc_sims_per_theta,
                abmx=True
            ))
            sim_output_avg_TM = sim_output_MTS.T 
            sim_results_list.append(torch.from_numpy(sim_output_avg_TM).float())
        except Exception as e:
            print(f"Error in sbi_simulator_wrapper for param set {single_param_set_np}: {e}")
            # Return a tensor of NaNs of the expected shape if simulation fails
            dummy_shape = x_o.shape # (T, M)
            sim_results_list.append(torch.full(dummy_shape, float('nan')))
    
    stacked_sims = torch.stack(sim_results_list) # Shape (B_sbi, T, M)
    # print(f"Simulator returning batch of shape: {stacked_sims.shape}")
    return stacked_sims

# --- Define Prior ---
prior_min = torch.tensor(bounds_np[0,:], dtype=torch.float32)
prior_max = torch.tensor(bounds_np[1,:], dtype=torch.float32)
# Ensure prior is on CPU by default, can be moved later if needed
prior = sbi_utils.BoxUniform(low=prior_min, high=prior_max, device='cpu')
print(f"Prior defined: BoxUniform with bounds {prior_min.tolist()} to {prior_max.tolist()}")

# --- Prepare Simulator and Prior for SBI ---
simulator_sbi, prior_sbi = prepare_for_sbi(sbi_simulator_wrapper, prior)
print("Simulator and prior prepared for SBI.")

# --- Define Embedding Network and Density Estimator ---
# x_o has shape (T, M)
# simulated x for a single theta also has shape (T, M)
num_timesteps = x_o.shape[0]  # T
num_features_per_ts = x_o.shape[1] # M (number of economic variables)

# Embedding network: GRU for time series
# Input to RNN: (batch_size, sequence_length, input_features_dim)
# Here, sequence_length = num_timesteps, input_features_dim = num_features_per_ts
embedding_net = time_series.RNN(
    input_dim=num_features_per_ts, # M
    hidden_dim=64,        # Example hidden dimension for GRU
    num_layers=2,         # Example number of GRU layers
    mlp_dims=[32, 16],    # Example MLP head for the embedding output
    flavour="gru"
)
print(f"Using GRU embedding net with input_dim={num_features_per_ts}, output_dim_mlp_head={embedding_net.final.out_features}")

# Density estimator for SNPE (e.g., MAF)
# `z_score_x=True` for posterior_nn means context `x` is z-scored before embedding_net
# `z_score_theta=True` means `theta` is z-scored before the flow
z_score_x_for_embedding = True # Based on 'gru' flavour in inference/neural.py
density_estimator_build_fn = sbi_utils.posterior_nn(
    model='maf',
    z_score_theta=True,
    z_score_x=z_score_x_for_embedding,
    embedding_net=embedding_net
)

# --- SBI Inference ---
# SNPE_C is the default for SNPE alias in sbi4abm
inference_method = sbi_inference.SNPE(
    prior=prior_sbi,
    density_estimator=density_estimator_build_fn,
    device='cpu' # Can be 'cuda' if GPU is available and desired
)
print(f"Using SBI method: SNPE (likely SNPE_C)")

# Simulate data for training
print(f"Generating {num_simulations_sbi} simulations for SBI training...")
# num_workers for simulate_for_sbi should ideally match num_servers for ABM
theta_train, x_train = simulate_for_sbi(
    simulator=simulator_sbi,
    proposal=prior_sbi,
    num_simulations=num_simulations_sbi,
    num_workers=min(num_servers, os.cpu_count() // 2 or 1), # Avoid oversubscribing CPUs
    simulation_batch_size=simulation_batch_size_sbi_call
)
print(f"Generated theta_train shape: {theta_train.shape}, x_train shape: {x_train.shape}")

# Train the density estimator
# The z_score_x in train() is the one mentioned in the README for NPE/NRE.
# It controls z-scoring of 'x' data when fed to the loss during training.
# It should be consistent with how the network expects its input (set by z_score_x in posterior_nn).
print("Training the density estimator...")
start_train_time = time.time()
density_estimator_trained = inference_method.append_simulations(theta_train, x_train).train(
    training_batch_size=batch_size_sbi_training,
    z_score_x=z_score_x_for_embedding, # Matches how embedding_net input is treated
    show_train_summary=True,
    max_num_epochs= int(os.environ.get('MAX_EPOCHS_SBI', 200)) # Allow more epochs for potentially complex data
)
end_train_time = time.time()
print(f"Density estimator training finished in {end_train_time - start_train_time:.2f} seconds.")

# Build the posterior
posterior = inference_method.build_posterior(density_estimator_trained)
print("Posterior built.")

# Prepare observed data for sampling (add batch dimension and move to device)
x_o_for_sampling = x_o.unsqueeze(0).to(posterior._device)

# Sample from the posterior
print(f"Sampling {num_posterior_samples} from the posterior for x_o...")
start_sample_time = time.time()
# Default MCMC method for SNPE is rejection sampling if leakage is low,
# otherwise MCMC (slice_np).
# We can explicitly choose MCMC if rejection is too slow or has issues.
# For complex posteriors, MCMC might be more robust.
posterior_samples = posterior.sample(
    (num_posterior_samples,),
    x=x_o_for_sampling,
    show_progress_bars=True,
    # sample_with="mcmc", # Optionally force MCMC
    # mcmc_method="slice_np_vectorized" # Or another MCMC method
)
posterior_samples_np = posterior_samples.cpu().numpy()
end_sample_time = time.time()
print(f"Posterior sampling finished in {end_sample_time - start_sample_time:.2f} seconds.")
print(f"Generated posterior samples shape: {posterior_samples_np.shape}")

# --- Save Results ---
# Using the user's io.save_output for consistency with their Black-it example
# It saves posterior to 'posteriors.pkl' and samples to 'samples.txt' in outloc
# user_io.save_output expects a list of posteriors
user_io.save_output([posterior], posterior_samples_np, None, saving_folder)
print(f"SBI posterior and samples saved to '{saving_folder}' using user_io.save_output.")

# Additionally, save observed data for reference
np.save(Path(saving_folder) / "observed_data_x_o.npy", x_o.cpu().numpy())
# If true_theta were known (e.g. for synthetic data tests):
# np.save(Path(saving_folder) / "true_theta.npy", true_theta_np)

print(f"SBI calibration process completed. Results in {saving_folder}")

# --- Optional: Plotting ---
try:
    from sbi4abm.sbi import analysis
    import matplotlib.pyplot as plt

    param_names = ["theta_UNION", "phi_DP", "phi_F_Q"]

    fig, axes = analysis.pairplot(
        posterior_samples_np,
        limits=bounds_np.T.tolist(),
        labels=param_names,
        figsize=(8,8)
    )
    fig.suptitle("SBI Posterior Samples for BeforeIT.jl (Union Model)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = Path(saving_folder) / "sbi_posterior_pairplot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Pairplot of posterior samples saved to {plot_path}")
    plt.close(fig)
except ImportError:
    warnings.warn("Matplotlib or other plotting libraries not fully available. Skipping plots.")
except Exception as e:
    warnings.warn(f"Error during plotting: {e}")