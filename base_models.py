from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
import matplotlib.pyplot as plt
from function import f
from get_fitted_model import get_fitted_model
import os
import torch

from plotting_gp import plotting_gp
SMOKE_TEST=os.environ.get("SMOKE_TEST")

# suppress GPyTorch warnings about adding jitter
import warnings
warnings.filterwarnings("ignore", "^.*jitter.*", category=RuntimeWarning)

def create_base_models(shift_c,BOUNDS, noise_std,dtype,device):
    best_rgpe_all = []
    best_random_all = []
    best_vanilla_nei_all = []
    N_BATCH = 10 if not SMOKE_TEST else 2
    NUM_POSTERIOR_SAMPLES = 128 if not SMOKE_TEST else 16
    RANDOM_INITIALIZATION_SIZE = 3
    MC_SAMPLES = 64 if not SMOKE_TEST else 32
    N_RESTART_CANDIDATES = 256 if not SMOKE_TEST else 8
    N_RESTARTS = 10 if not SMOKE_TEST else 2
    Q_BATCH_SIZE = 1

    # Fit base model
    base_model_list = []
    for j,task in enumerate(shift_c):
        print(f"Fitting base model {j},",task)
        # Average over multiple trials
        best_vanilla_nei = []
        # Initial random observations
        raw_x = draw_sobol_samples(bounds=BOUNDS, n=RANDOM_INITIALIZATION_SIZE, q=1, seed=278+j).squeeze(1)    
        train_x = normalize(raw_x, bounds=BOUNDS)
        #train_x=raw_x
        train_y_noiseless = f(raw_x,task)
        train_y = train_y_noiseless + noise_std*torch.randn_like(train_y_noiseless)
        train_yvar = torch.full_like(train_y, noise_std**2)
        vanilla_nei_train_x = train_x.clone()
        vanilla_nei_train_y = train_y.clone()
        vanilla_nei_train_yvar = train_yvar.clone()
        # keep track of the best observed point at each iteration
        best_value = train_y.max().item()
        vanilla_nei_best_value = best_value
        best_vanilla_nei.append(vanilla_nei_best_value)
        # Run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(N_BATCH): 
            #print(best_value)
            # Run Vanilla NEI for comparison
            vanilla_nei_model = get_fitted_model(
                vanilla_nei_train_x, 
                vanilla_nei_train_y, 
                vanilla_nei_train_yvar,
            )
            vanilla_nei_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
            vanilla_qNEI = qNoisyExpectedImprovement(
                model=vanilla_nei_model, 
                X_baseline=vanilla_nei_train_x,
                sampler=vanilla_nei_sampler,
            )
            vanilla_nei_candidate, _ = optimize_acqf(
                acq_function=vanilla_qNEI,
                bounds=torch.tensor([[0.],[1.]], dtype=dtype, device=device),
                #bounds=BOUNDS,
                q=Q_BATCH_SIZE,
                num_restarts=N_RESTARTS,
                raw_samples=N_RESTART_CANDIDATES,
            )
            # fetch the new values 
            vanilla_nei_new_x = vanilla_nei_candidate.detach()
            vanilla_nei_new_y_noiseless = f(unnormalize(vanilla_nei_new_x, bounds=BOUNDS),task)
            #vanilla_nei_new_y_noiseless = f(vanilla_nei_new_x)
            vanilla_nei_new_y = vanilla_nei_new_y_noiseless + noise_std*torch.randn_like(vanilla_nei_new_y_noiseless)
            vanilla_nei_new_yvar = torch.full_like(vanilla_nei_new_y, noise_std**2)

            # update training points
            vanilla_nei_train_x = torch.cat([vanilla_nei_train_x, vanilla_nei_new_x])
            vanilla_nei_train_y = torch.cat([vanilla_nei_train_y, vanilla_nei_new_y])
            vanilla_nei_train_yvar = torch.cat([vanilla_nei_train_yvar, vanilla_nei_new_yvar])
            # get the new best observed value
            vanilla_nei_best_value = vanilla_nei_train_y.max().item()
            best_vanilla_nei.append(vanilla_nei_best_value)
        # plt.plot(vanilla_nei_train_x.cpu().numpy(), vanilla_nei_train_y.cpu().detach(),'.')
        # plt.show()
        vanilla_nei_model_final = get_fitted_model(vanilla_nei_train_x, 
                            vanilla_nei_train_y, 
                            vanilla_nei_train_yvar,
                            )
        plotting_gp(vanilla_nei_model_final,vanilla_nei_train_x,                            vanilla_nei_train_y, 
                            vanilla_nei_train_yvar,device,BOUNDS)
        base_model_list.append(vanilla_nei_model_final) 
    return base_model_list 