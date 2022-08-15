import torch
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize

from plotting_gp import plotting_gp
from get_fitted_model import get_fitted_model
from function import f
from computing_weights import compute_rank_weights
from base_models import create_base_models
from RGPE_model import RGPE

# suppress GPyTorch warnings about adding jitter
import warnings

torch.manual_seed(60)
warnings.filterwarnings("ignore", "^.*jitter.*", category=RuntimeWarning)
def main():
    dtype=torch.double
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BOUNDS = torch.tensor([[0], [4]], dtype=dtype, device=device)
    SMOKE_TEST = os.environ.get("SMOKE_TEST")
    shift_c=shift_c=[-1.4,-1,-0.7,1,1.5]
    noise_std=0.05
    rank_weights_all=[]   
    best_rgpe_all = []
    inter_best_rgpe_all=[]
    best_random_all = []
    best_vanilla_nei_all = []
    train_x_all=[]
    vanilla_nei_train_x_all=[]
    train_y_all=[]
    vanilla_nei_train_y_all=[]
    N_BATCH = 10 if not SMOKE_TEST else 2
    NUM_POSTERIOR_SAMPLES = 128 if not SMOKE_TEST else 16
    RANDOM_INITIALIZATION_SIZE = 3
    N_TRIALS = 1 if not SMOKE_TEST else 2
    MC_SAMPLES = 256 if not SMOKE_TEST else 32
    N_RESTART_CANDIDATES = 256 if not SMOKE_TEST else 8
    N_RESTARTS = 10 if not SMOKE_TEST else 2
    Q_BATCH_SIZE = 1
    inter=4
    base_model_list=create_base_models(shift_c,BOUNDS, noise_std,dtype,device)
    # Average over multiple trials
    for trial in range(1,N_TRIALS+1):
        print(f"Trial {trial} of {N_TRIALS}")
        best_rgpe = []
        inter_best_rgpe=[]
        best_random = [] 
        best_vanilla_nei = []
        ranking_weights=[]
        # Initial random observations
        raw_x = draw_sobol_samples(bounds=BOUNDS, n=RANDOM_INITIALIZATION_SIZE, q=1, seed=trial+72).squeeze(1)    
        train_x = normalize(raw_x, bounds=BOUNDS)
        #train_x=raw_x
        train_y_noiseless = f(raw_x) 
        train_y = train_y_noiseless + noise_std*torch.randn_like(train_y_noiseless)
        train_yvar = torch.full_like(train_y, noise_std**2)
        vanilla_nei_train_x = train_x.clone()
        vanilla_nei_train_y = train_y.clone()
        vanilla_nei_train_yvar = train_yvar.clone()
        # keep track of the best observed point at each iteration
        best_value = train_y.max().item()
        best_rgpe.append(best_value)
        best_random.append(best_value)
        vanilla_nei_best_value = best_value
        best_vanilla_nei.append(vanilla_nei_best_value)

        # Run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(N_BATCH): 
            target_model = get_fitted_model(train_x, train_y, train_yvar)
            model_list = base_model_list + [target_model]
            rank_weights = compute_rank_weights(
                train_x, 
                train_y,
                train_yvar, 
                base_model_list, 
                target_model, 
                NUM_POSTERIOR_SAMPLES,
                device
            )
            ranking_weights.append(rank_weights)
            #print(rank_weights.shape,"rshape")
            # create model and acquisition function
            rgpe_model = RGPE(model_list, rank_weights)
            sampler_qnei = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
            qNEI = qNoisyExpectedImprovement(
            model=rgpe_model, 
            X_baseline=train_x,
            sampler=sampler_qnei,

            )
            
            # optimize
            candidate, _ = optimize_acqf(
                acq_function=qNEI,
                bounds=torch.tensor([[0.],[1.]], dtype=dtype, device=device),
                #bounds=BOUNDS,
                q=Q_BATCH_SIZE,
                num_restarts=N_RESTARTS,
                raw_samples=N_RESTART_CANDIDATES,

            )
            #print(candidate,"c")
            # fetch the new values 
            new_x = candidate.detach()
            new_y_noiseless = f(unnormalize(new_x, bounds=BOUNDS))
            #new_y_noiseless=f(new_x)
            new_y = new_y_noiseless + noise_std*torch.randn_like(new_y_noiseless)
            new_yvar = torch.full_like(new_y, noise_std**2)

            # update training points
            train_x = torch.cat((train_x, new_x))
            train_y = torch.cat((train_y, new_y))
            train_yvar = torch.cat((train_yvar, new_yvar))
            
            # get the new best observed value
            best_value = train_y.max().item()
            best_rgpe.append(best_value)
            #print(best_value)
            plotting_gp(target_model,train_x,train_y,train_yvar,device,BOUNDS)

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
            vanilla_nei_new_y_noiseless = f(unnormalize(vanilla_nei_new_x, bounds=BOUNDS))
            #vanilla_nei_new_y_noiseless=f(vanilla_nei_new_x)
            vanilla_nei_new_y = vanilla_nei_new_y_noiseless + noise_std*torch.randn_like(new_y_noiseless)
            vanilla_nei_new_yvar = torch.full_like(vanilla_nei_new_y, noise_std**2)

            # update training points
            vanilla_nei_train_x = torch.cat([vanilla_nei_train_x, vanilla_nei_new_x])
            vanilla_nei_train_y = torch.cat([vanilla_nei_train_y, vanilla_nei_new_y])
            vanilla_nei_train_yvar = torch.cat([vanilla_nei_train_yvar, vanilla_nei_new_yvar])

            # get the new best observed value
            vanilla_nei_best_value = vanilla_nei_train_y.max().item()
            best_vanilla_nei.append(vanilla_nei_best_value)

        # vanilla_nei_model_final = get_fitted_model(vanilla_nei_train_x, 
        #                     vanilla_nei_train_y, 
        #                     vanilla_nei_train_yvar,
        #                     )
        # plotting_gp(vanilla_nei_model_final,vanilla_nei_train_x, 
        #                     vanilla_nei_train_y, 
        #                     vanilla_nei_train_yvar,device,BOUNDS)

        # rgpe_model_final=get_fitted_model(train_x,train_y,train_yvar)
        # plotting_gp(rgpe_model_final,train_x,train_y,train_yvar,device,BOUNDS)
        
        rank_weights_all.append(ranking_weights)
        vanilla_nei_train_x_all.append(vanilla_nei_train_x)
        train_x_all.append(train_x) 
        vanilla_nei_train_y_all.append(vanilla_nei_train_y)
        train_y_all.append(train_y) 
        best_rgpe_all.append(best_rgpe)
        inter_best_rgpe_all.append(inter_best_rgpe)
        #best_random_all.append(best_random)
        best_vanilla_nei_all.append(best_vanilla_nei)
    
    best_rgpe_all = np.array(best_rgpe_all)
    best_random_all = np.array(best_random_all)
    inter_best_rgpe_all=np.array(inter_best_rgpe_all)
    best_vanilla_nei_all = np.array(best_vanilla_nei_all)

    x = range(RANDOM_INITIALIZATION_SIZE, RANDOM_INITIALIZATION_SIZE + N_BATCH + 1)
    y=range(RANDOM_INITIALIZATION_SIZE+inter,RANDOM_INITIALIZATION_SIZE + N_BATCH + 1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # plt.plot(x,best_rgpe_all.mean(axis=0),label="RGPE-NEI")
    # plt.plot(x,best_vanilla_nei_all.mean(axis=0),label="FixedNoiseGP-NEI")

    #Plot RGPE - NEI
    ax.errorbar(
        x, 
        best_rgpe_all.mean(axis=0), 
        yerr= best_rgpe_all.std(axis=0) / math.sqrt(N_TRIALS), 
        label="RGPE - NEI", 
        linewidth=3, 
        capsize=5,
        capthick=3,
    )
    # Plot FixedNoiseGP - NEI
    ax.errorbar(
        x, 
        best_vanilla_nei_all.mean(axis=0), 
        yerr= best_vanilla_nei_all.std(axis=0) / math.sqrt(N_TRIALS), 
        label="FixedNoiseGP - NEI", 
        linewidth=3,
        capsize=5,
        capthick=3,
    )

    #ax.set_ylim(bottom=0)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Observed Value', fontsize=12)
    ax.set_title('Best Observed Value by Iteration', fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    plt.show()

if __name__=='__main__':
    main()