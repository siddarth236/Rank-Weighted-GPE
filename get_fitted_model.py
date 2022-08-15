from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.kernels import RBFKernel,ScaleKernel
from gpytorch.priors import MultivariateNormalPrior,GammaPrior,NormalPrior

def get_fitted_model(train_X, train_Y, train_Yvar, state_dict=None):
    """
    Get a single task GP. The model will be fit unless a state_dict with model 
        hyperparameters is provided.
    """
    Y_mean = train_Y.mean(dim=-2, keepdim=True)
    Y_std = train_Y.std(dim=-2, keepdim=True)
    #print(train_X.shape,train_Y.shape)
    #print(Y_std,Y_mean,"fitting")
    model = FixedNoiseGP(train_X, (train_Y - Y_mean)/Y_std,train_Yvar,
            covar_module=ScaleKernel(base_kernel=RBFKernel(lengthscale_prior=NormalPrior(0.08,3.16))
            ,outputscale_prior=NormalPrior(9.48,3.16)))
    model.Y_mean = Y_mean
    model.Y_std = Y_std
    if state_dict is None:
        mll= ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
        fit_gpytorch_model(mll)
    else:
        model.load_state_dict(state_dict)
    return model