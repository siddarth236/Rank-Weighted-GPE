from botorch.sampling.samplers import SobolQMCNormalSampler
import torch
from target_model_loss import get_target_model_loocv_sample_preds
from computing_loss import compute_ranking_loss


def compute_rank_weights(train_x,train_y,yvar, base_models, target_model, num_samples, device):
    """
    Compute ranking weights for each base model and the target model (using 
        LOOCV for the target model). Note: This implementation does not currently 
        address weight dilution, since we only have a small number of base models.
    
    Args:
        train_x: `n x d` tensor of training points (for target task)
        train_y: `n` tensor of training targets (for target task)
        base_models: list of base models
        target_model: target model
        num_samples: number of mc samples
    
    Returns:
        Tensor: `n_t`-dim tensor with the ranking weight for each model
    """
    ranking_losses = []
    # compute ranking loss for each base model
    for task in range(len(base_models)):
        #print(task,"task")
        model = base_models[task]
        # compute posterior over training points for target task
        posterior = model.posterior(train_x)
        #print(posterior)
        sampler = SobolQMCNormalSampler(num_samples=num_samples)
        base_f_samps = sampler(posterior).squeeze(-1).squeeze(-1)
        #print(base_f_samps.shape,task)
        #base_f_samps is the other models prediction at train_X
        # compute and save ranking loss
        ranking_losses.append(compute_ranking_loss(base_f_samps, train_y))
    # compute ranking loss for target model using LOOCV
    # f_samps

    target_f_samps = get_target_model_loocv_sample_preds(
        train_x, train_y, yvar, target_model, num_samples,device)
    ranking_losses.append(compute_ranking_loss(target_f_samps, train_y))
    ranking_loss_tensor = torch.stack(ranking_losses)
    #print(ranking_loss_tensor,"stack")
    # compute best model (minimum ranking loss) for each sample
    best_models = torch.argmin(ranking_loss_tensor, dim=0)
    # compute proportion of samples for which each model is best
    rank_weights = best_models.bincount(minlength=len(ranking_losses)).type_as(train_x) / num_samples
    #print(best_models,"bm")
    #print(best_models.bincount(minlength=len(ranking_losses)),"binc")
    #print(rank_weights,"rw")
    return rank_weights