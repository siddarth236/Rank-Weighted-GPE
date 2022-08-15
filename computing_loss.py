import torch
from roll_col import roll_col

def compute_ranking_loss(f_samps, target_y):
    """
    Compute ranking loss for each sample from the posterior over target points.
    
    Args:
        f_samps: `n_samples x (n) x n`-dim tensor of samples
        target_y: `n x 1`-dim tensor of targets
    Returns:
        Tensor: `n_samples`-dim tensor containing the ranking loss across each sample
    """
    n = target_y.shape[0]
    #print(n,"n")
    #print(f_samps.shape,"fsamps shape")
    if f_samps.ndim == 3:
        # Compute ranking loss for target model
        # take cartesian product of target_y
        cartesian_y = torch.cartesian_prod(
            target_y.squeeze(-1), 
            target_y.squeeze(-1),
        ).view(n, n, 2)
        #print(cartesian_y)
        # the diagonal of f_samps are the out-of-sample predictions
        # for each LOO model, compare the out of sample predictions to each in-sample prediction
        rank_loss = (
            (f_samps.diagonal(dim1=1, dim2=2).unsqueeze(-1) < f_samps) ^
            (cartesian_y[..., 0] < cartesian_y[..., 1])
        ).sum(dim=-1).sum(dim=-1)
        #print(rank_loss,"rankloss") 
    else:
        rank_loss = torch.zeros(f_samps.shape[0], dtype=torch.long, device=target_y.device)
        y_stack = target_y.squeeze(-1).expand(f_samps.shape)
        #print(y_stack,"ystack")
        #print(f_samps,"fsamps")
        for i in range(1,target_y.shape[0]):
            rank_loss += (
                (roll_col(f_samps, i) > f_samps) ^ (roll_col(y_stack, i) > y_stack)
            ).sum(dim=-1)
        #print(rank_loss,"rankloss") 
    return rank_loss