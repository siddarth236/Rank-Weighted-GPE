import torch
def roll_col(X, shift):  
    """
    Rotate columns to right by shift.
    """
    #print(torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1))
    return torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1)