import torch
import math

def f(X,shift=0):
    """
    Torch-compatible objective function for the target_task
    """
    # if shift==-1.4:
    #     f_X = X * torch.sin(X + math.pi + shift) + X/10.0
    # if shift==-1:
    #     f_X=torch.exp(-X+shift)*torch.sin(2*math.pi*X+shift)+shift
    # if shift==-0.7:
    #     f_X=0
    #     for i in range(1,7):
    #         f_X+=i*torch.sin((i+1)*X+2*shift*X+i)+X*shift/3
    # if shift==1:
    #     f_X=-(torch.sin(X+shift)+torch.sin(10*X/3)+shift*torch.log10(X+shift)-0.84*X+shift*X)
    # if shift==2.1:
    #     f_X = -torch.sin(X+2*shift)-(shift+1)*torch.sin(10*(X+shift*0.5)/3)-shift
    # if shift==0:
    #     f_X=X*torch.sin(X+shift)+shift*X*torch.cos(2*X)

    #f_X = Hartmann(negate=True)
    #f_X = -torch.sin(X+2*shift)-(shift+1)*torch.sin(10*(X+shift*0.5)/3)-shift
    #f_X = torch.cos(10 * X+shift) * torch.exp(.2 * X+(shift/10)) + torch.exp(-5 * (X - 1) ** 2+(shift/15))
    # f_X=0
    # for i in range(1,7):
    #     f_X+=i*torch.sin((i+1)*X+2*shift*X+i)+X*shift/3
    #f_X=(1.4-3*X+shift)*torch.sin(18*X+shift)
    #f_X=-(torch.sin(X+shift/2)+(shift/2+1)*torch.sin(10*X/3)+shift*torch.log10(X)/2-0.84*X+shift/4*X)
    f_X=torch.exp(-X+shift/2)*torch.sin(2*math.pi*X+shift/2)+shift/2
    #f_X=X*torch.sin(X+shift)+shift*X*torch.cos(2*X)
    return f_X