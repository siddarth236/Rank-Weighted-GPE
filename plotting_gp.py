import torch
import matplotlib.pyplot as plt

def plotting_gp(model,train_x,train_y,train_yvar,device,BOUNDS):
    model.eval()
    likelihood=model.likelihood
    likelihood.eval()
    with torch.no_grad():
        test_x = torch.linspace(0,1, 51).to(device)
        #print(test_x.shape)
        observed_pred = likelihood(model(test_x),noise=train_yvar)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        model_mean = observed_pred.mean.squeeze(-1)*model.Y_std + model.Y_mean
        lower=lower*model.Y_std + model.Y_mean
        upper=upper*model.Y_std + model.Y_mean
        # Plot training data as red stars
        ax.plot(train_x.cpu().numpy(), -train_y.cpu().numpy(), 'r*',label='observed data')
        # Plot predictive means as blue line
        ax.plot(test_x.cpu().numpy(), -model_mean.cpu().numpy()[0,:], 'b',label='mean')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.cpu().numpy(), -lower.cpu().numpy()[-1,:],
             -upper.cpu().numpy()[-1,:], alpha=0.5,label='confidence')
        x=torch.linspace(BOUNDS[0,0],BOUNDS[1,0],100)
        #ax.plot(normalize(x,bounds=BOUNDS.cpu()).detach().numpy(),f(x,task).cpu().numpy(),color='k',label='base fn')
        #ax.set_ylabel("Metabolic Cost")
        #ax.set_xlabel("Stiffness Parameter")
        ax.legend(loc='upper left')
        plt.show()
    return