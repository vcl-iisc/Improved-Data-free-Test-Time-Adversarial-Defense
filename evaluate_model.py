import torch
import torch

def evaluate(model, dataloader):
    model.eval()
    accs = 0
    n_samples = 0
    device = next(model.parameters()).device 

    for iter_n, batch in enumerate(dataloader):
        images = batch[0].to(device)
        targets = batch[1].to(device)
        n_samples += targets.shape[0]

        with torch.no_grad():
            outputs = model(images)
            acc = outputs.max(1)[1].eq(targets).float().sum()
            acc = acc.detach().cpu()
        accs += acc
    
    accuracy = (accs / n_samples) * 100
    return accuracy.item() , accs, n_samples




