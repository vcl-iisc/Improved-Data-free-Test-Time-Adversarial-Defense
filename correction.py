import tqdm
import torch
from pytorch_msssim import ssim
from frequencyHelper import generateDataWithDifferentFrequencies_3Channel as freq_3t_new
ssims = []
selected_r = []


method=""
def enable_dropout(m):
    #m[1][1][3].train()   #oxford pet model
    #m[1].dad_dropout.train()        # changes made for tinyimagenet
     ## Enable Dropout for populations
    if "shot" in method:
        m[1][2].dad_dropout.train()
    else:
        for each_module in m.modules():
            if each_module.__class__.__name__.startswith('Dropout'):
                each_module.train()

def get_ssim(X, Y):
    ## Calculate SSIM between data and data component
    ssim_val = ssim( X, Y, data_range=torch.max(X).item(), size_average=False,  nonnegative_ssim=True) # return (N,)
    return ssim_val


#this function uses ssim metric
def correct_data_ssim(model, data, labels, r_range ,pop, return_corr_data=True,detector=None, detector_base=None, soft_detection_r=32):
    
    global ssims
    
    # Get Precition of Original Sample
    model.eval()
    device = next(model.parameters()).device 
    output_adv = model(data)
    _, pred_adv = torch.max(output_adv, 1)
    #del output_adv 

    if detector is not None:
        logits = detector_base(data)
        logits = logits.detach()
        ## Check Detector's Prediction
    
        out_detect = detector(logits)

        
        det_pred = torch.nn.functional.softmax(out_detect)
        min_r = soft_detection_r * det_pred[:,0]        #update this using confidence score
        min_r = torch.where(min_r <4, 4, min_r)
        min_r = min_r.long()
    else:
        min_r =  (torch.ones(data.size(0))*4).to(device)


    step_size = 2

    metrics = {r: torch.zeros(size=(data.size(0),)).to(device)
               for r in range(2, r_range+1, step_size)}
    ssim_metric = {r: torch.zeros(size=(data.size(0),)).to(
        device) for r in range(2, r_range+1, step_size)}
    

    # For each radius
    for r in range(2, r_range+1, step_size):

        pred_low_list = None
        enable_dropout(model)
        
        # Run multiple forward passes -> different classifiers
        # Get Low-High Componenets
        data_l, _ = freq_3t_new(data, radius=r, device=device)
        data_l = data_l.to(device, dtype=torch.float)
        ssim_metric[r] += get_ssim(data, data_l) 
        # For each population
        for idx in range(1, pop+1):

            output_low = model(data_l)
            _, pred_low = torch.max(output_low, 1)

            pred_low = pred_low.unsqueeze(0)
            if pred_low_list is None:
                pred_low_list = pred_low
            else:
                pred_low_list = torch.cat((pred_low_list, pred_low), dim=0)

        # Check for how many models out of populations did the label predicted similar to original adversarial pertubation
        lcr_rad = torch.sum(pred_low_list == pred_adv.unsqueeze(0), dim=0)
        metrics[r] = lcr_rad / pop  # compute adv_cont

    # Pick the maximum non-zero lcr radius
    best_r = min_r
    for i in range(data.size(0)):
        for (r, lcr_temp) in metrics.items():
            if ssim_metric[r][i] - lcr_temp[i] <= 0:
                ssims.append(r)
                break
            if r > best_r[i]:
                # save r in dictionaray with count
                best_r[i] = torch.tensor(r)

    

    # Get prediction on low-pass version of non-dropout model
    model.eval()
    data_best_r = torch.empty_like(data)
    for i in range(data.size(0)):
        r = best_r[i]
        img = data[i]
        x, _ = freq_3t_new(img, radius=r.item(), device=device)
        x = x.to(device, dtype=torch.float)
        data_best_r[i] = x

    if return_corr_data:
        return data_best_r

    output_best_r = model(data_best_r)
    _, pred_best_r = torch.max(output_best_r, 1)
    corr_best_r = (pred_best_r == labels).float().sum(0).item()
    
   

    return corr_best_r, best_r

def compute_adv_accuracy_after_correction(model,adv_dataloader, r_range, pop,soft_detection_r=32):
    global ssims
    correct, total = 0., 0.
    device = next(model.parameters()).device 


    pbar = tqdm.tqdm(adv_dataloader, unit='batches', leave=False, total=len(adv_dataloader))
    # For each sample
    for i, (data, labels) in tqdm.tqdm(enumerate(pbar)):

        data, labels = data.to(device), labels.to(device)
        total += data.size(0)

        with torch.no_grad():
            denoised_data = correct_data_ssim(model, data, labels, r_range, pop,soft_detection_r=soft_detection_r)
            output = model(denoised_data)
            _, pred = torch.max(output, 1)
        
       
        correct += (pred == labels).float().sum(0).item()
        pbar.set_description(f"Acc : {(correct/total)*100.:.2f}")

    acc = (correct/total)*100.

    return acc, correct, total


