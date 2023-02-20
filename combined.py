import os
import tqdm
import torch
import argparse
import numpy as np
import wandb
from correction import compute_adv_accuracy_after_correction , correct_data_ssim
from create_adv_data import create_adv_attack_multiple_attacks
from dataset.combined_dataset import CombDataset
from dataset.common_corruption import CommonCorruption
import evaluate_model
import evaluate_detector
from utils import get_correct
import utils
import train_target_detector

def divide_clean_and_adv(data, labels, det_labels, adv_indices,device):

    contanimnated_data = torch.index_select(data,dim=0,index = adv_indices)   #filter data
    contaminated_labels =  torch.index_select(labels,dim=0,index = adv_indices)  #filter corresponding labels
    contaminated_det_labels =  torch.index_select(det_labels,dim=0,index = adv_indices)  #filter corresponding labels
    
    indices = set(adv_indices.tolist()) #set of index related to contaminated data
    clean_data_indices = torch.Tensor(list(set(np.arange(0,data.size(0))) - indices)).to(device).to(torch.int32)        # set of indexes of clean data

    clean_data = torch.index_select(data,dim=0,index = clean_data_indices)   #filter data
    clean_labels =  torch.index_select(labels,dim=0,index = clean_data_indices)  #filter corresponding labels
    clean_det_labels =  torch.index_select(det_labels,dim=0,index = clean_data_indices)  #filter corresponding labels
    
    return contanimnated_data, contaminated_labels , contaminated_det_labels , clean_data, clean_labels, clean_det_labels
    
    

def calc_comb_acc(loader, model, detector, args, detector_base):
    
    model.eval()
    detector_base.eval()
    detector.eval()


    correct, total = 0,0 ## Overall Acc. -> Detector's Accuracy
    metrics = {'clean':{'correct':0, 'total':0+1e-5}, 'adv':{'correct':0, 'total':0+1e-5}} ## Classifier Acc. (Not Detector)
    pbar = tqdm.tqdm(enumerate(loader), unit='batches', leave=False, total=len(loader))

    total_predicted_as_contaminated = 0
    total_predicted_as_clean=0

    for idx, (data, labels, det_labels , _) in pbar:
        total += data.size(0)
        with torch.no_grad():

            data, labels, det_labels = data.to(args.device), labels.to(args.device), det_labels.to(args.device)
            logits = detector_base(data)
            logits = logits.detach()
           
            out_detect = detector(logits)

        
            _, pred_detect = torch.max(out_detect, 1)
            correct += get_correct(out_detect, det_labels)

            acc = (correct/total)*100.

            #find index of contaminted data samples
            m = (pred_detect==1) 
            adv_indices = m.nonzero().squeeze(1) # 

            #divide data between contaminated and clean using predicted contaminated indices
            contanimnated_data, contaminated_labels ,contaminated_det_labels, clean_data, clean_labels , clean_det_labels = divide_clean_and_adv(data, labels,det_labels,adv_indices,args.device)

            total_predicted_as_contaminated += len(contanimnated_data)
            total_predicted_as_clean += len(clean_data)

            #remove noise from contaminated data
            if len(contanimnated_data) > 0:
                if args.soft_detection:
                    corrected_data =  correct_data_ssim(model, contanimnated_data, contaminated_labels, args.r_range, args.pop,detector = detector, detector_base = detector_base, soft_detection_r=args.soft_detection_r) ## Get Corrected Data
                else:
                    corrected_data = correct_data_ssim(model, contanimnated_data, contaminated_labels, args.r_range, args.pop,detector = None, detector_base = None,soft_detection_r=args.soft_detection_r)
                
                model.eval() ## Since it could be coming in from dropout-enabled

                output = model(corrected_data)  #get output on corrected data
                for i in range(len(corrected_data)):
                    if contaminated_det_labels[i].item() == 1:   #if data sample is actually contaminated
                        key='adv'
                    else:    #data is actually clean
                        key='clean'
                    
                    metrics[key]['correct'] += get_correct(output[i].unsqueeze(0), contaminated_labels[i])
                    metrics[key]['total'] += 1
            
            model.eval()
            if len(clean_data) > 0: 
                if args.soft_detection:
                    clean_data =  correct_data_ssim(model, clean_data, clean_labels, args.r_range, args.pop,detector = detector, detector_base = detector_base,soft_detection_r=args.soft_detection_r)
                output = model(clean_data)
            for i in range(len(clean_data)):
                if clean_det_labels[i].item() == 1:
                    key='adv'
                else:
                    key='clean'
                
                metrics[key]['correct'] += get_correct(output[i].unsqueeze(0), clean_labels[i])
                metrics[key]['total'] += 1

        pbar.set_description(f'D-Acc : {acc:.2f} | Clean-Acc : {(metrics["clean"]["correct"]/metrics["clean"]["total"])*100.:.2f} | Adv-Acc : {(metrics["adv"]["correct"]/metrics["adv"]["total"])*100.:.2f}')

   
    print(f'D-Acc : {acc:.2f} | Clean-Acc : {(metrics["clean"]["correct"]/metrics["clean"]["total"])*100.:.2f} | Adv-Acc : {(metrics["adv"]["correct"]/metrics["adv"]["total"])*100.:.2f}')
    print(f'Total: Detector : {total} \t|\t Clean : {metrics["clean"]["total"]} \t|\t Adv : {metrics["adv"]["total"]}')
    print(f'Total predicted as contaminated : {total_predicted_as_contaminated} \t|\t Total predicted as clean : {total_predicted_as_clean}')
    if args.use_wandb:
        #wandb log clean and adv acc
        wandb.log({"Clean_Acc":(metrics["clean"]["correct"]/metrics["clean"]["total"])*100.,"Adv_Acc":(metrics["adv"]["correct"]/metrics["adv"]["total"])*100.})


def main(args):

    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    print("soft detection:", args.soft_detection)
    
    if args.use_wandb:
        wandb.init(project="dad")
        wandb.config.update(args)


    #fix seed value
    if args.seed !=-1:
        print("using fixed seed :",  args.seed)
        utils.fix_seed_value(args.seed)

    # load device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    args.device = device

    args.n_classes = 10

    
    if args.detector_base_name is None:
        args.detector_base_name = args.model_name
        args.detector_base_path = args.model_path
        args.detector_method = args.method

    
    print(f"Model : {args.model_name} \t|\t Dataset : {args.dataset} \t|\t Arb. Dataset : {args.s_dataset} \t|\t Attack : {args.attacks}")
    
    args = utils.update_channels_and_num_classe_from_dataset(args)
    print(args.attacks)


    #load model
    model = utils.get_normalized_model(args).to(device)
    model.eval()

    #load clean test dataset
    _, test_dataset = utils.load_dataset(args)
    # take subset of 300 samples
    #test_dataset = torch.utils.data.Subset(test_dataset, range(100))


    clean_dataloader =  torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle =False)
    clean_acc = evaluate_model.evaluate(model,clean_dataloader)[0]
   
    if args.attacks[0] in common_corruptions:
        print("using common corruption " , args.attacks[0])
        adv_data = CommonCorruption(args.common_corruption_root, args.attacks[0])
        #load subset of 300 samples
        #adv_data = torch.utils.data.Subset(adv_data, range(300))
        
    else:
        adv_data_path = os.path.join(os.path.dirname(args.model_path) , 'test_{}_data.pt'.format("_".join(args.attacks)))

        if os.path.isfile(adv_data_path) and not args.recreate_adv_data:
            print("using created adv data")
            adv_data = torch.load(adv_data_path, map_location="cpu")
        else:
            print("creating adv data")
            adv_data = create_adv_attack_multiple_attacks(clean_dataloader,args.dataset,args.attacks,model, sample_percent=args.sample_percent, batch_size = args.correction_batch_size)
            torch.save(adv_data, adv_data_path)
    
    print("test datset size : ", len(test_dataset))
    print("adv data size : ", len(adv_data))

    clean_dataloader =  torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle =False)
    adv_dataloader = torch.utils.data.DataLoader(adv_data, batch_size = args.correction_batch_size, shuffle =False)
    
    combined_dataset = CombDataset(test_dataset, adv_data, return_idx=True)
    combined_dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size = args.batch_size, shuffle =True)
    print(f'Combined Data Size : ',len(combined_dataset))
    #compute classifier accuracy
    print('Data and Model Loaded...') 
   
    clean_acc = evaluate_model.evaluate(model,clean_dataloader)[0]
    adv_acc = evaluate_model.evaluate(model,adv_dataloader)[0]
    print("accuracy of target model on clean data : ",clean_acc)    #compute model accuracy on clean data
    print("accuracy of target model on adv data : ",adv_acc)  # compute model accuracy on adv data
    print('='*100)


    #Check Performance Assuming Ideal Detector
    args.return_corr_data = False
    acc, corr, total = compute_adv_accuracy_after_correction(model, adv_dataloader,args.r_range, args.pop,soft_detection_r=args.soft_detection_r)
    print(f'Accuracy : {acc} \t|\t Correct : {corr} \t|\t Total : {total}  (Only Correction)')
    print('='*100)


    #load source detector
    detector = utils.load_detector(args, load_checkpoint=True)
    detector_base  = model


    #train target detector using data free adaption
    args.save_path = os.path.join(os.path.dirname(args.model_path) , 'source_{}_{}_{}_seed_{}_target_detector.pt'.format(args.s_dataset,args.s_model,'_'.join(args.attacks), args.seed))
    args.issave=True
    if os.path.exists(args.save_path) and not args.retrain_detector :
        print("Loading saved detector")
        detector.load_state_dict(torch.load(args.save_path, map_location="cpu")["detector_state_dict"])
        detector = detector.to(device)
        detector_base.to(args.device)
    else:
        print("Training detector")
        ## Evaluate Performance of detector Without Adaptation
        print('-'*100)
        print('Performance of T-I detector w/0 adaptation')
        detector.to(args.device)
        detector_base.to(args.device)
        detector.eval()
        detector_base.eval()
        evaluate_detector.evaluate(combined_dataloader, detector_base ,detector)
        print('-'*100)
        detector = train_target_detector.adapt_detector(detector, combined_dataloader, args, detector_base)
    

    
    combined_dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size = args.correction_batch_size, shuffle =False)
    
    #torch.cuda.empty_cache()
    
    #evaluate best detector on test data
    acc ,clean_acc , adv_acc = evaluate_detector.evaluate(combined_dataloader, detector_base,detector)
    print("test adversarial best detection accuracy" , adv_acc*100)
    print("test clean best detection accuracy", clean_acc*100)
    print("test best total accuracy " , acc)
    print("==========================================") 
    print('='*100)

    ## Check Combined Peformance
    args.return_corr_data = True
    args.soft_detection = True
    print("Combined Performance with soft detction")
    calc_comb_acc(combined_dataloader, model, detector, args, detector_base)
    print('='*100)

    
    if args.use_wandb:
        #save wandb id for future use
        #create file if it doesn't exist
        if not os.path.exists(args.log_path):
            with open(args.log_path, 'w') as f:
                f.write('')

        with open(args.log_path, 'a') as f:
            # get wandb run id
            if args.use_wandb:
                wandb_id = wandb.run.id
            s = "{}_{}_{}_{}_{}_{}  {}".format(args.dataset, args.model_name,args.attacks,args.method,args.s_dataset,args.s_model, wandb_id)
            f.write(s)
            f.write('\n')
        
        wandb.finish()


if __name__ == '__main__':

    ## Add Arguments
    parser = argparse.ArgumentParser(description='Check Detector Performance')
    
    parser.add_argument('--dataset',help='Dataset',default='cifar10')
    parser.add_argument('--batch_size',help='Batch Size',default=64,type=int)
    parser.add_argument('--model_name',help='Model Choice', default='WRN-16-1')
    parser.add_argument('--model_path' , type=str)
    parser.add_argument('--detector_path', type=str , help="source detector path")

    parser.add_argument('--attacks', nargs='+', default=['pgd'])
    parser.add_argument('--r_range', help='max radius range', default=16, type = int)
    parser.add_argument('--pop', help='population count for each radius', default=10, type = int)
    parser.add_argument('--gpu',help='Model Choice', default='0')
    parser.add_argument('--droprate', type=float, default=0.005)
    parser.add_argument('--soft_detection_r', type=int, default=32)
    
    parser.add_argument('--method', type=str, required=True, help="method used to train target model, different methods use different meand and std")
    parser.add_argument('--use_wandb' ,action='store_true', default=False)
    
    parser.add_argument('--detector_base_name', type=str , help="source detector base name. set scatternet if detector base is scatternet", default=None) 
    parser.add_argument('--num_scatternet_layers', type=int, default=3)
    parser.add_argument('--detector_hidden_size', type=int, default=64)
    parser.add_argument('--seed', type =int ,default =-1)
    parser.add_argument('--attack_mode', type=str, default="only_classifier")
    parser.add_argument('--s_dataset', type=str, default="fmnist")
    parser.add_argument('--soft_detection', default=True , action='store_true')
    parser.add_argument('--recreate_adv_data', action='store_true', default=False)
    parser.add_argument('--correction_batch_size', type=int, default=128)
    
    # list of float values argument 
    parser.add_argument('--sample_percent', nargs='+', type=float, default=[])

    # detector adapt arguments
    parser.add_argument('--epochs' , type=int , default = 20)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--ent_par', type=float, default=0.8)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--issave', type=bool, default=False)
    parser.add_argument('--use_label_smoothing', action = 'store_true', default=False)
    parser.add_argument('--retrain_detector', action = 'store_true', default=False) # retrain detector on target dataset
    parser.add_argument('--s_model', type=str, default="resnet18")


    #shot method specific argument
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"]) 
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--s', type=str, default='A', help="source")
    parser.add_argument('--t', type=str, default='D', help="target")
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--log_path', type=str, default='./logs/logs.txt')

    #common corruption arguments
    parser.add_argument('--common_corruption_root', type=str, default='/media2/inder/dad_shubham/data-free-defense/clean_data/common_corruption/CIFAR-10-C')

    args = parser.parse_args()

    # get start time
    import time
    start_time = time.time()
    main(args)
    print("Total time taken: ", time.time() - start_time)