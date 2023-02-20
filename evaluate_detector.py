import argparse
import torch
from dataset.combined_dataset import CombDataset
import utils
import torch.optim
from torchmetrics import Accuracy

def evaluate(loader,detector_base,detector, pseudo_labels=None):

    device = next(detector_base.parameters()).device 
    detector_base.eval()

    #if detector is list
    if isinstance(detector, list):
        for d in detector:
            d.eval()

    else:
        detector.eval()
   
    acc = Accuracy(average='none', num_classes=2).to(device)
    total_acc = Accuracy(num_classes=2).to(device)
   
    with torch.no_grad():

        for data, _,labels, idx in loader:

            data, labels = data.to(device), labels.to(device)
            
            if pseudo_labels is not None:  
                labels = torch.tensor(pseudo_labels[idx])
                labels = labels.to(device)

            logits = detector_base(data)
            logits = logits.view((data.size(0), -1))
            
            op = detector(logits.float())

            acc.update(op,labels)
            total_acc.update(op,labels)
            
    
    acc = acc.compute()
    
   
    return total_acc.compute().item() ,acc[0].item() , acc[1].item()
    


        ## Add Arguments
    parser = argparse.ArgumentParser(description='Adapt Source Detector to Target Detector')
    
    parser.add_argument('--dataroot', type=str )
    parser.add_argument('--dataset',help='Target Dataset',default='cifar10')
   
    parser.add_argument('--batch_size',help='Batch Size',default=64,type=int)
    parser.add_argument('--model_name',help='Model Choice', default='resnet18')
    parser.add_argument('--model_path' , type=str)
    parser.add_argument('--detector_path' , type=str , help="path to source detector")
    parser.add_argument('--method', type=str, required=True, help="method used to train target model, different methods use different meand and std")

    parser.add_argument('--gpu',help='Model Choice', default='0')
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--cls_par', type=float, default=0.5)
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")    
    parser.add_argument('--droprate', help='dropout rate', default=0.005, type = float)
    parser.add_argument('--use_wandb' ,action='store_true', default=False)
    parser.add_argument('--detector_base_name', type=str , help="target model is default detector base ")
    parser.add_argument('--detector_base_path', type=str , help="target model is default detector base")
    parser.add_argument('--detector_method', type=str , default="vanila", help="target model is default detector base")

    parser.add_argument('--num_scatternet_layers', type=int, default=3)
    parser.add_argument('--detector_hidden_size', type=int, default=64)
    parser.add_argument('--adv_data_path', type =str)
    parser.add_argument('--seed', type =int ,default =42)
    args = parser.parse_args()
    
    if args.detector_base_name is None:
        args.detector_base_name = args.model_name
        args.detector_base_path = args.model_path
        args.detector_method = args.method      # source detector base is always trained using vanila method, target is trained using different methods

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    args.device = device

    utils.fix_seed_value(args.seed)
    args.channels = 3
   

    args.num_classes =10
    if args.dataset in ["fmnist", "mnist"] :
        args.channels = 1
    if args.dataset in ["cifar100"]:
        args.num_classes = 100
    if args.method == "shot_office":
        args.num_classes =31
    if args.method in ["dine_office_home" , "decision_office_home"]:
        args.num_classes = 65

    print("dataset channels :", args.channels)
    
  

    #load model
    model = utils.get_normalized_model(args).to(device)
    args.model = model

    #load clean test dataset
    _, test_dataset = utils.load_dataset(args)

    #load adv dataset generated for target model
    adv_data = utils.load_adv_data(args.adv_data_path)

    clean_dataloader =  torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle =False)
    adv_dataloader =  torch.utils.data.DataLoader(adv_data, batch_size = args.batch_size, shuffle =False)

 
    # combine clean and adv dataset together to train detector
    combined_dataset = CombDataset(test_dataset, adv_data, return_idx=True)
    combined_dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size = args.batch_size, shuffle =True)
    print(f'Combined Data Size : ',len(combined_dataset))
    

    ## Load source Detector
    detector_base, detector = utils.load_detector(args, load_checkpoint=True)
    detector_base.to(device)
    detector.to(device)
    acc ,clean_acc , adv_acc =   evaluate_detector(combined_dataloader, detector_base, detector)
    print("test adversarial detection accuracy" , adv_acc)
    print("test clean detection accuracy", clean_acc)
    print("test total accuracy " , acc)
    print("==========================================")   
    
    print('= '*75)