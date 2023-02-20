import argparse
import torch
import os
import create_adv_data
from dataset.combined_dataset import CombDataset
from evaluate_detector import evaluate
from metric import AverageMeter
import utils
import tqdm
import torch.nn as nn
import torch.optim as optim
import wandb 
import loss_adapt
import evaluate_model

def train_detector(args,loader,test_dataloader):

    epochs = args.epochs
    device = args.device

    detector = utils.load_detector(args, load_checkpoint=False)
    detector_base = args.model
    detector_base.to(device)
    detector.to(device)
    detector_base.eval()

    if args.use_label_smoothing:
        criterion_id = loss_adapt.CrossEntropyLabelSmooth(2).to(device)  
    else:
        criterion_id = nn.CrossEntropyLoss().to(device)  


    lr  = args.lr
    if args.optim=="rms":
        optimizer = optim.RMSprop(detector.parameters(), lr=lr)
    elif args.optim == "adam":
        optimizer = optim.Adam(detector.parameters(), lr=lr)
    elif args.optim =="sgd":
        optimizer = optim.SGD([{'params':detector.parameters(), 'lr':lr}],
                 weight_decay=5e-4, momentum=0.9, nesterov=True)

    
    detector_acc = AverageMeter()
    detector_loss = AverageMeter()

    
    print(f'saving @ : {args.save_path}')

    pbar = tqdm.tqdm(range(epochs), leave=False)
    best_acc = 0
    for epoch in pbar:
        
        for data, cls, labels , _ in loader:

            optimizer.zero_grad()

            data, cls, labels = data.to(device), cls.to(device), labels.to(device)
            
            with torch.no_grad():
                logits = detector_base(data)
                logits = logits.detach()
            logits = logits.view((data.size(0), -1))
   
            output = detector(logits)

            loss = criterion_id(output, labels)
            loss.backward()
            optimizer.step()

            _, pred = output.max(1)
            acc = (pred.eq(labels).sum().item() / pred.size(0)) * 100.
            detector_acc.update(acc, data.size(0)) ## Detector Accuracy Update
            detector_loss.update(loss.item(), data.size(0))

        
        pbar.set_description(f'Acc : {detector_acc.avg} | Loss : {detector_loss.avg}')
    
        acc , clean_acc, adv_acc = evaluate(test_dataloader, detector_base, detector)
        

        print("test adversarial detection accuracy" , adv_acc)
        print("test clean detection accuracy", clean_acc)
        print("test total accuracy " , acc)
        print("==========================================")   
        detector.train()
        
        if args.use_wandb:
            wandb.log({"loss": detector_loss.avg , "clean_acc": clean_acc , "adv_acc": adv_acc , "total_acc": acc})
        
        if acc >=best_acc:
            print("best test accuracy {} obtained at epoch {}".format(acc, epoch) )
            torch.save({
            'detector_state_dict': detector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': detector_acc.avg,
            'loss': detector_loss.avg
            }, args.save_path)
            best_acc =acc
        


    print(f'train Accuracy : {detector_acc.avg} \t|\t train Loss : {detector_loss.avg}')
    return detector_base, detector

def main(args):
    
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    device = args.device
    print(f"Model : {args.model_name} \t|\t Dataset : {args.dataset} \t|\t Attack : {args.attack}")

    args = utils.update_channels_and_num_classe_from_dataset(args)
    print("dataset channels :", args.channels)
    
    #load model
    model = utils.get_normalized_model(args).to(device)
    args.model = model
    
    #load dataset
    train_dataset, test_dataset = utils.load_dataset(args)

    
    #create adv train and test dataset
    attack = args.attack
    adv_data_path = os.path.join(os.path.dirname(args.model_path) , 'train_{}_data.pt'.format(attack))
    adv_test_data_path = os.path.join(os.path.dirname(args.model_path) , 'test_{}_data.pt'.format(attack))
    
    #create adversarial dataset
    print("generate adv dataset")
    if os.path.isfile(adv_data_path) and not args.recreate_adv_data:
            print("using created adv data")
            adv_train_data = torch.load(adv_data_path, map_location="cpu")
            adv_test_data = torch.load(adv_test_data_path, map_location="cpu")
    else:
        print("creating adv data")
        adv_train_data = create_adv_data.create_adv_attack_multiple_attacks(torch.utils.data.DataLoader(train_dataset),args.dataset,["pgd"],model,sample_percent=[])
        adv_test_data = create_adv_data.create_adv_attack_multiple_attacks(torch.utils.data.DataLoader(test_dataset),args.dataset,["pgd"],model,sample_percent=[])

        #save dataset
        torch.save(adv_train_data, adv_data_path)
        torch.save(adv_test_data, adv_test_data_path)
    
    #create combined dataset
    combined_dataset = CombDataset(train_dataset, adv_train_data, return_idx=True)
    combined_test_dataset = CombDataset(test_dataset, adv_test_data, return_idx=True)

    dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size = args.batch_size, shuffle =True)
    test_dataloader = torch.utils.data.DataLoader(combined_test_dataset, batch_size = args.batch_size, shuffle =False)

    acc, _, _ = evaluate_model.evaluate(model, torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle =False))
    print("base model clean accuracy:" , acc)

    

    args.save_path = os.path.dirname(args.model_path)
   
    s = "seed_"+str(args.seed) + "_"
    if args.use_label_smoothing:
        s+="label_smoothed_"
    args.save_path = os.path.join(args.save_path, "{}_{}_{}source_detector.pt".format(args.dataset, args.attack,s))

    #train detector
    detector = train_detector(args, dataloader, test_dataloader)
    detector_base = model
    detector_base.eval()

    #testing
    print("testing the final model")
    print(args.save_path)
    detector = utils.load_detector(args, load_checkpoint=False)
    ckpt = torch.load(args.save_path)
    detector.load_state_dict(ckpt['detector_state_dict'])
    detector.to(device)
    detector_base.to(device)

    acc, clean_acc, adv_acc = evaluate(test_dataloader , detector_base , detector)
    print("test adversarial detection accuracy" , adv_acc)
    print("test clean detection accuracy", clean_acc)
    print("test total accuracy " , acc)
    print("==========================================")   

    return detector


if __name__ == '__main__':
    ## Add Arguments
    parser = argparse.ArgumentParser(description='Train Source Detector')
    
    parser.add_argument('--name', type=str, help="experiment name for wandb")
    parser.add_argument('--dataroot')
    parser.add_argument('--dataset',help='Dataset') ## 'source/arbitrary' dataset
    parser.add_argument('--batch_size',help='Batch Size',default=128,type=int) 
    parser.add_argument('--model_name',help='Model Choice', default='resnet18_source') ## 'model' -> F_s
    parser.add_argument('--model_path', type=str)
    parser.add_argument("--attack",help='Attack choice', default = "PGD",type=str) ## Same attack for all arbitrary dataset (With Diff. params though)
    parser.add_argument('--gpu',help='GPU Choice', default='0')
    parser.add_argument('--only_eval',help='Only Do Detector Evaluation', action='store_true')
    parser.add_argument('--droprate', help='dropout rate', default=0.005, type = float)
    parser.add_argument('--method', type=str, help="ignore it")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--use_wandb' ,action='store_true', default=False)
    parser.add_argument('--detector_base_name', type=str, default=None)
    parser.add_argument('--num_scatternet_layers', type=int, default=3)
    parser.add_argument('--detector_hidden_size', type=int, default=64)
    parser.add_argument('--use_label_smoothing', action = 'store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--optim', type=str, default="rms", choices=["rms", "sgd", "adam"])
    parser.add_argument('--seed', type =int ,default =0)
    parser.add_argument('--recreate_adv_data', action='store_true', default=False)
    args = parser.parse_args()
    
    utils.fix_seed_value(args.seed)
    
    if args.use_wandb:
        wandb.init(name=args.name, project="dad")

    main(args)
    
    wandb.finish()