import numpy as np
import tqdm
from scipy.spatial.distance import cdist
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import evaluate_detector
import loss_adapt
import copy



def eval_dec(detector, loader, model):
    
    ## Evaluate Detector on Cifar
    detector.eval()
    device = next(model.parameters()).device 
    correct, total = 0,0
    for data ,_, labels, idxs in tqdm.tqdm(loader, leave=False):

        data, labels = data.to(device), labels.to(device)
        logits = model(data)
        logits = logits.detach()
        logits = logits.view((data.size(0), -1))
        output = detector(logits.float())
        _, pred = output.max(1)
        
        correct += (pred == labels).float().sum(0).item()
        total += data.size(0)

    print(f"Accuracy : {(correct/total)*100:.2f}")
                
    return correct, total

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def adapt_detector(detector, loader, args , model, interval_iter=3000):

    epochs = args.epochs
    device = next(model.parameters()).device 
    best_acc = -np.Inf
    model.eval()
    detector.train()

    max_iter = epochs * len(loader)
    interval_iter = min ( interval_iter // loader.batch_size ,  len(loader) )
    print("with chekcpoint  interval_iter", interval_iter)
    iter_num = 0
        
    ## Freeze Params of last layer
    for param in detector.fc3.parameters():
        param.requires_grad = False

    ## Training on for other two layers
    param_group = []
    #parameters of linear layer   
    if args.detector_base_name == "scatternet":
        for k, v in detector.linear.named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]

    for k, v in detector.fc1.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in detector.fc2.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    # for k, v in detector.fc3.named_parameters():
    #     param_group += [{'params': v, 'lr': args.lr}]
    for k, v in detector.batchnorm.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]


    optimizer = optim.SGD(param_group) 

    if args.use_label_smoothing:
        print("using label smoothing")
        criterion = loss_adapt.CrossEntropyLabelSmooth(2).to(device)  
    else:
        criterion = nn.CrossEntropyLoss().to(device)  


    for e in range(epochs):
        print("starting epoch " , e)
        for data, _, labels, idxs in tqdm.tqdm(loader, leave=False):

            optimizer.zero_grad()

            data, labels  = data.to(device), labels.to(device)
            logits = model(data)
            logits = logits.detach()
            logits = logits.view((data.size(0), -1))

            if iter_num % interval_iter == 0 and args.cls_par > 0:
                detector.eval()
                mem_label , _ = obtain_label(loader, detector,model)
                mem_label = torch.from_numpy(mem_label).to(device)
                detector.train()
                

            iter_num += 1
            # lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)


            outputs_test = detector(logits.float())
            features_test = detector.features_test


            if args.cls_par > 0:
                pred = mem_label[idxs] 
                classifier_loss = args.cls_par * criterion(outputs_test, pred)
            else:
                classifier_loss = torch.tensor(0.0).to(device)

                
            ## Check sign of ENTROPY LOSS
            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(loss_adapt.Entropy(softmax_out))
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    div_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                    im_loss = entropy_loss - div_loss   

                im_loss = im_loss * args.ent_par
                
            else:
                im_loss = torch.tensor(0.0).to(device)
            
            total_loss = classifier_loss + im_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if iter_num % interval_iter == 0:
                detector.eval()
                #mem_label, _  = obtain_label(loader, detector,model)

                """pseudo_acc, clean_acc, adv_acc = evaluate_detector.evaluate(loader, model, detector=detector, pseudo_labels=mem_label)
                #print(f'Iteration : {iter_num} \t|\t pseudo Acc : {pseudo_acc} \t|\t pseudo clean Acc : {clean_acc} \t|\t pseudo adv Acc: {adv_acc}')
                if args.use_wandb: 
                    wandb.log({"target_pseudo_acc": pseudo_acc, "target_pseudo_clean_acc": clean_acc, "target_pseudo_adv_acc": adv_acc})
                    """
                
                acc, clean_acc, adv_acc = evaluate_detector.evaluate(loader, model, detector=detector, pseudo_labels=None)
                if args.use_wandb:
                    wandb.log({"target_clean_acc": clean_acc, "target_adv_acc": adv_acc})
                    #wandb log pseudo acc, clean acc and adv acc
                   
                """print(f'IM-Loss : {im_loss}')
                print(f'Classifier-Loss : {classifier_loss}')
                print(f'total_loss : {total_loss}')"""
                
                print(f'Iteration : {iter_num} \t|\t Acc : {acc} \t|\t clean Acc : {clean_acc} \t|\t adv Acc: {adv_acc}')
                if args.use_wandb:
                    wandb.log({"target_im_loss":im_loss, "target_classifier_loss":classifier_loss, "target_acc": acc})
                    wandb.log({"target_ent_loss": entropy_loss , "target_div_loss": div_loss, "target_total_loss": total_loss})
               
                # since we are assuming data free setup. we have to use pseudo accuracy
                if acc >= best_acc:
                    best_acc = acc
                    #best_loss= total_loss
                    best_iter = iter_num
                    best_model = copy.deepcopy(detector)
                    torch.cuda.empty_cache()
                    if args.issave:

                        ckpt = {'detector_state_dict':detector.state_dict(),
                                'acc': best_acc,
                                'iter_num': best_iter}
                        torch.save(ckpt, args.save_path)
                        
                
                #creat a ensemble of detectors based on the total loss. ensemble should contain 5 detectors with the least loss
                """if len(ensemble) < 5:
                    ensemble.append((total_loss, copy.deepcopy(detector)))
                else:
                    ensemble.sort(key=lambda x: x[0])
                    if total_loss < ensemble[-1][0]:
                        ensemble[-1] = (total_loss, copy.deepcopy(detector))"""

                detector.train()
    #ensemble.sort(key=lambda x: x[0])
    #ensemble = [x[1] for x in ensemble]

    # save the ensemble of detectors
    #torch.save([x.state_dict() for x in ensemble], args.save_path + "_ensemble")
    return  best_model                   # ensemle is list of best models according to the total loss, best_model is actual best model

def obtain_label(loader, detector, model):
    start_test = True
    device = next(model.parameters()).device 
    model.eval()
    with torch.no_grad():
        for data, _, labels, idxs in loader:
            data , labels = data.to(device) , labels.to(device)

            logits = model(data)
            logits = logits.detach()
            data = data.cpu()
            labels = labels.cpu()

            logits = logits.view((data.size(0), -1))
            outputs = detector(logits.float())
            logits = logits.cpu()

            feas = detector.features_test
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str)
    return pred_label.astype('int') , acc
