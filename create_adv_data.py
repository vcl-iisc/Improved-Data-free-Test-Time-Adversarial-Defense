# program to create adversarial data on loader on classifier 
import torch
import adversarial_attack
from torch.utils.data import TensorDataset
import tqdm

#create adversarial data on detector 
def create_adv_data_detector(dataloader, detector ,dataset, attack):
    # dataloader is for clean images
    print("creating adversarial data")
    attack = adversarial_attack.get_attack(dataset, attack, detector)
    attack.set_return_type('int') # Save as integer.
    adv_images = []
    labels = []
    for images, cls_labels in tqdm.tqdm(dataloader):
        det_labels = torch.zeros((images.size(0)),dtype=int)
        adv_images.append(attack(images,det_labels).cpu().detach())  #clean images have label 0
        labels.append(cls_labels.cpu().detach())
    adv_images = torch.cat(adv_images, dim=0)
    labels = torch.cat(labels, dim=0)

    adv_data = TensorDataset(adv_images.float()/255, labels)
    print("done")
    return adv_data


#create adversarial data on detector 
def create_adv_data_detector_predict_adv_as_clean(dataloader, detector ,dataset, attack):
    # dataloader is for clean images
    print("creating adversarial data")
    attack = adversarial_attack.get_attack(dataset, attack, detector)
    attack.set_return_type('int') # Save as integer.
    adv_images = []
    labels = []
    for images, cls_labels in tqdm.tqdm(dataloader):
        det_labels = torch.ones((images.size(0)),dtype=int)
        adv_images.append(attack(images,det_labels).cpu().detach())  #clean images have label 0
        labels.append(cls_labels.cpu().detach())
    adv_images = torch.cat(adv_images, dim=0)
    labels = torch.cat(labels, dim=0)

    adv_data = TensorDataset(adv_images.float()/255, labels)
    print("done")
    return adv_data

#create adversarial data on classifier 
def create_adv_data_classifier(dataloader ,dataset, attack, model):
    # dataloader is for clean images
    attack = adversarial_attack.get_attack(dataset, attack, model)
    attack.set_return_type('int') # Save as integer.
    adv_images = []
    labels = []
    for images, cls_labels in tqdm.tqdm(dataloader):
        adv_images.append(attack(images,cls_labels).cpu().detach())  #clean images have label 0
        labels.append(cls_labels.cpu().detach())
    adv_images = torch.cat(adv_images, dim=0)
    labels = torch.cat(labels, dim=0)

    adv_data = TensorDataset(adv_images.float()/255, labels)
    return adv_data

def create_adv_data_detector_classifier(dataloader,dataset, attack,detector , model , batch_size):
    adv_data = create_adv_data_detector(dataloader, detector,dataset, attack)
    dataloader= torch.utils.data.DataLoader(adv_data, batch_size= batch_size,shuffle=False)
    return create_adv_data_classifier(dataloader ,dataset, attack , model)

def create_adv_data_classifier_detector(dataloader,dataset, attack,detector , model , batch_size):
    #note here detector is sequential of (detector_base , classifier)
    adv_data = create_adv_data_classifier(dataloader ,dataset, attack , model)
    dataloader= torch.utils.data.DataLoader(adv_data, batch_size= batch_size,shuffle=False)
    return create_adv_data_detector_predict_adv_as_clean(dataloader, detector,dataset, attack)


# create adversarial image dataset, attack_list is a list of attacks. for every batch attack is randomly selected from attack_list
def create_adv_attack_multiple_attacks(dataloader,dataset,attack_list,model,sample_percent=[], batch_size=64):
    
    #change batch size of dataloader to 64
    # each batch randomly uses attack from the attack list. for better representation of each attack we use smaller batch size of 64
    dataloader= torch.utils.data.DataLoader(dataloader.dataset, batch_size= 512,shuffle=False)

    
    #create attack list 
    attacks = []
    for a  in attack_list:
        attacks.append(adversarial_attack.get_attack(dataset, a, model))
        attacks[-1].set_return_type('int') # Save as integer.
    
    #create torch multinomial distribution 
    if len(sample_percent) == 0:
        sample_percent = [1/len(attack_list) for i in range(len(attack_list))]
    sample_percent = torch.tensor(sample_percent)
    sample_percent = sample_percent/sample_percent.sum()


    adv_images = []
    labels = []
    for images, cls_labels in tqdm.tqdm(dataloader):
        attack_index =  int(torch.multinomial(sample_percent, num_samples=1).item())
        attack = attacks[attack_index]
        adv_images.append(attack(images,cls_labels).cpu().detach())  #clean images have label 0
        labels.append(cls_labels.cpu().detach())
    adv_images = torch.cat(adv_images, dim=0)
    labels = torch.cat(labels, dim=0)

    adv_data = TensorDataset(adv_images.float()/255, labels)
    return adv_data


def create_adv_attack_unsupervised(dataloader,dataset,attack_list,model,sample_percent=[], batch_size=64):
    dataloader= torch.utils.data.DataLoader(dataloader.dataset, batch_size= 512,shuffle=False)
    def hungarian_evaluate(model, dataloader):
        model.eval()
        accs = 0
        n_samples = 0
        device = next(model.parameters()).device 
        n_classes = 10
        predictions = []
        targets = []
        for iter_n, batch in enumerate(dataloader):
            #for each batch get predictions and save in list
            images = batch[0].to(device)
            target = batch[1].to(device)
            
            n_samples += target.shape[0]
            with torch.no_grad():
                outputs = model(images)
                predictions.append(torch.argmax(outputs, dim=1).detach().cpu())
                targets.append(target)
            
        predictions = torch.cat(predictions, dim=0)
        #convert predictions to list python
        #predictions = predictions.tolist()

        return predictions
    
    psuedo_labels = hungarian_evaluate(model, dataloader)
    #targets = dataloader.dataset.targets
    targets = dataloader.dataset.labels
    dataloader.dataset.labels = psuedo_labels
    


    adv_data = create_adv_attack_multiple_attacks(dataloader,dataset,attack_list,model,sample_percent=[], batch_size=64)
    dataloader.dataset.labels = targets
    return adv_data
