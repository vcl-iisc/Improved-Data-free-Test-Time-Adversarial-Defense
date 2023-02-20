import torchattacks


def get_attack_shot_office_method(attack , model):
    if attack == 'pgd':
        #eps, alpha, steps = 3/255, 1/255, 20       
        eps, alpha , steps = 8/255 , 0.007, 40
        attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    
    elif attack == 'auto_attack':
        eps = 8/255
        attack = torchattacks.AutoAttack(model, eps=eps, n_classes=31, version='standard')
    return attack

def get_attack_shot_digit_method(attack , model):
    '''
    Return attack 
    '''

    if attack == 'pgd':   
        eps, alpha, steps = 0.3, 0.01, 40
        attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)

    elif attack == 'bim':
        eps, alpha, steps = 0.3, 0.03, 40
        attack = torchattacks.BIM(model, eps=eps, alpha=alpha, steps=steps)

    elif attack == 'fgsm':
        eps = 0.3
        attack = torchattacks.FGSM(model, eps=eps)

    elif attack == 'auto_attack':
        eps = 0.3
        attack = torchattacks.AutoAttack(model, eps=eps, n_classes=10, version='standard')

    return attack

    
# return adversarial attack with test time parameters
def get_attack(dataset, attack, model):

    if dataset in  [ "shot_office" , "dine_office_home", "decision_office_home" , "dine_office" , "decision_office"]:
        return get_attack_shot_office_method(attack, model)
    elif dataset == "shot_digit":
        return get_attack_shot_digit_method(attack, model)
    
    if attack in [ 'pgd' , 'bim','fgsm' ]:
        if dataset == "cifar10":
            print('Attack using pgd on cifar10 dataset')
            eps, alpha, steps = 8 / 255, 2 / 255, 20
        elif dataset == "cifar100":
            print('Attack using pgd on cifar100 dataset')
            eps, alpha, steps = 8 / 255, 2 / 255, 20
        elif dataset == "mnist":
            print('Attack using pgd on mnist dataset')
            eps, alpha, steps = 0.3, 0.01, 100
        elif dataset == "fmnist":
            print('Attack using pgd on fmnist dataset')
            eps, alpha, steps = 0.2, 0.02, 100
        elif dataset == "svhn":
            print("Attack using pgd on svhn dataset")
            #eps, alpha, steps = 4 / 255, 2 / 255, 20
            eps, alpha, steps = 8/255,2/255,20
        elif dataset =="stl":
            #values copied from paper https://arxiv.org/pdf/2105.14240.pdf ref: page10
            eps, alpha, steps = 8/255, 2/255, 20
        elif dataset == 'usps':
            #eps, alpha, steps = 4/255,2/255,20
            eps, alpha, steps = 0.3, 0.01, 40
        elif dataset == "office":
            eps, alpha, steps = 3/255, 1/255, 20 
        #values copied from paper https://arxiv.org/pdf/2105.14240.pdf ref: page10
        elif dataset == "tiny_imagenet":
            #ref : https://arxiv.org/pdf/1905.11971.pdf
            #eps , alpha , steps = 0.007 ,0.003 , 50    
            eps , alpha , steps = 8/255 ,0.01 , 100    
        elif dataset == "oxford_pet":
            eps , alpha , steps = None ,0.003 , 50    
        elif dataset  == "cub":
            eps, alpha, steps = 3/255, 1/255, 20       
        if attack == 'pgd':
            attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
        elif attack == 'bim':
            attack = torchattacks.BIM(model, eps=eps, alpha=alpha, steps=steps)
        elif attack == 'fgsm':
            attack = torchattacks.FGSM(model, eps=eps)
            
    elif attack == 'auto_attack':
        if dataset == "cifar10":
            print('Attack using AA on cifar10 dataset')
            eps = 8 / 255
        elif dataset == "cifar100":
            print('Attack using AA on cifar100 dataset')
            eps, alpha, steps = 8 / 255, 2 / 255, 20
        elif dataset == 'mnist':
            print('Attack using AA on mnist dataset')
            eps = 0.3

        elif dataset == 'fmnist':
            print('Attack using AA on fmnist dataset')
            eps = 0.2
        
        elif dataset == 'svhn':
            eps = 8/255

        elif dataset == 'stl':
            eps = 8/255
        
        elif dataset == 'usps':
            #eps, alpha, steps = 4/255,2/255,20
            eps, alpha, steps = 0.3, 0.01, 40
        elif dataset == "office":
            #TODO which attack parameters to use ?
            
            #eps, alpha, steps = 3/255, 1/255, 20       
            eps = 8/255
        elif dataset == "tiny_imagenet":
            #eps = 0.007
            eps = 8/255
        elif dataset == "oxford_pet":
            eps = None
        elif dataset  == "cub":
            eps= 3/255
        attack = torchattacks.AutoAttack(model, eps=eps, n_classes=10, version='standard')
    else:
        #throw exception
        raise Exception("unknown attack type")

        

    return attack
