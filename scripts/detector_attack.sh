python combined1.py --dataset svhn --batch_size 64 --model_name resnet18 --model_path data/dafl/svhn/resnet18/net.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks pgd --method dafl --gpu 2 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist --ent_par 0.8 --cls_par 0.3 --retrain_detector --correction_batch_size 128 --pop 10 --recreate_adv_data --use_wandb

python combined1.py --dataset cifar10 --batch_size 64 --model_name resnet18 --model_path data/source/cifar10/resnet18/net.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks pgd --method vanila --gpu 2 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist --ent_par 0.8 --cls_par 0.3 --retrain_detector --correction_batch_size 128 --pop 10 --recreate_adv_data --use_wandb

python combined1.py --dataset cifar10 --batch_size 64 --model_name resnet18 --model_path data/source/cifar10/resnet18/net.pt --detector_path data/source/mnist/resnet18/mnist_pgd_seed_0_source_detector.pt --attacks pgd --method vanila --gpu 2 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset mnist --ent_par 0.8 --cls_par 0.3 --retrain_detector --correction_batch_size 128 --pop 10 --recreate_adv_data --use_wandb


python combined1.py --dataset cifar10 --batch_size 64 --model_name resnet18 --model_path data/dafl/cifar10/resnet18/net.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks pgd --method dafl --gpu 2 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist --ent_par 0.8 --cls_par 0.3 --retrain_detector --correction_batch_size 128 --pop 10 --recreate_adv_data --use_wandb

