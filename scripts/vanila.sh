python combined.py --dataset cifar10 --batch_size 64 --model_name resnet18 --model_path data/source/cifar10/resnet18/net.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks pgd --method vanila --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist --ent_par 0.8 --cls_par 0.3  --correction_batch_size 128 --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_vanila.txt --pop 10  --recreate_adv_data --retrain_detector --use_wandb
python combined.py --dataset cifar10 --batch_size 64 --model_name resnet18 --model_path data/source/cifar10/resnet18/net.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks auto_attack --method vanila --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist --ent_par 0.8 --cls_par 0.3  --correction_batch_size 128 --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_vanila.txt --pop 10  --recreate_adv_data --retrain_detector --use_wandb

python combined.py --dataset cifar10 --batch_size 64 --model_name resnet18 --model_path data/source/cifar10/resnet18/net.pt --detector_path data/source/fmnist/resnet18/mnist_pgd_seed_0_source_detector.pt --attacks pgd --method vanila --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist --ent_par 0.8 --cls_par 0.3  --correction_batch_size 128 --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_vanila.txt --pop 10  --recreate_adv_data --retrain_detector --use_wandb
python combined.py --dataset cifar10 --batch_size 64 --model_name resnet18 --model_path data/source/cifar10/resnet18/net.pt --detector_path data/source/fmnist/resnet18/mnist_pgd_seed_0_source_detector.pt --attacks auto_attack --method vanila --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist --ent_par 0.8 --cls_par 0.3  --correction_batch_size 128 --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_vanila.txt --pop 10  --recreate_adv_data --retrain_detector --use_wandb


python combined.py --dataset svhn --batch_size 64 --model_name resnet18 --model_path data/source/svhn/resnet18/net.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks pgd --method vanila --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist --ent_par 0.8 --cls_par 0.3  --correction_batch_size 128 --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_vanila.txt --pop 10  --recreate_adv_data --retrain_detector --use_wandb
python combined.py --dataset svhn --batch_size 64 --model_name resnet18 --model_path data/source/svhn/resnet18/net.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks auto_attack --method vanila --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist --ent_par 0.8 --cls_par 0.3  --correction_batch_size 128 --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_vanila.txt --pop 10  --recreate_adv_data --retrain_detector --use_wandb

python combined.py --dataset svhn --batch_size 64 --model_name resnet18 --model_path data/source/svhn/resnet18/net.pt --detector_path data/source/cifar10/resnet18/cifar10_pgd_seed_0_source_detector.pt --attacks pgd --method vanila --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset cifar10 --ent_par 0.8 --cls_par 0.3  --correction_batch_size 128 --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_vanila.txt --pop 10  --recreate_adv_data --retrain_detector --use_wandb
python combined.py --dataset svhn --batch_size 64 --model_name resnet18 --model_path data/source/svhn/resnet18/net.pt --detector_path data/source/cifar10/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks auto_attack --method vanila --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset cifar10 --ent_par 0.8 --cls_par 0.3  --correction_batch_size 128 --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_vanila.txt --pop 10  --recreate_adv_data --retrain_detector --use_wandb