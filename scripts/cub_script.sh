


python3 combined.py --dataset cub --batch_size 64 --model_name resnet34 --model_path data/source/cub/resnet34/net_finetuned.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks pgd --method vanila --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 10 --s_dataset fmnist --ent_par 0.8 --cls_par 0.3 --retrain_detector --correction_batch_size 64 --pop 10 --recreate_adv_data --r_range 128 --use_wandb --soft_detection_r=224 --log_path=./logs/logs_cub.txt




python3 combined.py --dataset cub --batch_size 64 --model_name resnet34 --model_path data/source/cub/resnet34/net_finetuned.pt --detector_path data/source/mnist/resnet18/mnist_pgd_seed_0_source_detector.pt --attacks pgd --method vanila --gpu 1 --droprate 0.005 --seed 0 --lr 0.005 --epochs 10 --s_dataset mnist --ent_par 0.8 --cls_par 0.3 --retrain_detector --correction_batch_size 64 --pop 10 --recreate_adv_data --r_range 128 --use_wandb --soft_detection_r=224 --log_path=./logs/logs_cub.txt



python3 combined.py --dataset cub --batch_size 64 --model_name resnet34 --model_path data/source/cub/resnet34/net_finetuned.pt --detector_path data/source/cifar10/resnet18/cifar10_pgd_seed_0_source_detector.pt --attacks pgd --method vanila --gpu 2 --droprate 0.005 --seed 0 --lr 0.005 --epochs 10 --s_dataset cifar10 --ent_par 0.8 --cls_par 0.3 --retrain_detector --correction_batch_size 64 --pop 10 --recreate_adv_data --r_range 128 --use_wandb --soft_detection_r=224 --log_path=./logs/logs_cub.txt


