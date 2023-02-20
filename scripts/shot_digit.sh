python combined.py --dataset mnist  --batch_size 64  --model_name s2m  --model_path data/shot_digit/mnist/s2m/target_B_par_0.1.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks pgd --method shot_digit --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb &

python combined.py --dataset mnist  --batch_size 64  --model_name s2m  --model_path data/shot_digit/mnist/s2m/target_B_par_0.1.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks auto_attack --method shot_digit --gpu 1 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb &


python combined.py --dataset mnist  --batch_size 64  --model_name s2m  --model_path data/shot_digit/mnist/s2m/target_B_par_0.1.pt --detector_path data/source/cifar10/resnet18/cifar10_pgd_seed_0_source_detector.pt --attacks pgd --method shot_digit --gpu 2 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset cifar10   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb &


python combined.py --dataset mnist  --batch_size 64  --model_name s2m  --model_path data/shot_digit/mnist/s2m/target_B_par_0.1.pt --detector_path data/source/cifar10/resnet18/cifar10_pgd_seed_0_source_detector.pt --attacks auto_attack --method shot_digit --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset cifar10   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb 




#python combined.py --dataset mnist  --batch_size 64  --model_name u2m  --model_path data/shot_digit/mnist/u2m/target_B_par_0.1.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks pgd --method shot_digit --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb &
#python combined.py --dataset mnist  --batch_size 64  --model_name u2m  --model_path data/shot_digit/mnist/u2m/target_B_par_0.1.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks auto_attack --method shot_digit --gpu 1 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb &
#python combined.py --dataset mnist  --batch_size 64  --model_name u2m  --model_path data/shot_digit/mnist/u2m/target_B_par_0.1.pt --detector_path data/source/cifar10/resnet18/cifar10_pgd_seed_0_source_detector.pt --attacks pgd --method shot_digit --gpu 2 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset cifar10   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb

#wait
#python combined.py --dataset mnist  --batch_size 64  --model_name u2m  --model_path data/shot_digit/mnist/u2m/target_B_par_0.1.pt --detector_path data/source/cifar10/resnet18/cifar10_pgd_seed_0_source_detector.pt --attacks auto_attack --method shot_digit --gpu 1 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset cifar10   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb &
#python combined.py --dataset mnist  --batch_size 64  --model_name u2m  --model_path data/shot_digit/mnist/u2m/target_B_par_0.1.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks pgd --method shot_digit --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb &
#python combined.py --dataset mnist  --batch_size 64  --model_name u2m  --model_path data/shot_digit/mnist/u2m/target_B_par_0.1.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks auto_attack --method shot_digit --gpu 2 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb


#wait
#python combined.py --dataset mnist  --batch_size 64  --model_name s2m  --model_path data/shot_digit/mnist/u2m/target_B_par_0.1.pt --detector_path data/source/cifar10/resnet18/cifar10_pgd_seed_0_source_detector.pt --attacks pgd --method shot_digit --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset cifar10   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb &
#python combined.py --dataset mnist  --batch_size 64  --model_name u2m  --model_path data/shot_digit/mnist/u2m/target_B_par_0.1.pt --detector_path data/source/cifar10/resnet18/cifar10_pgd_seed_0_source_detector.pt --attacks auto_attack --method shot_digit --gpu 1 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset cifar10   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb &
#python combined.py --dataset usps  --batch_size 64  --model_name m2u  --model_path data/shot_digit/usps/m2u/target_B_par_0.1.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks pgd --method shot_digit --gpu 2 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb

#wait 
#python combined.py --dataset usps  --batch_size 64  --model_name m2u  --model_path data/shot_digit/usps/m2u/target_B_par_0.1.pt --detector_path data/source/fmnist/resnet18/fmnist_pgd_seed_0_source_detector.pt --attacks auto_attack --method shot_digit --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset fmnist   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb &
#python combined.py --dataset usps  --batch_size 64  --model_name m2u  --model_path data/shot_digit/usps/m2u/target_B_par_0.1.pt --detector_path data/source/cifar10/resnet18/cifar10_pgd_seed_0_source_detector.pt --attacks pgd --method shot_digit --gpu 1 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset cifar10   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb &
#python combined.py --dataset usps  --batch_size 64  --model_name m2u  --model_path data/shot_digit/usps/m2u/target_B_par_0.1.pt --detector_path data/source/cifar10/resnet18/cifar10_pgd_seed_0_source_detector.pt --attacks auto_attack --method shot_digit --gpu 2 --droprate 0.005 --seed 0 --lr 0.005 --epochs 5 --s_dataset cifar10   --ent_par 0.8  --cls_par 0.3 --recreate_adv_data --retrain_detector --correction_batch_size 512  --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_shot_digit.txt --use_wandb

