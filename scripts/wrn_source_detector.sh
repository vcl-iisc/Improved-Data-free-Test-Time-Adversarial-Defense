
python combined1.py --dataset tiny_imagenet --batch_size 32 --model_name resnet34 --model_path data/source/tinyimagenet/resnet34/net.pt --detector_path data/svhn/wrn_16_1/svhn_WRN-16-1_pgd_source_detector.pt --attacks pgd --method vanila --gpu 1 --droprate 0.005 --seed 0 --lr 0.005 --epochs 10 --s_dataset svhn --ent_par 0.8 --cls_par 0.3 --correction_batch_size 256 --r_range 32 --soft_detection_r 64 --log_path ./logs/logs_wrn_source.txt --pop 10 --retrain_detector --recreate_adv_data --use_wandb &
python combined1.py --dataset tiny_imagenet --batch_size 32 --model_name resnet34 --model_path data/source/tinyimagenet/resnet34/net.pt --detector_path data/svhn/wrn_16_1/svhn_WRN-16-1_pgd_source_detector.pt --attacks auto_attack --method vanila --gpu 2 --droprate 0.005 --seed 0 --lr 0.005 --epochs 10 --s_dataset svhn --ent_par 0.8 --cls_par 0.3 --correction_batch_size 256 --r_range 32 --soft_detection_r 64 --log_path ./logs/logs_wrn_source.txt --pop 10 --retrain_detector --recreate_adv_data --use_wandb 

wait


python combined1.py --dataset cub --batch_size 32 --model_name resnet34 --model_path data/source/cub/resnet34/net.pt --detector_path data/svhn/wrn_16_1/svhn_WRN-16-1_pgd_source_detector.pt --attacks auto_attack --method vanila --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 10 --s_dataset svhn --ent_par 0.8 --cls_par 0.3 --correction_batch_size 256 --r_range 112 --soft_detection_r 224 --log_path ./logs/logs_wrn_source.txt --pop 10 --retrain_detector --recreate_adv_data --use_wandb  &
python combined1.py --dataset cub --batch_size 32 --model_name resnet34 --model_path data/source/cub/resnet34/net.pt --detector_path data/svhn/wrn_16_1/svhn_WRN-16-1_pgd_source_detector.pt --attacks pgd --method vanila --gpu 1 --droprate 0.005 --seed 0 --lr 0.005 --epochs 10 --s_dataset svhn --ent_par 0.8 --cls_par 0.3 --correction_batch_size 256 --r_range 112 --soft_detection_r 224 --log_path ./logs/logs_wrn_source.txt --pop 10 --retrain_detector --recreate_adv_data --use_wandb  