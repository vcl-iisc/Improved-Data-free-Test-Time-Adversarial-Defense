
# DAD++: Improved Data-free Test Time Adversarial Defense. (Official Implementation) 
(Under Submission)

## How to run this code
Following commands explain test time defense of target model Resnet18 trained on CIFAR10 dataset. Arbitary model is also Resnet18. Arbitrary dataset is Fmnist. 

### Train Target model
``` 
python train_model.py --dataset cifar10 --batch_size 32 --lr 0.01 --image_size 32 --epochs 100 --model_name resnet18  --save_path <checkpoint_output_path> --wandb
``` 

### Train Arbitary model

``` 
python train_model.py --dataset fmnist --batch_size 64 --lr 0.01 --image_size 32 --epochs 50 --model_name resnet18 --save_path <checkpoint_output_path> --wandb
``` 

### Train Arbitrary detector 
``` 
python train_arbitary_detector.py --name source_detector --dataroot clean_data/fmnist --dataset fmnist --batch_size 128 --model_name resnet18 --model_path <arbitar model path> --attack pgd --gpu 0 --method vanila --epochs 10  --seed 0 --use_wandb
``` 
Arbitary detector is saved in same directory as that of arbitary model. 


### Compute combined performance
This command evaluates the performance of Target model with DAD defense. For each test set, we adapt the source detector to target detector. Using the target detector and correction module, the clean and adversarial accuracy of model is computed.
 ```
python combined.py --dataset cifar10 --batch_size 64 --model_name resnet18 --model_path <target_model_path> --detector_path <arbitary_detector_path> --attacks pgd --method vanila --gpu 0 --droprate 0.005 --seed 0 --lr 0.005 --epochs 10 --s_dataset fmnist --ent_par 0.8 --cls_par 0.3 --correction_batch_size 256 --r_range 16 --soft_detection_r 32 --log_path ./logs/logs_imbalanced.txt --pop 10 --retrain_detector --recreate_adv_data --use_wandb 
 ``` 
 
 ## Citing
If you use this code, please cite our work:

```
@article{Nayak2022DADDA,
  title={DAD: Data-free Adversarial Defense at Test Time},
  author={Gaurav Kumar Nayak and Ruchit Rawal and Anirban Chakraborty},
  journal={2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2022},
  pages={3788-3797}
}
```
