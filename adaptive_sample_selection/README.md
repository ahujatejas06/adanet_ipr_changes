# Adanet IPR Implementation with Adaptive Sample Selection

To run and reproduce the results in the paper (This script is written to be run on Linux (Tested on Ubuntu 24.04 LTS))
1. Setup miniconda/anaconda on your machine as the code uses tensorflow 1 with python 3.6
2. Change the directory to the repo folder
3. Run this command to make the bash script executable on your linux machine
```bash
chmod +x setup_environment.sh
```
4. Setup the environment on your machine, this will install the requirements needed and setup and activate the conda environment
``` bash
source setup_environment.sh
```
5. Use this Command to download the dataset
```bash
python cifar10.py --data_dir=./dataset/cifar10/ --dataset_seed=1
python cifar10.py --data_dir=./dataset/cifar10/ --dataset_seed=2
python cifar10.py --data_dir=./dataset/cifar10/ --dataset_seed=3

```
6. Download the model weights from this link and save the weights folder(rename it to weights) in the root of the repo. The Weights folder should have all the weights without any subdirectory
```
https://drive.google.com/drive/folders/1CfxE0ZuWTZ_kjLsKD9bmvEfR-5hwCuSF?usp=sharing
```
7. To test the code on the dataset seed 1 use this code. Possible seed options = 1,2,3
```bash
python test_cifar.py --dataset=cifar10 --data_dir=./dataset/cifar10/ --log_dir=./weights/ --dataset_seed=1
```

8. To train the model in continuation of the currently trained weights, use this
``` bash
python train_cifar.py --dataset=cifar10 --data_dir=./dataset/cifar10/ --log_dir=./weights/ --num_epochs=200 --epoch_decay_start=1500 --aug_flip=True --aug_trans=True --dataset_seed=1
```

9. If you want to train the model from the beginning empty the weights folder and use the command in 8



## References
This Repository is reference to https://github.com/qinenergy/adanet.git

## Citation
```
@article{wang2019semi,
  title={Semi-Supervised Learning by Augmented Distribution Alignment},
  author={Wang, Qin and Li, Wen and Van Gool, Luc},
  journal={arXiv preprint arXiv:1905.08171},
  year={2019}
}
```