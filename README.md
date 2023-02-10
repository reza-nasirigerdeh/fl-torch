# fl-torch
**fl-torch** is a **federated learning** simulation framework to compare the performance of  **ResNets**,  **PreactResNets** and 
**VGGNets** using different normalization layers including **BatchNorm**, **LayerNorm**, **GroupNorm**, and **NoNorm** as baseline in federated environments.

**The code of KernelNorm, our proposed normalization layer, is coming soon.**

# Requirements
- Python +3.8
- PyTorch +1.11

# Installation
Clone the fl-torch repository:
```
git clone https://github.com/reza-nasirigerdeh/fl-torch
```
Install the dependencies:
```
pip3 install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html
```

# Options
### Dataset
| Dataset              | option                  |
|:---------------------|:------------------------|
| MNIST                | --dataset mnist         |
| Fashion-MNIST        | --dataset fashion_mnist |
| CIFAR-10             | --dataset cifar10       |
| CIFAR-100            | --dataset cifar100      |
| Imagenette 160-pixel | --dataset imagenette_160px |
| Imagenette 320-pixel | --dataset imagenette_320px |
| Imagewoof 160-pixel  | --dataset imagewoof_160px |
| Imagewoof 320-pixel  | --dataset imagewoof_320px |

### Preprocessing
| operation                                                          | option                           |
|:-------------------------------------------------------------------|:---------------------------------|
| resize train images (e.g. to 192x192)                              | --resize-train 192x192           |
| resize test images (e.g. to 192x192)                               | --resize-test 192x192            |
| random horizontal flip                                             | --random-hflip                   |
| random cropping with given size e.g. 32x32 and padding e.g. 4x4    | --random-crop 32x32-4x4          |
| random resized crop with given size (e.g. 128x128)                 | --random-resized-crop 128x128    |
| center crop test images with given size (e.g. 128x128)             | --center-crop 128x128            |
| normalize with mean (e.g. with 0.4625,0.4580,0.4298)               | --norm-mean 0.4625,0.4580,0.4298 |
| normalize with standard-deviation (e.g. with 0.2786,0.2755,0.2982) | --norm-std 0.2786,0.2755,0.2982  |

### Loss function
| Loss function | option               |
|:--------------|:---------------------|
| Cross-entropy | --loss cross_entropy |

### Optimizer
#### Stochastic gradient descent (SGD)
| Optimizer                                      | option                             |
|:-----------------------------------------------|:-----------------------------------|
| SGD                                            | --optimizer sgd                    |
| learning rate (e.g. 0.01)                      | --learning-rate  0.01              |
| momentum (e.g. 0.9)                            | --momentum 0.9                     |
| weight decay (e.g. 0.0001)                     | --weight-decay 0.0001              |
| dampening (e.g. 0.0)                           | --dampening 0.0                    |
| Nesterov momentum                              | --nesterov                         |


### Model
#### Toy models
| Model                                                      | option                                 |
|:-----------------------------------------------------------|:---------------------------------------|
| simple fully-connected model                               | --model fnn                            |
| simple convolutional model                                 | --model cnn                            |

#### VGG-6
| Model                                                | option                          |
|:-----------------------------------------------------|:--------------------------------|
| VGG-6 with no normalization layer                    | --model vgg6_nn                 |
| VGG-6 with batch normalization                       | --model vgg6_bn                 |
| VGG-6 with layer normalization                       | --model vgg6_ln                 |
| VGG-6 with group normalization of group size e.g. 32 | --model vgg6_gn --group-size 32 |

#### ResNet18/34/50/101/152
| Model                                                   | option                              |
|:--------------------------------------------------------|:------------------------------------|
| ResNet18 with no normalization                          | --model resnet18_nn                 |
| ResNet18 with batch normalization                       | --model resnet18_bn                 |
| ResNet18 with layer normalization                       | --model resnet18_ln                 |
| ResNet18 with group normalization of group size e.g. 32 | --model resnet18_gn --group-size 32 |

#### PreactResNet18/34/50/101/152
| Model                                                         | option                                     |
|:--------------------------------------------------------------|:-------------------------------------------|
| PreactResNet18 with no normalization                          | --model preact_resnet18_nn                 |
| PreactResNet18 with batch normalization                       | --model preact_resnet18_bn                 |
| PreactResNet18 with layer normalization                       | --model preact_resnet18_ln                 |
| PreactResNet18 with group normalization of group size e.g. 32 | --model preact_resnet18_gn --group-size 32 |

#### Note: For the other versions of ResNet and PreactResNet, please specify the corresponding version number instead of 18. 
**Example2**: ResNet50 with group normalization --> --model resnet50_gn \
**Example3**: PreactResNet34 with layer normalization --> --model preact_resnet34_ln 

### Federated setting
| Description                                                                   | option                 |
|:------------------------------------------------------------------------------|:-----------------------|
| number of clients                                                             | --clients 100          |
| number of labels per clients (label distribution)                             | --classes-per-client 5 |
| sample size distribution across clients (1.0 balanced, 0.0 highly imbalanced) | --balance-degree 1.0   |
| log level (e.g. debug)                                                        | --log-level debug      |

### Other
| Description                                                     | option               |
|:----------------------------------------------------------------|:---------------------|
| batch size (e.g. 32)                                            | --batch-size 32      |
| number of communication rounds (e.g. 1000)                      | --rounds 1000        |
| selection rate of clients (e.g. 0.2)                            | --selection-rate 0.2 |
| run number to have a separate result file for each run (e.g. 1) | --run 1              |
| log level (e.g. debug)                                          | --log-level debug    |
# Run
**Example1**: Train batch normalized version of VGG-6 on CIFAR-10 with cross-entropy loss function, SGD optimizer
with learning rate of 0.025 and momentum of 0.9, and batch size of 32 for 100 communication rounds, in a cross-silo federated environment
with 10 clients, 2 labels per client (highly non-iid), and balanced sample size distribution:

```
python3 simulate.py --dataset cifar10 --model vgg6_bn --optimizer sgd --momentum 0.9 \
                    --loss cross_entropy --learning-rate 0.025 --batch-size 32 --rounds 100 \
                    --clients 10 --classes-per-client 2 --balance-degree 1.0
                    
```

**Example2**: Train layer normalized version of ResNet-34 on imagenette-160-pixel with SGD optimizer, learning rate of 0.005, and batch size of 16 for 100 rounds. For preprocessing, apply random-resized-crop of shape 128x128 and random horizontal flipping to the train images, and resize test images to 160x160 first, and then, center crop them to 128x128, finally normalize them with the mean and std of imagenet.
The cross-silo federated environment contains 5 clients, with 10 labels per client (IID), and balance degree of 0.75 (relatively imbalanced):

```
python3 simulate.py --dataset imagenette_160px --random-hflip \
                    --random-resized-crop 128x128 --resize-test 160x160 --center-crop 128x128  \
                    --norm-mean 0.485,0.456,0.406  --norm-std 0.229,0.224,0.225 \
                    --model resnet34_ln --optimizer sgd --momentum 0.0 \
                    --learning-rate 0.005 --batch-size 16 --rounds 100 \
                    --clients 5 --classes-per-client 10 --balance-degree 0.75
```

**Example3**: Train group normalized version of PreactResNet-18 with group size of 32 on CIFAR-100 with SGD optimizer, learning rate of 0.01, and batch size of 16 for 200 rounds. For preprocessing, apply random horizontal flipping and cropping with padding 4x4 to the images. Federated environment consists of 100 clients, 
with 10 labels per client (highly non-iid), balanced sample distribution, and client selection rate of 0.2 (cross-device).

```
python3 simulate.py --dataset cifar100 --random-crop 32x32-4x4 \
                    --random-hflip --model preact_resnet18_gn --group-size 32  \
                    --optimizer sgd --learning-rate 0.01 \
                    --batch-size 16 --rounds 200 \
                    --clients 100 --classes-per-client 10 --balance-degree 1.0 --selection-rate 0.2
```

## Citation
If you use **fl-torch** in your study, please cite the following paper: <br />
   ```
@inproceedings{
nasirigerdeh2023knconvnets-ppml,
title={Kernel Normalized Convolutional Networks for Privacy-Preserving Machine Learning},
author={Reza Nasirigerdeh and Javad Torkzadehmahani and Daniel Rueckert and Georgios Kaissis},
booktitle={First IEEE Conference on Secure and Trustworthy Machine Learning},
year={2023},
url={https://openreview.net/forum?id=pyfGjjDmrC}
}
   ```