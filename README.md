Setup
-----------------------

<!--- 1. The code is tested with python3 and TensorFlow v1.10 and v1.12 (TensorFlow v1.8 is not compatible). The following packages are required.

```
sudo apt-get install python3-pip python3-dev
sudo pip3 install --upgrade pip
sudo pip3 install pillow numpy scipy pandas h5py tensorflow numba posix_ipc matplotlib
```
(Akhilan: above doesn't work for me... pip3 seems to have [problem](https://github.com/pypa/pip/issues/5240))
what works for me is:
--->

1. The code is tested with python3 and TensorFlow v1.10 and v1.12 (TensorFlow v1.8 is not compatible). The following packages are required.
```
conda create --name deepcert python=3.6
source activate deepcert
conda install pillow numpy scipy pandas h5py tensorflow numba posix_ipc matplotlib
```

2. Ready to run our project.
```
python pymain.py
```
It will provide the results of mnist_cnn_8layer_5_3 and lenet with sigmoid, tanh and arctan as activation function.

Results of robustness certificate and runtime are saved into `log_pymain_{timestamp}.txt`

How to Run
--------------------
We have provided a interfacing file `pymain.py` to compute certified bounds. This file contains a function `run_cnn` that computes DeepCert bounds and runtime averaged over a given number of samples.
```
Usage: run_cnn(file_name, n_samples, norm, core, activation, cifar)
```
* file_name: file name of network to certify
* n_samples: number of samples to certify
* norm: p-norm, 1,2 or i (infinity norm)
* core: True or False, use True for networks with only convolution and pooling
* activation: followed by a activation function of network- use ada for adaptive relu bounds (relu, ada, tanh, sigmoid or arctan)
* cifar: True or False, uses MNIST if False

Training your own models and evaluating DeepCert.
-----------------------
We provide pre-trained models [here](https://www.dropbox.com/s/mhe8u2vpugxz5ed/models.zip?dl=0). The pre-trained models use the follwing naming convention:

Pure CNN Models:
```
{dataset}_cnn_{number of layers}layer_{filter size}_{kernel size}[_{non ReLU activation}]
Example: mnist_cnn_4layer_10_3
Example: cifar_cnn_5layer_5_3_tanh
```

General CNN Models:
```
{dataset}_cnn_{number of layers}layer[_{non ReLU activation}]
Example: mnist_cnn_7layer
Example: mnist_cnn_7layer_tanh
```

Fully connected Models:
```
{dataset}_{number of layers}layer_fc_{layer size}
Example: mnist_2layer_fc_20
```

Resnet Models:
```
{dataset}_resnet_{number of residual blocks}[_{non ReLU activation}]
Example: mnist_resnet_3
Example: mnist_resnet_4_tanh
```
LeNet Models:
```
{dataset}_cnn_lenet[_nopool][_{non ReLU activation}]
Example: mnist_cnn_lenet_tanh
Example: mnist_cnn_lenet_nopool
```
If you want to run CNN-Cert, you can refer to https://github.com/AkhilanB/CNN-Cert.git. Our environment "deepcert" can also support CNN-Cert.

If you want to run FROWN, you can use cnn_to_mlp.py to transfer pure cnns to mlps, then clone the tool FROWN:

FROWN's code is tested with python 3.7.3, Pytorch 1.1.0 and CUDA 9.0. Run the following to install pytorch and its CUDA toolkit:
```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

Then clone this repository:

```
git clone https://github.com/ZhaoyangLyu/FROWN.git
cd FROWN
```
Then refer to the README.md in folder "FROWN".

