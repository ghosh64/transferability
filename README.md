# Improving Transferability of Network Intrusion Detection in a Federated Learning Setup

Code base of the paper Improving Transferability of Network Intrusion Detection in a Federated Learning Setup @ IEEE International Conference on Machine Learning for Communication and Networks(ICMLCN) 2024. 

# Environment

To setup the environment:

conda env create -f environment.yml

# Training and Testing

Run main.py to start training and testing. This script can be run in two modes: 

1) With 5 nodes where each node contains the same distribution of data
2) With 11 nodes where each node contains one type of attack and benign data

To start training, run: python main.py

To run the bootstrapped version of (1) and (2), run main_bootstrap.py

Other supported parameters include:

  --n_classes    2,5(default)

  --algo         FedAvg(default)

--device       0(default) GPU

--silobn       True(default),False

--weight_path  weights/

--tb_path      tensorboard/

--transferability True/False(default)

--cf           path of the saved confusion matrix

--n_rounds     number of communication rounds

--tmp          enable temporal averaging of data as a preprocessing step

--diff         enable differential inputs as a preprocessing step

--epochs       no of epochs of training for each device

