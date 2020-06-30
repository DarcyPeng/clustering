import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import numpy as np
import argparse
from lib.stackedDAE import StackedDAE
from lib.dcn import DeepClusteringNetwork
from lib.datasets import MNIST
from lib.utils import write_log
import os
import warnings
warnings.filterwarnings('ignore')

datasetname = "mnist"
# datasetname="cifar"
repeat = 1
batch_size = 64

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--sdae_lr', type=float, default=0.1, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--dcn_lr', type=float, default=0.01, metavar='N',
                        help='learning rate for training (default: 0.001)')
    args = parser.parse_args()
    log_dir = 'logs/dec-' + datasetname
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    for i in range(1, repeat+1):
        sdae_savepath = ("model/sdae-run-"+datasetname+"-%d.pt" % i)
        if os.path.exists(sdae_savepath)==False:
            print("Experiment #%d" % i)
            write_log("Experiment #%d" % i,log_dir)
            train_loader=None
            test_loader=None
            if datasetname=='mnist':
                train_loader = torch.utils.data.DataLoader(
                    MNIST('./dataset/mnist', train=True, download=True),
                    batch_size=batch_size, shuffle=True, num_workers=0)
                # test_loader = torch.utils.data.DataLoader(
                #     MNIST('./dataset/mnist', train=False),
                #     batch_size=batch_size, shuffle=False, num_workers=0)
            elif datasetname=='cifar':
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                trainset = datasets.CIFAR10(
                    root='./dataset/cifar', train=True, download=False, transform=transform)  # download=True会通过官方渠道下载
                train_loader = torch.utils.data.DataLoader(
                    trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                testset = datasets.CIFAR10(
                    root='./dataset/cifar', train=False, download=False, transform=transform)
                test_loader = torch.utils.data.DataLoader(
                    testset, batch_size=batch_size, shuffle=False, num_workers=2)
            else:
                exit()
            # pretrain
            sdae = StackedDAE(input_dim=784, z_dim=10, binary=False,
                              encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu",
                              dropout=0,log_dir=log_dir)
            print(sdae)
            sdae.pretrain(train_loader, lr=args.sdae_lr, batch_size=batch_size,
                num_epochs=300, corrupt=0.2, loss_type="mse")
            sdae.fit(train_loader, lr=args.sdae_lr, num_epochs=500, corrupt=0.2, loss_type="mse")
            sdae.save_model(sdae_savepath)
        if os.path.exists("model/dcn-run-mnist-%d.pt" % i)==False:
            # finetune
            fit_train=None
            fit_test = None
            X=None
            y=None
            if datasetname=='mnist':
                train_loader = torch.utils.data.DataLoader(
                    MNIST('./dataset/mnist', train=True, download=False),
                    batch_size=batch_size, shuffle=True, num_workers=0)
            elif datasetname=='cifar':
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                fit_train = datasets.CIFAR10(root='./dataset/cifar', train=True, download=False, transform=transform)
                fit_test = datasets.CIFAR10(root='./dataset/cifar', train=False, download=False, transform=transform)
                X = fit_train.train_data
                y = fit_train.train_labels
                X=np.array(X,dtype=np.float32)
                X=np.transpose(X,(0,3,1,2))
            # X,y=X.float(),y.long()
            # X=X.view(X.size(0), -1).float()
            #
            # y=y.long()
            dcn = DeepClusteringNetwork(input_dim=784, z_dim=10, n_centroids=10,
                      encodeLayer=[500, 500, 2000], decodeLayer=[2000,500,500], activation="relu", dropout=0,lambd=1)
            print(dcn)
            dcn.load_model(sdae_savepath)
            dcn.fit(train_loader, lr=args.dcn_lr, batch_size=batch_size, num_epochs=100)
            dcn_savepath = ("model/dcn-run-mnist-%d.pt" % i)
            dcn.save_model(dcn_savepath)