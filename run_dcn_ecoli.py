import torch
import torch.utils.data as data
# from torchvision import datasets, transforms
import numpy as np
import argparse
from lib.stackedDAE import StackedDAE
from lib.dec import DEC
from lib.utils import write_log
import os
from  sklearn.utils import shuffle
import sklearn
from data_loader import myDataset
from tensorboardX import SummaryWriter
from lib.dcn import DeepClusteringNetwork

import warnings
warnings.filterwarnings('ignore')


datasetname="ecoli"
repeat = 1
batch_size = 50

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--sdae_pre_lr', type=float, default=0.01, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--sdae_fit_lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--dcn_lr', type=float, default=0.00001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--cuda', type=str, default='0', metavar='N')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if os.path.exists('logs/'+datasetname)==False:
        os.makedirs('logs/'+datasetname)
    writer = SummaryWriter(log_dir='logs/'+datasetname)
    log_dir='logs/'+datasetname
    train_path = 'dataset/ecoli/ecoli.pkl'
    # test_path =  '/DATACENTER1/xiao.peng/DCN_keras-master/dataset/RCV1/Processed/data-0.pkl'

    # all_path='./dataset/wine/wine.data'
    for i in range(1, repeat+1):
        sdae_savepath = ("model/sdae-dcn-run-"+datasetname+"-%d.pt" % i)
        # sdae_savepath="D:\code\dec-pytorch\model\sdae-run-wine-1.pt"
        if os.path.exists(sdae_savepath)==False:
            print("Experiment #%d" % i)
            write_log("Experiment #%d" % i,log_dir)

            train_data=myDataset(train_path,-1, '.pkl')
            # test_data=myDataset(test_path,-1, '.pkl')
            train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                           collate_fn=train_data.collate_fn,num_workers=4)
            # test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True,
            #                                collate_fn=train_data.collate_fn,num_workers=4)


            # pretrain
            sdae = StackedDAE(input_dim=7, z_dim=3, binary=False,
                encodeLayer=[5], decodeLayer=[5], activation="relu",
                dropout=0,log_dir=log_dir)
            sdae.cuda()
            # print(sdae)
            sdae.pretrain(train_loader, lr=args.sdae_pre_lr, batch_size=batch_size,
                num_epochs=50, corrupt=0.2, loss_type="mse")
            torch.cuda.empty_cache()
            sdae.fit(train_loader, lr=args.sdae_fit_lr, num_epochs=10, corrupt=0.2, loss_type="mse")
            sdae.save_model(sdae_savepath)
            del sdae
            del train_loader
            # del test_loader
            del train_data
            # del test_data
            torch.cuda.empty_cache()
        if os.path.exists("model/dcn-run-"+datasetname+"-%d.pt" % i)==False:
        # if True:
            # finetune

            fit_train=myDataset(train_path,-1, '.pkl')
            # fit_test = myDataset(test_path,-1)

            # X = fit_train.dataSource[:,:-1]
            y = fit_train.dataSource[:,-1]
            # X,y=torch.from_numpy(np.array(X,dtype=np.float32)),torch.from_numpy(np.array(y,dtype=np.float32))
            # y=y.long()
            # n_centroids=len(np.unique(y.numpy()))
            train_loader = data.DataLoader(dataset=fit_train, batch_size=batch_size, shuffle=True,
                                           collate_fn=fit_train.collate_fn, num_workers=4)
            dcn = DeepClusteringNetwork(input_dim=7, z_dim=3, n_centroids=7, binary=False,
                                        encodeLayer=[5], decodeLayer=[5], activation="relu",
                                        dropout=0,writer=writer,log_dir=log_dir,lambd=0.1)
            print(dcn)
            dcn.load_model(sdae_savepath)
            dcn.fit(train_loader, lr=args.dcn_lr, batch_size=batch_size, num_epochs=10)
            dcn_savepath = ("model/dcn-run-"+datasetname+"-%d.pt" % i)
            # dcn.save_model(dcn_savepath)
            # del X
            # del y
            del train_loader
            del dcn
            torch.cuda.empty_cache()
