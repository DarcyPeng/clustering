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

import warnings
warnings.filterwarnings('ignore')

datasetname="rcv1"
repeat = 1
batch_size = 512

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--sdae_lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--dec_lr', type=float, default=0.0001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--cuda', type=str, default='0', metavar='N')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args = parser.parse_args()
    log_dir = 'logs/dec-' + datasetname
    if os.path.exists(log_dir)==False:
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    train_path = '/DATACENTER1/xiao.peng/DCN_keras-master/dataset/RCV1/Processed/data-0.pkl'
    test_path =  '/DATACENTER1/xiao.peng/DCN_keras-master/dataset/RCV1/Processed/data-0.pkl'
    # log_dir = 'logs/dec-'+datasetname

    # all_path='./dataset/wine/wine.data'
    for i in range(1, repeat+1):
        sdae_savepath = ("model/sdae-run-"+datasetname+"-%d.pt" % i)
        if os.path.exists("model/sdae-run-"+datasetname+"-%d.pt" % i)==False:
            print("Experiment #%d" % i)
            write_log("Experiment #%d" % i,log_dir)

            train_data=myDataset(train_path,-1, '.pkl')
            # test_data=myDataset(test_path,-1, '.pkl')
            train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                           collate_fn=train_data.collate_fn,num_workers=4)
            # test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True,
            #                                collate_fn=train_data.collate_fn,num_workers=4)


            # pretrain
            sdae = StackedDAE(input_dim=3000, z_dim=50, binary=False,
                encodeLayer=[2000,1000,1000], decodeLayer=[1000,1000,2000], activation="relu",
                dropout=0,log_dir=log_dir)
            sdae.cuda()
            # print(sdae)
            sdae.pretrain(train_loader, lr=args.sdae_lr, batch_size=batch_size,
                num_epochs=50, corrupt=0.2, loss_type="mse")
            torch.cuda.empty_cache()
            sdae.fit(train_loader, lr=args.sdae_lr, num_epochs=50, corrupt=0.2, loss_type="mse")
            sdae.save_model(sdae_savepath)
            del sdae
            del train_loader
            # del test_loader
            del train_data
            # del test_data
            torch.cuda.empty_cache()
        if os.path.exists("model/dec-run-"+datasetname+"-%d.pt" % i)==False:
            # finetune

            fit_train=myDataset(train_path,-1, '.pkl')
            # fit_test = myDataset(test_path,-1)

            # X = fit_train.dataSource[:,:-1]
            # y = fit_train.dataSource[:,-1]
            # X,y=torch.from_numpy(np.array(X,dtype=np.float32)),torch.from_numpy(np.array(y,dtype=np.float32))
            # y=y.long()
            train_loader = data.DataLoader(dataset=fit_train, batch_size=batch_size, shuffle=True,
                                           collate_fn=fit_train.collate_fn, num_workers=4)
            dec = DEC(input_dim=3000, z_dim=50, n_clusters=4,
                encodeLayer=[2000,1000,1000], activation="relu", dropout=0, writer=writer,log_dir=log_dir)
            # print(dec)
            dec.load_model(sdae_savepath)
            dec.cuda()
            dec.fit(dataloader=train_loader, lr=args.dec_lr, batch_size=batch_size, num_epochs=10,
                update_interval=5)
            dec_savepath = ("model/dec-run-"+datasetname+"-%d.pt" % i)
            # del X
            # del y
            del dec
            torch.cuda.empty_cache()
            # dec.save_model(dec_savepath)