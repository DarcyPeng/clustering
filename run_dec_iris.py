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

import warnings
warnings.filterwarnings('ignore')

# dataset = "mnist"
# datasetname="cifar"
datasetname="iris"
repeat = 1
batch_size = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--sdae_lr', type=float, default=0.1, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--dec_lr', type=float, default=0.0001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    args = parser.parse_args()
    if os.path.exists('logs/'+datasetname)==False:
        os.makedirs('logs/'+datasetname)
    log_dir = 'logs/' + datasetname
    train_path = './dataset/iris/iris_train.data'
    test_path = './dataset/iris/iris_test.data'
    for i in range(1, repeat+1):
        sdae_savepath = ("model/sdae-run-iris-%d.pt" % i)
        if os.path.exists("model/sdae-run-iris-%d.pt" % i)==False:
            print("Experiment #%d" % i)
            write_log("Experiment #%d" % i,log_dir)

            train_data=myDataset(train_path,-1)
            # test_data=myDataset(test_path,-1)
            train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                           collate_fn=train_data.collate_fn)
            # test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True,
            #                                collate_fn=train_data.collate_fn)


            # pretrain
            sdae = StackedDAE(input_dim=4, z_dim=2, binary=False,
                encodeLayer=[8], decodeLayer=[8], activation="relu",
                dropout=0,log_dir=log_dir)
            # print(sdae)
            sdae.pretrain(train_loader,  lr=args.sdae_lr, batch_size=batch_size,
                num_epochs=5, corrupt=0.2, loss_type="mse")
            sdae.fit(train_loader, lr=args.sdae_lr, num_epochs=5, corrupt=0.2, loss_type="mse")
            sdae.save_model(sdae_savepath)
        if os.path.exists("model/dec-run-iris-%d.pt" % i)==False:
        # if True:
            # finetune
            fit_train=myDataset(train_path,-1)
            # fit_test = myDataset(test_path,-1)

            # X = fit_train.dataSource[:,:-1]
            # y = fit_train.dataSource[:,-1]
            # X,y=torch.from_numpy(np.array(X,dtype=np.float32)),torch.from_numpy(np.array(y,dtype=np.float32))
            # y=y.long()
            train_loader = data.DataLoader(dataset=fit_train, batch_size=batch_size, shuffle=True,
                                           collate_fn=fit_train.collate_fn, num_workers=4)
            dec = DEC(input_dim=4, z_dim=2, n_clusters=3,
                encodeLayer=[8], activation="relu", dropout=0,log_dir=log_dir)
            print(dec)
            dec.load_model(sdae_savepath)
            dec.fit(dataloader=train_loader, lr=args.dec_lr, batch_size=batch_size, num_epochs=20,
                update_interval=1)
            # dec_savepath = ("model/dec-run-iris-%d.pt" % i)
            # dec.save_model(dec_savepath)