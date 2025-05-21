import torch
from torch import nn as nn
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
from audtorch.metrics.functional import pearsonr

import numpy as np
import glob
from tqdm import tqdm
import random
import os
import sys

from hhsrnet_arch import HHSRNet

# define loss
class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()

    def forward(self, pre, label):
        loss = 1-pearsonr(pre, label)
        return loss

def dropout_loss(loss1,loss2):
    dropout_p = random.random()
    if dropout_p<=0.1:
        return loss1
    if dropout_p>0.1:
        return loss2
        
#create SubClass
class subDataset(Dataset.Dataset):
    #initialize, define data content and labels
    def __init__(self, data_path, label_path, min_, max_):
        self.data_paths = data_path
        self.label_path = label_path
        self.min_ = min_
        self.max_ = max_
		
    #Return the size of the dataset
    def __len__(self):
        return len(self.data_paths)#, len(self.data_paths)
    #get data content and labels
    def __getitem__(self, index):
        data_path_now = self.data_paths[index]
        
        imgname = os.path.splitext(os.path.basename(data_path_now))[0]
        
        label_path_now = self.label_path + imgname + '.npy'
        try:
            data = np.load(data_path_now)
            for i in range(data.shape[0]):
                data[i, :, :] = (data[i, :, :] - self.min_[i]) / (self.max_[i] - self.min_[i] + 1e-9)
            
            label = np.load(label_path_now)
            label = label*0.294924707
            label = (label - self.min_[0]) / (self.max_[0] - self.min_[0] + 1e-9)
            norm_data = data.astype(np.float32)
            norm_label = label.astype(np.float32)
            
            return norm_data, norm_label
        except:
            print(data_path_now + ' error!!!')

if __name__ == '__main__':
    
    model_save_path = __file__ + os.sep + 'model_save' + os.sep
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    data_path = glob.glob( __file__ + os.sep + 'data_for_train' + os.sep + 'input' + os.sep + '*.npy')
    label_path = __file__ + os.sep + 'data_for_train' + os.sep + 'label' + os.sep

    min_ = [-8.22675475e-01, 9.09467635e-05, 8.51292459e-09, -6.97696829e+00, -8.57879066e+00, 2.72961121e+02, 2.65513802e+01]
    max_ = [32.33490529, 7.21356487, 3.1894002, 8.69364929, 6.26419163, 308.44396973, 95.88899231]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('data loading.....')
    indexes = np.arange(len(data_path))
    np.random.shuffle(indexes)
    dataset = subDataset(np.array(data_path)[indexes].tolist(), label_path, min_, max_)
    
    indexes = np.arange(len(dataset))
    
    train_index = indexes[: int( len(indexes)*0.7 )]
    val_index = indexes[int( len(indexes)*0.7 ):int( len(indexes)*0.9 )]
    test_index = indexes[int( len(indexes)*0.9 ):]
    # training set:validation set:testing set=7：2：1
    
    # input data
    train_dataset = np.array(dataset, dtype=object)[train_index][:,0]
    val_dataset = np.array(dataset, dtype=object)[val_index][:,0]
    test_dataset = np.array(dataset, dtype=object)[test_index][:,0]
    # label data
    train_labelset = np.array(dataset, dtype=object)[train_index][:,1]#np.array(dataset)[train_index][:,1]
    val_labelset = np.array(dataset, dtype=object)[val_index][:,1]
    test_labelset = np.array(dataset, dtype=object)[test_index][:,1]
    
    #dataloader
    g=torch.Generator()
    g.manual_seed(0)
    train_dataloader = DataLoader.DataLoader(train_dataset, batch_size= 100000, shuffle = False, num_workers= 4, generator=g)
    val_dataloader = DataLoader.DataLoader(val_dataset, batch_size= 100000, shuffle = False, num_workers= 4, generator=g)
    test_dataloader = DataLoader.DataLoader(test_dataset, batch_size= 100000, shuffle = False, num_workers= 4, generator=g)
    
    train_labelloader = DataLoader.DataLoader(train_labelset, batch_size= 100000, shuffle = False, num_workers= 4, generator=g)
    val_labelloader = DataLoader.DataLoader(val_labelset, batch_size= 100000, shuffle = False, num_workers= 4, generator=g)
    test_labelloader = DataLoader.DataLoader(test_labelset, batch_size= 100000, shuffle = False, num_workers= 4, generator=g)
    
    all_dataset = np.array(dataset, dtype=object)[indexes][:,0]
    all_labelset = np.array(dataset, dtype=object)[indexes][:,1]
    all_dataloader = DataLoader.DataLoader(all_dataset, batch_size= 100000, shuffle = False, num_workers= 4, generator=g)
    all_labelloader = DataLoader.DataLoader(all_labelset, batch_size= 100000, shuffle = False, num_workers= 4, generator=g)
    
    #loading model
    model = HHSRNet(num_in_ch=7,
                 num_out_ch=1)
    model = model.to(device)
    
    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
    loss_fn_pear = myLoss()
    loss_fn_MSE = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4, eps=1e-5 )
    loss_pear_com = 10
    print('start training.....')
    for epoch in tqdm(range(1000)):
        for i, item in enumerate(train_dataloader):
            for j, label in enumerate(train_labelloader):
                if i == j:
                    input = item
                    label = label
                    input = input.to(device)
                    label = label.to(device)
                    output = model(input)
                    loss_pear = loss_fn_pear(output[:,0,40,40], label)
                    loss_MSE = loss_fn_MSE(output[:,0,40,40], label)
                    loss = dropout_loss(loss_pear, loss_MSE)
                    loss_train = loss_pear
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        
        with torch.no_grad():
            for i, item in enumerate(val_dataloader):
                for j, label in enumerate(val_labelloader):
                    if i == j:
                        input = item
                        label = label
                        input = input.to(device)
                        label = label.to(device)
                        output = model(input)
                        loss_pear = loss_fn_pear(output[:,0,40,40], label)
                        loss_val = loss_pear

        with torch.no_grad():
            for i, item in enumerate(test_dataloader):
                for j, label in enumerate(test_labelloader):
                    if i == j:
                        input = item
                        label = label
                        input = input.to(device)
                        label = label.to(device)
                        output = model(input)
                        loss_pear = loss_fn_pear(output[:,0,40,40], label)
                        loss_test = loss_pear
        
        if loss_pear_com>loss_val.cpu().detach().numpy()[0]:
            loss_pear_com = loss_val.cpu().detach().numpy()[0]
            torch.save(model, model_save_path + 'HHSR_new.pth')
            torch.save(model.state_dict(), model_save_path + 'HHSR_state_dict_new.pth')
            with torch.no_grad():
                for i, item in enumerate(all_dataloader):
                    for j, label in enumerate(all_labelloader):
                        if i == j:
                            input = item
                            label = label
                            input = input.to(device)
                            label = label.to(device)
                            output = model(input)
                            loss_all = loss_fn_pear(output[:,0,40,40], label)
            print('epoch ' + str(epoch) + ', best correlation coefficient: ' + str(1-loss_all.cpu().detach().numpy()[0]))
    