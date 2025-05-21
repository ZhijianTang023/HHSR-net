import torch
from torch.nn import functional as F

import numpy as np
import glob
from tqdm import tqdm
import os

from hhsrnet_arch import HHSRNet

def get_kernel(kernel_size, k):
    sigma = 1
    sigma_3 = k * sigma
    X_ = np.linspace(-sigma_3, sigma_3, kernel_size)
    Y_ = np.linspace(-sigma_3, sigma_3, kernel_size)
    x_, y_ = np.meshgrid(X_, Y_)
    gauss_1 = 1 / (2 * np.pi * sigma ** 2) * np.exp(- (x_ ** 2 + y_ ** 2) / (2 * sigma ** 2))
    Z = gauss_1.sum()
    gauss_2 = (1/Z)*gauss_1
    
    return gauss_2
    

if __name__ == '__main__':
    print(__file__)
    data_path = glob.glob(__file__ + os.sep + 'output_demo' + os.sep + 'input' + os.sep + '*.npy')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = __file__ + os.sep + 'model_save' + os.sep + 'HHSR_state_dict.pth'
    result_path = __file__ + os.sep + 'output_demo' + os.sep + 'output'
    
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        
    k=16
    kernel_size = 41
    kernel_road = np.expand_dims(0.07*get_kernel(kernel_size, 2*kernel_size/k), axis=0)
    kernel_road = torch.cuda.FloatTensor(kernel_road).unsqueeze(0)
    kernel_POI = np.expand_dims(24*get_kernel(kernel_size, kernel_size/k), axis=0)
    kernel_POI = torch.cuda.FloatTensor(kernel_POI).unsqueeze(0)

    model = HHSRNet(num_in_ch=7,
                 num_out_ch=1)
    model = model.to(device)
    
    saved_state_dict = torch.load(model_path)
    model.load_state_dict(saved_state_dict)
    model.eval()
    
    min_ = [-8.22675475e-01, 9.09467635e-05, 8.51292459e-09, -6.97696829e+00, -8.57879066e+00, 2.72961121e+02, 2.65513802e+01]
    max_ = [32.33490529, 7.21356487, 3.1894002, 8.69364929, 6.26419163, 308.44396973, 95.88899231]
    
    data_input = np.load(data_path[0])
    data_output = np.zeros((data_input.shape[1],data_input.shape[2]))
    for data_day_pathi in tqdm(range(len(data_path))):
        data_day_path = data_path[data_day_pathi]
        data_input = np.load(data_day_path)
        
        POI = np.expand_dims( data_input[1,:,:], 0 )
        POI_torch = torch.from_numpy(POI).float().to(device)
        POI_torch = F.conv2d(POI_torch, kernel_POI, stride=1, padding=20)
        POI = POI_torch.cpu().detach().numpy()
        POI = np.squeeze(POI)
        data_input[1,:,:] = np.expand_dims( POI, 0 )

        road = np.expand_dims( data_input[2,:,:], 0 )
        road_torch = torch.from_numpy(road).float().to(device)
        road_torch = F.conv2d(road_torch, kernel_road, stride=1, padding=20)
        road = road_torch.cpu().detach().numpy()
        road = np.squeeze(road)
        data_input[2,:,:] = np.expand_dims( road, 0 )
            
            
        imgname = os.path.splitext(os.path.basename(data_day_path))[0]
        norm_data = np.zeros((data_input.shape))
        for i in range(norm_data.shape[0]):
            norm_data[i, :, :] = (data_input[i, :, :] - min_[i]) / (max_[i] - min_[i] + 1e-9)
        norm_data = np.expand_dims(norm_data,0)
        norm_data = torch.from_numpy(norm_data).float().to(device)
        for i in tqdm( range(40, data_output.shape[0]-40) ):
            for j in range(40, data_output.shape[1]-40):
                output = model(norm_data[:,:,i-40:i+41,j-40:j+41])
                output = np.squeeze(output.cpu().detach().numpy())
                data_output[i,j] = output[40,40]
        data_output = ( data_output * (max_[0] - min_[0] + 1e-9) + min_[0] ) / 0.294924707
        np.save(result_path + imgname + '.npy', data_output)
