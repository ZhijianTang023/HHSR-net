import torch
from torch import nn as nn

class HHSRNet(nn.Module):
    """
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
    """
    def __init__(self, num_in_ch, num_out_ch):
        super(HHSRNet, self).__init__()
        self.conv_no2 = nn.Conv2d(1, 1, 3, 1, 1)
        self.conv_POI_1 = nn.Conv2d(5, 8, 5, 1, 2)
        self.conv_POI_2 = nn.Conv2d(8, 1, 5, 1, 2)
        self.conv_road = nn.Conv2d(5, 1, 5, 1, 2)
        self.conv_hr = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv_last = nn.Conv2d(8, num_out_ch, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        no2 = x[:,0:1,:,:]
        POI = x[:,1:2,:,:]
        road = x[:,2:3,:,:]
        wrf = x[:,3:7,:,:]
        no2 = self.conv_no2(no2)
        
        POI = self.conv_POI_1(torch.cat((POI,wrf), dim=1))
        POI = self.conv_POI_2(POI)
        road = self.conv_road(torch.cat((road,wrf), dim=1))
        no2 = torch.cat((no2,POI), dim=1)
        no2 = torch.cat((no2,road), dim=1)
        
        out = self.lrelu(self.conv_hr(no2))
        out = self.conv_last(out)
        
        return out
