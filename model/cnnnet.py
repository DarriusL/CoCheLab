# @Time   : 2023.03.03
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch
from lib import glb_var

class VerticalConv(torch.nn.Module):
    ''' Vertical Convolutional Network

    Parameters:
    -----------

    channal:int

    height:int

    width:int

    num_kernels:int
        the numbers of the kernels

    Methods:
    --------

    forward(self, x):
        argin:
            -x:(batch, channal, height, width)
        argout:
            -output:(batch, channal, num_kernels, 1)
    '''
    def __init__(self, channal, height, width, num_kernels) -> None:
        super().__init__();
        self.height =height;
        self.num_kernels = num_kernels;
        #calculate repeat times
        rpt = num_kernels//height;
        endptr = num_kernels%height;
        h = list(range(1, height + 1));
        layers = [torch.nn.Conv2d(channal, channal, (h[j], width)) for _ in range(rpt) for j in range(height)];
        for i in range(endptr):
            layers.append(torch.nn.Conv2d(channal, channal, (h[i], width)));
        self.convlayers = torch.nn.ModuleList(layers);
        #create maxpool layers
        h = list(map(lambda x: height-x+1, h));
        self.maxpool_layers = torch.nn.ModuleList([torch.nn.MaxPool2d((h[j], 1), stride = 1) for j in range(height)]);
        #activation function
        self.relu = torch.nn.ReLU();
    
    def forward(self, x):
        #x:(batch, channal, height, width)
        batch_size, channal = x.shape[0], x.shape[1];
        #output:(batch, channal, num_kernels, 1)
        output = torch.zeros((batch_size, channal, self.num_kernels, 1)).to(glb_var.get_value('device'));
        for idx, layer in enumerate(self.convlayers):
            ptr = idx%self.height;
            output[:, :, idx, :] = self.relu(self.maxpool_layers[ptr](layer(x)))[:, :, 0, :];
        return output;

class HorizontalConv(torch.nn.Module):
    def __init__(self, channal, height, width, num_kernels) -> None:
        super().__init__();
        self.width =width;
        self.num_kernels = num_kernels;
        #calculate repeat times
        rpt = num_kernels//width;
        endptr = num_kernels%width;
        w = list(range(1, width + 1));
        layers = [torch.nn.Conv2d(channal, channal, (height, w[j])) for _ in range(rpt) for j in range(width)];
        for i in range(endptr):
            layers.append(torch.nn.Conv2d(channal, channal, (height, w[i])));
        self.convlayers = torch.nn.ModuleList(layers);
        #create maxpool layers
        w = list(map(lambda x: width-x+1, w));
        if rpt > 0:
            self.maxpool_layers = torch.nn.ModuleList([torch.nn.MaxPool2d((1, w[j]), stride = 1) for j in range(width)]);
        else:
            self.maxpool_layers = torch.nn.ModuleList([torch.nn.MaxPool2d((1, w[j]), stride = 1) for j in range(endptr)]);
        #activation function
        self.relu = torch.nn.ReLU();

    def forward(self, x):
        #x:(batch, channal, height, width)
        batch_size, channal = x.shape[0], x.shape[1];
        #output:(batch, channal, 1, num_kernels)
        output = torch.zeros((batch_size, channal, 1, self.num_kernels)).to(glb_var.get_value('device'));
        for idx, layer in enumerate(self.convlayers):
            ptr = idx%self.width;
            output[:, :, :, idx] = self.relu(self.maxpool_layers[ptr](layer(x)))[:, :, :, 0];
        return output;
