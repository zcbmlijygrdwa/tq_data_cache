import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math

class NetBlock(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(NetBlock, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.layer_norm(x)

        return x

class Net(nn.Module):

    def __init__(self, input_dim):
        super(Net, self).__init__()
        #self.mid_width = 128
        #self.mid_width = 256
        #self.mid_width = 512
        self.mid_width = input_dim
        self.output = 3


        self.nb1 = NetBlock(input_dim, self.mid_width)
        #self.nb2 = NetBlock(self.mid_width, self.mid_width)
        #self.nb3 = NetBlock(self.mid_width, self.mid_width)
        #self.nb4 = NetBlock(self.mid_width, self.mid_width)
        #self.nb5 = NetBlock(self.mid_width, self.mid_width)
        #self.nb6 = NetBlock(self.mid_width, self.mid_width)
        #self.nb7 = NetBlock(self.mid_width, self.mid_width)
        #self.nb7 = NetBlock(self.mid_width, self.mid_width)
        #self.nb8 = NetBlock(self.mid_width, self.mid_width)
        #self.nb9 = NetBlock(self.mid_width, self.mid_width)
        #self.nb10 = NetBlock(self.mid_width, self.mid_width)
        #self.nb11 = NetBlock(self.mid_width, self.mid_width)
        #self.nb12 = NetBlock(self.mid_width, self.mid_width)
        #self.nb13 = NetBlock(self.mid_width, self.mid_width)
        #self.nb14 = NetBlock(self.mid_width, self.mid_width)
        #self.nb15 = NetBlock(self.mid_width, self.mid_width)

        self.fc_final = nn.Linear(self.mid_width, self.output)
        #self.fc_final = nn.Linear(input_dim, self.output)

    def forward(self, x):

        #x = self.nb1(x)
        #res = x
        #x = self.nb2(x)
        #x += res
        #res = x
        #x = self.nb3(x)
        #x += res
        #res = x
        #x = self.nb4(x)
        #x += res
        #res = x
        #x = self.nb5(x)
        #x += res
        #res = x
        #x = self.nb6(x)
        #x += res
        #res = x
        #x = self.nb7(x)
        #x += res
        #res = x
        #x = self.nb8(x)
        #x += res
        #res = x
        #x = self.nb9(x)
        #x += res
        #res = x
        #x = self.nb10(x)
        #x += res
        #res = x
        #x = self.nb11(x)
        #x += res
        #res = x
        #x = self.nb12(x)
        #x += res
        #res = x
        #x = self.nb13(x)
        #x += res
        #res = x
        #x = self.nb14(x)
        #x += res
        #res = x
        #x = self.nb15(x)
        #x += res
        x = self.fc_final(x)

        return x

class MyNet(nn.Module):

    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        #self.mid_width = 128
        #self.mid_width = 256
        self.net = Net(input_dim)


    def forward(self, x):

        x = self.net(x)

        return x

