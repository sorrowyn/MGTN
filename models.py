from torch.nn import Parameter
from util import *
from gtn import *
import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MGTNResnet(nn.Module):
    def __init__(self, model_name, num_classes, in_channel=300, t1=0.0, t2=0.0, adj_file=None, mod_file=None, ml_threshold=0.999):
        super(MGTNResnet, self).__init__()

        _mods = np.loadtxt(mod_file, dtype=int)
        
        self.backbones = []
        self.poolings = []
        # Create multiple backbones
        for i in range(int(max(_mods)) - int(min(_mods)) + 1):
            model = load_model(model_name)
            backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            )
            self.add_module('backbone' + str(i+1), backbone)
            self.backbones.append(backbone)
            pooling = nn.MaxPool2d(14, 14)
            self.add_module('pooling' + str(i+1), pooling)
            self.poolings.append(pooling)

        self.num_classes = num_classes

        # Graph Convolutions
        self.gc1 = GraphConvolution(in_channel, 2048)
        self.gc2 = GraphConvolution(2048, 4096)
        self.relu = nn.LeakyReLU(0.2)

        # Topology
        A_Tensor = torch.eye(num_classes).type(torch.FloatTensor).unsqueeze(-1)
        s_adj = gen_A(num_classes, 1.0, t1, adj_file)
        s_adj = torch.from_numpy(s_adj).type(torch.FloatTensor)
        A_Tensor = torch.cat([A_Tensor,s_adj.unsqueeze(-1)], dim=-1)
        w_adj = gen_A(num_classes, t1, t2, adj_file)
        w_adj = torch.from_numpy(w_adj).type(torch.FloatTensor)
        A_Tensor = torch.cat([A_Tensor,w_adj.unsqueeze(-1)], dim=-1)

        self.gtn = GTLayer(A_Tensor.shape[-1], 1, first=True)
        self.A = A_Tensor.unsqueeze(0).permute(0,3,1,2) 

        self.mods = Parameter(torch.from_numpy(gen_M(_mods, dims=2048, t=ml_threshold)).float())

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        fs = []

        for i in range(len(self.backbones)):
            f = self.backbones[i](feature)
            f = self.poolings[i](f)
            f = f.view(f.size(0), -1)
            fs.append(f)

        feature = torch.cat(fs, 1)

        inp = inp[0]
        
        adj, _ = self.gtn.forward(self.A)
        adj = torch.squeeze(adj, 0) + torch.eye(self.num_classes).type(torch.FloatTensor).cuda()
        adj = gen_adj(adj)

        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        x = torch.mul(x, self.mods)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def get_config_optim(self, lr, lrp):
        config_optim = []
        for backbone in self.backbones:
            config_optim.append({'params': backbone.parameters(), 'lr': lr * lrp})
        config_optim.append({'params': self.gc1.parameters(), 'lr': lr})
        config_optim.append({'params': self.gc2.parameters(), 'lr': lr})
        return config_optim



def mgtn_resnet(num_classes, t1, t2, pretrained=True, adj_file=None, mod_file=None, in_channel=300, ml_threshold=0.999):
    return MGTNResnet('resnext50_32x4d_swsl', num_classes, t1=t1, t2=t2, adj_file=adj_file, mod_file=mod_file, in_channel=in_channel, ml_threshold=ml_threshold)
