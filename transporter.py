import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet

class TwoStreamFCN(nn.Module):
    def __init__(self):
        super(TwoStreamFCN, self).__init__()

class PickModel(nn.Module):
    def __init__(self):
        super(PickModel, self).__init__()
        self.resnet = ResNet(out_channels=1)
    
    def forward(self, x):
        x = self.resnet(x)
        return nn.Softmax2d()(x)

class PlaceModel(nn.Module):
    def __init__(self, dense_features=512, crop_size=20):
        super(PlaceModel, self).__init__()
        self.resnet = ResNet(out_channels=dense_features)
    
    def forward(self, x):
        x = self.resnet(x)
        return nn.Softmax2d()(x)

if __name__ == '__main__':
    model = PickModel()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)  # Should be [1, 1, 224, 224]