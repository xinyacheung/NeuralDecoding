import torch.nn as nn
import torch.nn.functional as F

class inverse_SpikeDecoder(nn.Module):
    
    def __init__(self, ncell):
        
        super(inverse_SpikeDecoder, self).__init__()
        
        self.dense1 = nn.Linear(512,ncell)
        self.bn1 = nn.BatchNorm2d(45)
        self.dense2 = nn.Linear(1024*3,512)
        self.reshape = nn.Flatten(start_dim=2)

    def forward(self, x):
        
        x = x.permute(0,2,1,3,4)
        x = self.reshape(x)
        x = self.dense2(x)
        x = x.unsqueeze(-1)
        x = self.bn1(x)
        x = x.squeeze(-1)
        x = F.relu(x)
        x = self.dense1(x)
        
        return x
    
class inverse_VideoUNet(nn.Module):
    
    def __init__(self, input_shape, ncell,hiddensize=64):
        
        super(inverse_VideoUNet, self).__init__()
        
        self.reverse_Dense = inverse_SpikeDecoder(ncell)
        
        self.encoder = nn.Sequential(
            nn.Conv3d(input_shape[0], hiddensize, kernel_size=(7, 7, 7),padding='same'),
            nn.BatchNorm3d(hiddensize),
            nn.ReLU(),  
            nn.MaxPool3d([3,2,2]),
            nn.Conv3d(hiddensize, hiddensize*2, kernel_size=(5,5, 5), padding='same'),
            nn.BatchNorm3d(hiddensize*2),
            nn.ReLU(),
            nn.MaxPool3d([3,2,2]),
            nn.Conv3d(hiddensize*2, hiddensize*4, kernel_size=(3, 3, 3),  padding='same'),
            nn.BatchNorm3d(hiddensize*4),
            nn.ReLU(),
            nn.MaxPool3d([5,2,2]),
            nn.Conv3d(hiddensize*4, hiddensize*4, kernel_size=(3, 3, 3),  padding='same'),
            nn.BatchNorm3d(hiddensize*4),
            nn.ReLU(),
            nn.MaxPool3d([1,2,2]),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=tuple([1,2,2]), mode='nearest'),
            nn.Conv3d(hiddensize*4, hiddensize*4, kernel_size=(3, 3, 3), padding='same'),
            nn.BatchNorm3d(hiddensize*4),
            nn.ReLU(),
            nn.Upsample(scale_factor=tuple([5,2,2]), mode='nearest'),
            nn.Conv3d(hiddensize*4,hiddensize*2, kernel_size=(3, 3, 3), padding='same'),
            nn.BatchNorm3d(hiddensize*2),
            nn.ReLU(),
            nn.Upsample(scale_factor=tuple([3,2,2]), mode='nearest'),
            nn.Conv3d(hiddensize*2, hiddensize, kernel_size=(5,5,5), padding='same'),
            nn.BatchNorm3d(hiddensize),
            nn.ReLU(),
            nn.Upsample(scale_factor=tuple([3,2,2]), mode='nearest'),
            nn.Conv3d(hiddensize, 3, kernel_size=(7, 7, 7), padding='same'),
            nn.BatchNorm3d(3)
        )
        
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.reverse_Dense(x)
        
        return x
