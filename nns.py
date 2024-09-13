import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2D(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=False, activation='tanh', dropout=0, bn=True, t=False):
        super(Conv2D, self).__init__()
        if t:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        else:
            self.bn = None
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation is None:
            self.activation = None
        else:
            raise ValueError("Invalid param 'activation' provided: {}".format(activation))
        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
    
class Linear(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True, activation='tanh', dropout=0, bn=True):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        if bn:
            self.bn = nn.BatchNorm1d(out_features, eps=1e-5, momentum=0.1)
        else:
            self.bn = None
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation is None:
            self.activation = None
        else:
            raise ValueError("Invalid param 'activation' provided: {}".format(activation))
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, x):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
    
class NoteRecog(nn.Module): 
    def __init__(self):
        super(NoteRecog, self).__init__() # in: (1x) 128 x 1280
        
        # p1280: 1 x 128 x 1280
        self.p1280_convs = nn.Sequential(
            Conv2D(1, 4, (7, 5), padding=(3, 2)), # 4 x 128 x 1280
        )
        self.p1280_step = nn.MaxPool2d((4, 2), stride=(4, 2)) # 4 x 32 x 640
        self.p1280_skip = nn.Sequential(
            Conv2D(4, 2, 1), # 2 x 32 x 640
            nn.MaxPool2d((128, 16), stride=(128, 16)), # 2 x 1 x 80
        )
        
        # p640: 4 x 32 x 640
        self.p640_convs = nn.Sequential(
            Conv2D(4, 8, 5, padding=2, dropout=0.005), # 8 x 32 x 640
        )
        self.p640_step = nn.MaxPool2d(2, stride=2) # 8 x 16 x 320
        self.p640_skip = nn.Sequential(
            Conv2D(8, 4, 1), # 4 x 16 x 320
            nn.MaxPool2d((32, 8), stride=(32, 8)), # 4 x 1 x 80
        )
        
        # p320: 8 x 12 x 320
        self.p320_convs = nn.Sequential(
            Conv2D(8, 14, 5, padding=2, dropout=0.01), # 14 x 16 x 320
        )
        self.p320_step = nn.MaxPool2d(2, stride=2) # 14 x 8 x 160
        self.p320_skip = nn.Sequential(
            Conv2D(14, 6, 1), # 3 x 8 x 160
            nn.MaxPool2d((16, 4), stride=(16, 4)), # 4 x 1 x 80
        )
            
        # p160: 12 x 8 x 160
        self.p160_convs = nn.Sequential(
            Conv2D(14, 24, 5, padding=2, dropout=0.015), # 18 x 8 x 160
        )
        self.p160_step = nn.MaxPool2d(2, stride=2) # 18 x 4 x 80
        
        # p80: 18 x 4 x 80
        self.to_20 = nn.Sequential(
            Conv2D(24, 64, (2, 4), stride=(1, 4), dropout=0.01), # 50 x 3 x 20
        )
        self.skip_20 = nn.Sequential(
            Conv2D(64, 8, 1), # 50 x 3 x 20
        )
        self.to_5 = nn.Sequential(
            Conv2D(64, 128, (2, 4), stride=(1, 4), dropout=0.01), # 128 x 2 x 5
        )
        self.skip_5 = nn.Sequential(
            Conv2D(128, 24, 1), # 50 x 3 x 20
        )
        self.to_1 = nn.Sequential(
            Conv2D(128, 400, (2, 5), stride=(1, 5), dropout=0.01), # 400 x 1 x 1
        )
        self.from_1 = nn.Sequential(
            Conv2D(400, 64, (2, 5), stride=(1, 5), dropout=0.01, t=True), # 64 x 2 x 5
        )
        self.from_5 = nn.Sequential(
            Conv2D(64+24, 16, (2, 4), stride=(1, 4), dropout=0.01, t=True), # 16 x 3 x 20
        )
        self.from_20 = nn.Sequential(
            Conv2D(16+8, 6, (2, 4), stride=(1, 4), dropout=0.01, t=True), # 6 x 4 x 80
        )
        
        # p80: (18+4) x 4 x 80
        self.squeeze = nn.Sequential(
            Conv2D(24+6, 68, (4, 1), stride=(4, 1)), # 24 x 1 x 80
        )
        
        # p80: 50 x 1 x 80 (50 = 44 + 1 + 2 + 3)
        self.fc_q = nn.Sequential( # 400 -> 1
            Linear(80, 40, dropout=0.012, bn=False),
            Linear(40, 20, dropout=0.006, bn=False),
            Linear(20, 10, bn=False),
            Linear(10, 5, bn=False),
            Linear(5, 1, bn=False, activation=None),
        )
        self.fc_l = nn.Sequential( # 400 -> 1
            Linear(80, 40, dropout=0.012, bn=False),
            Linear(40, 20, dropout=0.006, bn=False),
            Linear(20, 10, bn=False),
            Linear(10, 5, bn=False),
            Linear(5, 1, bn=False, activation=None),
        )
        self.fc_r = nn.Sequential( # 400 -> 1
            Linear(80, 40, dropout=0.012, bn=False),
            Linear(40, 20, dropout=0.006, bn=False),
            Linear(20, 10, bn=False),
            Linear(10, 5, bn=False),
            Linear(5, 1, bn=False, activation=None),
        )
        self.fc_c = nn.Sequential( # 400 -> 1
            Linear(80, 40, dropout=0.012, bn=False),
            Linear(40, 20, dropout=0.006, bn=False),
            Linear(20, 10, bn=False),
            Linear(10, 5, bn=False),
            Linear(5, 1, bn=False, activation='sigmoid'),
        )
        self.fc_type = nn.Sequential( # 400 -> 9
            Linear(80, 55, dropout=0.012, bn=False),
            Linear(55, 38, dropout=0.006, bn=False),
            Linear(38, 24, bn=False),
            Linear(24, 16, bn=False),
            Linear(16, 9, bn=False),
        )
        self.fc_dot_t = nn.Sequential( # 400 -> 4
            Linear(80, 50, dropout=0.012, bn=False),
            Linear(50, 28, dropout=0.006, bn=False),
            Linear(28, 16, bn=False),
            Linear(16, 8, bn=False),
            Linear(8, 4, bn=False),
        )
        self.fc_dot_b = nn.Sequential( # 400 -> 4
            Linear(80, 50, dropout=0.012, bn=False),
            Linear(50, 28, dropout=0.006, bn=False),
            Linear(28, 16, bn=False),
            Linear(16, 8, bn=False),
            Linear(8, 4, bn=False),
        )
        self.fc_dot_r = nn.Sequential( # 400 -> 4
            Linear(80, 50, dropout=0.012, bn=False),
            Linear(50, 28, dropout=0.006, bn=False),
            Linear(28, 16, bn=False),
            Linear(16, 8, bn=False),
            Linear(8, 4, bn=False),
        )
        self.fc_uline = nn.Sequential( # 400 -> 4
            Linear(80, 50, dropout=0.012, bn=False),
            Linear(50, 28, dropout=0.006, bn=False),
            Linear(28, 16, bn=False),
            Linear(16, 8, bn=False),
            Linear(8, 4, bn=False),
        )
        
    def forward(self, x):
        x = x.unsqueeze(1) # 320 x 1280 -> 1 x 320 x 1280, watch out for batch!
        
        x = self.p1280_convs(x)
        p1280_skipped = self.p1280_skip(x)
        x = self.p1280_step(x)
        
        x = self.p640_convs(x)
        p640_skipped = self.p640_skip(x)
        x = self.p640_step(x)
        
        x = self.p320_convs(x)
        p320_skipped = self.p320_skip(x)
        x = self.p320_step(x)
        
        x = self.p160_convs(x)
        x = self.p160_step(x)
        
        xxx = self.to_20(x)
        x20_skipped = self.skip_20(xxx)
        xxx = self.to_5(xxx)
        x5_skipped = self.skip_5(xxx)
        xxx = self.to_1(xxx)
        xxx = self.from_1(xxx)
        xxx = self.from_5(torch.cat([xxx, x5_skipped], axis=1))
        xxx = self.from_20(torch.cat([xxx, x20_skipped], axis=1))
        x = self.squeeze(torch.cat([x, xxx], dim=1))
        
        x = torch.cat([x, p1280_skipped, p640_skipped, p320_skipped], axis=1).squeeze(-2).transpose(-1, -2)
        return {
            'q': self.fc_q(x).squeeze(-1),
            'l': self.fc_l(x).squeeze(-1),
            'r': self.fc_r(x).squeeze(-1),
            'c': self.fc_c(x).squeeze(-1),
            'type': self.fc_type(x),
            'dot_t': self.fc_dot_t(x),
            'dot_b': self.fc_dot_b(x),
            'dot_r': self.fc_dot_r(x),
            'uline': self.fc_uline(x),
        }
    
class NoteRecog_removed_encoder_decoder(nn.Module): 
    def __init__(self):
        super(NoteRecog_removed_encoder_decoder, self).__init__() # in: (1x) 128 x 1280
        
        # p1280: 1 x 128 x 1280
        self.p1280_convs = nn.Sequential(
            Conv2D(1, 4, (7, 5), padding=(3, 2)), # 4 x 128 x 1280
        )
        self.p1280_step = nn.MaxPool2d((4, 2), stride=(4, 2)) # 4 x 32 x 640
        self.p1280_skip = nn.Sequential(
            Conv2D(4, 2, 1), # 2 x 32 x 640
            nn.MaxPool2d((128, 16), stride=(128, 16)), # 2 x 1 x 80
        )
        
        # p640: 4 x 32 x 640
        self.p640_convs = nn.Sequential(
            Conv2D(4, 8, 5, padding=2, dropout=0.005), # 8 x 32 x 640
        )
        self.p640_step = nn.MaxPool2d(2, stride=2) # 8 x 16 x 320
        self.p640_skip = nn.Sequential(
            Conv2D(8, 4, 1), # 4 x 16 x 320
            nn.MaxPool2d((32, 8), stride=(32, 8)), # 4 x 1 x 80
        )
        
        # p320: 8 x 12 x 320
        self.p320_convs = nn.Sequential(
            Conv2D(8, 14, 5, padding=2, dropout=0.01), # 14 x 16 x 320
        )
        self.p320_step = nn.MaxPool2d(2, stride=2) # 14 x 8 x 160
        self.p320_skip = nn.Sequential(
            Conv2D(14, 6, 1), # 3 x 8 x 160
            nn.MaxPool2d((16, 4), stride=(16, 4)), # 4 x 1 x 80
        )
            
        # p160: 12 x 8 x 160
        self.p160_convs = nn.Sequential(
            Conv2D(14, 24, 5, padding=2, dropout=0.015), # 18 x 8 x 160
        )
        self.p160_step = nn.MaxPool2d(2, stride=2) # 18 x 4 x 80
        
        # p80: (18+4) x 4 x 80
        self.squeeze = nn.Sequential(
            Conv2D(24, 68, (4, 1), stride=(4, 1)), # 24 x 1 x 80
        )
        
        # p80: 50 x 1 x 80 (50 = 44 + 1 + 2 + 3)
        self.fc_q = nn.Sequential( # 400 -> 1
            Linear(80, 40, dropout=0.012, bn=False),
            Linear(40, 20, dropout=0.006, bn=False),
            Linear(20, 10, bn=False),
            Linear(10, 5, bn=False),
            Linear(5, 1, bn=False, activation=None),
        )
        self.fc_l = nn.Sequential( # 400 -> 1
            Linear(80, 40, dropout=0.012, bn=False),
            Linear(40, 20, dropout=0.006, bn=False),
            Linear(20, 10, bn=False),
            Linear(10, 5, bn=False),
            Linear(5, 1, bn=False, activation=None),
        )
        self.fc_r = nn.Sequential( # 400 -> 1
            Linear(80, 40, dropout=0.012, bn=False),
            Linear(40, 20, dropout=0.006, bn=False),
            Linear(20, 10, bn=False),
            Linear(10, 5, bn=False),
            Linear(5, 1, bn=False, activation=None),
        )
        self.fc_c = nn.Sequential( # 400 -> 1
            Linear(80, 40, dropout=0.012, bn=False),
            Linear(40, 20, dropout=0.006, bn=False),
            Linear(20, 10, bn=False),
            Linear(10, 5, bn=False),
            Linear(5, 1, bn=False, activation='sigmoid'),
        )
        self.fc_type = nn.Sequential( # 400 -> 9
            Linear(80, 55, dropout=0.012, bn=False),
            Linear(55, 38, dropout=0.006, bn=False),
            Linear(38, 24, bn=False),
            Linear(24, 16, bn=False),
            Linear(16, 9, bn=False),
        )
        self.fc_dot_t = nn.Sequential( # 400 -> 4
            Linear(80, 50, dropout=0.012, bn=False),
            Linear(50, 28, dropout=0.006, bn=False),
            Linear(28, 16, bn=False),
            Linear(16, 8, bn=False),
            Linear(8, 4, bn=False),
        )
        self.fc_dot_b = nn.Sequential( # 400 -> 4
            Linear(80, 50, dropout=0.012, bn=False),
            Linear(50, 28, dropout=0.006, bn=False),
            Linear(28, 16, bn=False),
            Linear(16, 8, bn=False),
            Linear(8, 4, bn=False),
        )
        self.fc_dot_r = nn.Sequential( # 400 -> 4
            Linear(80, 50, dropout=0.012, bn=False),
            Linear(50, 28, dropout=0.006, bn=False),
            Linear(28, 16, bn=False),
            Linear(16, 8, bn=False),
            Linear(8, 4, bn=False),
        )
        self.fc_uline = nn.Sequential( # 400 -> 4
            Linear(80, 50, dropout=0.012, bn=False),
            Linear(50, 28, dropout=0.006, bn=False),
            Linear(28, 16, bn=False),
            Linear(16, 8, bn=False),
            Linear(8, 4, bn=False),
        )
        
    def forward(self, x):
        x = x.unsqueeze(1) # 320 x 1280 -> 1 x 320 x 1280, watch out for batch!
        
        x = self.p1280_convs(x)
        p1280_skipped = self.p1280_skip(x)
        x = self.p1280_step(x)
        
        x = self.p640_convs(x)
        p640_skipped = self.p640_skip(x)
        x = self.p640_step(x)
        
        x = self.p320_convs(x)
        p320_skipped = self.p320_skip(x)
        x = self.p320_step(x)
        
        x = self.p160_convs(x)
        x = self.p160_step(x)
        
        x = self.squeeze(x)
        
        x = torch.cat([x, p1280_skipped, p640_skipped, p320_skipped], axis=1).squeeze(-2).transpose(-1, -2)
        return {
            'q': self.fc_q(x).squeeze(-1),
            'l': self.fc_l(x).squeeze(-1),
            'r': self.fc_r(x).squeeze(-1),
            'c': self.fc_c(x).squeeze(-1),
            'type': self.fc_type(x),
            'dot_t': self.fc_dot_t(x),
            'dot_b': self.fc_dot_b(x),
            'dot_r': self.fc_dot_r(x),
            'uline': self.fc_uline(x),
        }
    
class NoteRecog_removed_shortcut(nn.Module): 
    def __init__(self):
        super(NoteRecog_removed_shortcut, self).__init__() # in: (1x) 128 x 1280
        
        # p1280: 1 x 128 x 1280
        self.p1280_convs = nn.Sequential(
            Conv2D(1, 4, (7, 5), padding=(3, 2)), # 4 x 128 x 1280
        )
        self.p1280_step = nn.MaxPool2d((4, 2), stride=(4, 2)) # 4 x 32 x 640
        
        # p640: 4 x 32 x 640
        self.p640_convs = nn.Sequential(
            Conv2D(4, 8, 5, padding=2, dropout=0.005), # 8 x 32 x 640
        )
        self.p640_step = nn.MaxPool2d(2, stride=2) # 8 x 16 x 320
        
        # p320: 8 x 12 x 320
        self.p320_convs = nn.Sequential(
            Conv2D(8, 16, 5, padding=2, dropout=0.01), # 14 x 16 x 320
        )
        self.p320_step = nn.MaxPool2d(2, stride=2) # 14 x 8 x 160
            
        # p160: 12 x 8 x 160
        self.p160_convs = nn.Sequential(
            Conv2D(16, 28, 5, padding=2, dropout=0.015), # 18 x 8 x 160
        )
        self.p160_step = nn.MaxPool2d(2, stride=2) # 18 x 4 x 80
        
        # p80: 18 x 4 x 80
        self.to_20 = nn.Sequential(
            Conv2D(28, 64, (2, 4), stride=(1, 4), dropout=0.01), # 50 x 3 x 20
        )
        self.skip_20 = nn.Sequential(
            Conv2D(64, 10, 1), # 50 x 3 x 20
        )
        self.to_5 = nn.Sequential(
            Conv2D(64, 128, (2, 4), stride=(1, 4), dropout=0.01), # 128 x 2 x 5
        )
        self.skip_5 = nn.Sequential(
            Conv2D(128, 24, 1), # 50 x 3 x 20
        )
        self.to_1 = nn.Sequential(
            Conv2D(128, 400, (2, 5), stride=(1, 5), dropout=0.01), # 400 x 1 x 1
        )
        self.from_1 = nn.Sequential(
            Conv2D(400, 64, (2, 5), stride=(1, 5), dropout=0.01, t=True), # 64 x 2 x 5
        )
        self.from_5 = nn.Sequential(
            Conv2D(64+24, 16, (2, 4), stride=(1, 4), dropout=0.01, t=True), # 16 x 3 x 20
        )
        self.from_20 = nn.Sequential(
            Conv2D(16+10, 10, (2, 4), stride=(1, 4), dropout=0.01, t=True), # 6 x 4 x 80
        )
        
        # p80: (18+4) x 4 x 80
        self.squeeze = nn.Sequential(
            Conv2D(28+10, 80, (4, 1), stride=(4, 1)), # 24 x 1 x 80
        )
        
        # p80: 50 x 1 x 80 (50 = 44 + 1 + 2 + 3)
        self.fc_q = nn.Sequential( # 400 -> 1
            Linear(80, 40, dropout=0.012, bn=False),
            Linear(40, 20, dropout=0.006, bn=False),
            Linear(20, 10, bn=False),
            Linear(10, 5, bn=False),
            Linear(5, 1, bn=False, activation=None),
        )
        self.fc_l = nn.Sequential( # 400 -> 1
            Linear(80, 40, dropout=0.012, bn=False),
            Linear(40, 20, dropout=0.006, bn=False),
            Linear(20, 10, bn=False),
            Linear(10, 5, bn=False),
            Linear(5, 1, bn=False, activation=None),
        )
        self.fc_r = nn.Sequential( # 400 -> 1
            Linear(80, 40, dropout=0.012, bn=False),
            Linear(40, 20, dropout=0.006, bn=False),
            Linear(20, 10, bn=False),
            Linear(10, 5, bn=False),
            Linear(5, 1, bn=False, activation=None),
        )
        self.fc_c = nn.Sequential( # 400 -> 1
            Linear(80, 40, dropout=0.012, bn=False),
            Linear(40, 20, dropout=0.006, bn=False),
            Linear(20, 10, bn=False),
            Linear(10, 5, bn=False),
            Linear(5, 1, bn=False, activation='sigmoid'),
        )
        self.fc_type = nn.Sequential( # 400 -> 9
            Linear(80, 55, dropout=0.012, bn=False),
            Linear(55, 38, dropout=0.006, bn=False),
            Linear(38, 24, bn=False),
            Linear(24, 16, bn=False),
            Linear(16, 9, bn=False),
        )
        self.fc_dot_t = nn.Sequential( # 400 -> 4
            Linear(80, 50, dropout=0.012, bn=False),
            Linear(50, 28, dropout=0.006, bn=False),
            Linear(28, 16, bn=False),
            Linear(16, 8, bn=False),
            Linear(8, 4, bn=False),
        )
        self.fc_dot_b = nn.Sequential( # 400 -> 4
            Linear(80, 50, dropout=0.012, bn=False),
            Linear(50, 28, dropout=0.006, bn=False),
            Linear(28, 16, bn=False),
            Linear(16, 8, bn=False),
            Linear(8, 4, bn=False),
        )
        self.fc_dot_r = nn.Sequential( # 400 -> 4
            Linear(80, 50, dropout=0.012, bn=False),
            Linear(50, 28, dropout=0.006, bn=False),
            Linear(28, 16, bn=False),
            Linear(16, 8, bn=False),
            Linear(8, 4, bn=False),
        )
        self.fc_uline = nn.Sequential( # 400 -> 4
            Linear(80, 50, dropout=0.012, bn=False),
            Linear(50, 28, dropout=0.006, bn=False),
            Linear(28, 16, bn=False),
            Linear(16, 8, bn=False),
            Linear(8, 4, bn=False),
        )
        
    def forward(self, x):
        x = x.unsqueeze(1) # 320 x 1280 -> 1 x 320 x 1280, watch out for batch!
        
        x = self.p1280_convs(x)
        x = self.p1280_step(x)
        
        x = self.p640_convs(x)
        x = self.p640_step(x)
        
        x = self.p320_convs(x)
        x = self.p320_step(x)
        
        x = self.p160_convs(x)
        x = self.p160_step(x)
        
        xxx = self.to_20(x)
        x20_skipped = self.skip_20(xxx)
        xxx = self.to_5(xxx)
        x5_skipped = self.skip_5(xxx)
        xxx = self.to_1(xxx)
        xxx = self.from_1(xxx)
        xxx = self.from_5(torch.cat([xxx, x5_skipped], axis=1))
        xxx = self.from_20(torch.cat([xxx, x20_skipped], axis=1))
        x = self.squeeze(torch.cat([x, xxx], dim=1))
        
        x = x.squeeze(-2).transpose(-1, -2)
        return {
            'q': self.fc_q(x).squeeze(-1),
            'l': self.fc_l(x).squeeze(-1),
            'r': self.fc_r(x).squeeze(-1),
            'c': self.fc_c(x).squeeze(-1),
            'type': self.fc_type(x),
            'dot_t': self.fc_dot_t(x),
            'dot_b': self.fc_dot_b(x),
            'dot_r': self.fc_dot_r(x),
            'uline': self.fc_uline(x),
        }