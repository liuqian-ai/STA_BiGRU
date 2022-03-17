import torch
import torch.nn as nn
import math

class Time_Attention(nn.Module):
    def __init__(self, in_channels, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.kernel = nn.Parameter(torch.FloatTensor(in_channels, self.output_dim))
        stdv = 1. / math.sqrt(self.output_dim)
        self.kernel.data.uniform_(-stdv, stdv)

        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, enc_out, bigru_out):
        # x.shape = (batch_size, time_steps, seq_len)
        WX = torch.matmul(enc_out, self.kernel) + self.bias
        relu_WX = self.relu(WX)
        time_scores = self.softmax(relu_WX)
        return time_scores, torch.matmul(time_scores, bigru_out)



class Self_Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.kernel = nn.Parameter(torch.FloatTensor(3, in_channels, in_channels))
        stdv = 1. / math.sqrt(in_channels)
        self.kernel.data.uniform_(-stdv, stdv)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)

    def forward(self, x):
        WQ = torch.matmul(x, self.kernel[0])
        WK = torch.matmul(x, self.kernel[1])
        WV = torch.matmul(x, self.kernel[2])
        QK = torch.matmul(WQ, WK.permute(0, 2, 1).contiguous())
        QK = QK / (self.in_channels ** 0.5)
        QK = self.softmax1(QK)
        # V = self.softmax2(torch.matmul(QK, WV))
        V = torch.matmul(QK, WV)
        return QK, V