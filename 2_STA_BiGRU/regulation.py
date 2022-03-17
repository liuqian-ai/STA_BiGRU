import torch
import torch.nn as nn

def l2_regularization(model, l2_alpha):
    l2_loss = None
    for module in model.modules():
        if type(module) is nn.GRU:   # 取GRU中的权重
            for name, param in module.named_parameters():
                if(name[:6]=='weight'):
                    if l2_loss is None:
                        l2_loss = param.pow(2).sum() / 2
                    else:
                        l2_loss += param.pow(2).sum() / 2
        if type(module) is nn.Linear:
            for name, param in module.named_parameters():
                if name=='weight':
                    if l2_loss is None:
                        l2_loss = param.pow(2).sum()/2
                    else:
                        l2_loss += param.pow(2).sum() / 2
    return l2_alpha * l2_loss

def spatial_regularization(spatial_score, l1_spatial_lambda,n_steps):   # batch, 130, 1106
    _, T, L = spatial_score.size()
    spatial_reg = torch.sum(torch.square(1 - torch.sum(spatial_score, axis=1) / n_steps), axis=1) / L
    return l1_spatial_lambda * spatial_reg

def temporal_regularization(time_score, l1_temporal_lambda,n_steps):
    temporal_reg = torch.sum(torch.square(1-torch.sum(time_score,axis=1)/n_steps),axis=1) / n_steps
    return l1_temporal_lambda*temporal_reg
