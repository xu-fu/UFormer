import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from skimage import filters, morphology, measure


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=1, gamma=2):
        
        #first compute binary cross-entropy 
        BCE = F.cross_entropy(inputs, targets, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        # focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
        focal_loss = (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss.mean()
 