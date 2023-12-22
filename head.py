import torch
import torch.nn as nn
from torch.nn import functional as F
from bigram import *


class Head(nn.Module):


  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(C1, head_size, bias=False)
    self.query = nn.Linear(C1, head_size, bias=False)
    self.value = nn.Linear(C1, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size))) # this creates the lower triangle matrix
    self.dropout = nn.Dropout(dropout)

  def forward(self, x): # copied from above
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)

    wei = q @ k.transpose(-2, -1) * C ** 0.5
    wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
    wei = F.softmax(wei,dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out