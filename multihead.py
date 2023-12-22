import torch 
import torch.nn as nn
from bigram import *
from block import *


class MultiHeadAttention(nn.Module):

  def __init__(self,num_heads,head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)]) # create multiple heads
    self.proj = nn.Linear(C1, C1)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out # concatenate all of the output