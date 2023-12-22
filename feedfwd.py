import torch
import torch.nn as nn
from bigram import *
from block import *
from multihead import *


class FeedForward(nn.Module):

  def __init__(self,n_embd):
    super().__init__()
    self.net = nn.Sequential( # multiplication of 4 comes from the fact that the dimensionality of input is x, but the inner layer dimensionality is 4*x
        nn.Linear(n_embd, 4*n_embd), # linear layer with n_embd input and n_embd output
        nn.ReLU(),# activation function, allows for non linearity (we use ReLU to get over vanishing gradients) -> vanishing gradients is essentially when
        nn.Linear(n_embd * 4, n_embd),    #  the gradients are propagated backward from the output layer to the input layer, they can become very small (vanish) as they pass through many layers.
        nn.Dropout(dropout)          # When the gradients become extremely small, the weights of the early layers are updated only by tiny amounts, if at all.
    )



  def forward(self, x):
    return self.net(x)