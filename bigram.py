import torch
import torch.nn as nn
from torch.nn import functional as F


C1 = 384
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2



class BigramLanguageModel_new(nn.Module):


  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, C1)
    self.position_embedding_table = nn.Embedding(block_size, C1)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(C1)
    self.lm_head = nn.Linear(C1, vocab_size)


  def forward(self, idx, targets=None):
    B,T = idx.shape
    tok_emb = self.token_embedding_table(idx) #
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))
    x = tok_emb + pos_emb # (B,T,C) array, includes both the information about the tokens and their positions in the sequence
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self,idx,max_new_tokens):
    for i in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # focus on the last time step
            probs = F.softmax(logits, dim=-1) # probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # get the i +1th prediction
            idx = torch.cat((idx, idx_next), dim=1)  # concatenate the prediction with the current sequence
    return idx