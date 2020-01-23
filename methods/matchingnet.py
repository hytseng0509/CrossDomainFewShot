# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

from methods import backbone
import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate
import utils

class MatchingNet(MetaTemplate):
  def __init__(self, model_func,  n_way, n_support, tf_path=None):
    super(MatchingNet, self).__init__( model_func,  n_way, n_support, tf_path=tf_path)

    # loss function
    self.loss_fn    = nn.NLLLoss()

    # metric
    self.FCE = FullyContextualEmbedding(self.feat_dim)
    self.G_encoder = backbone.LSTM(self.feat_dim, self.feat_dim, 1, batch_first=True, bidirectional=True)

    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)
    self.method = 'MatchingNet'

  def encode_training_set(self, S, G_encoder = None):
    if G_encoder is None:
      G_encoder = self.G_encoder
    out_G = G_encoder(S.unsqueeze(0))
    out_G = out_G.squeeze(0)
    G = S + out_G[:,:S.size(1)] + out_G[:,S.size(1):]
    G_norm = torch.norm(G,p=2, dim =1).unsqueeze(1).expand_as(G)
    G_normalized = G.div(G_norm+ 0.00001)
    return G, G_normalized

  def get_logprobs(self, f, G, G_normalized, Y_S, FCE = None):
    if FCE is None:
      FCE = self.FCE
    F = FCE(f, G)
    F_norm = torch.norm(F,p=2, dim =1).unsqueeze(1).expand_as(F)
    F_normalized = F.div(F_norm + 0.00001)
    scores = self.relu( F_normalized.mm(G_normalized.transpose(0,1))  ) *100 # The original paper use cosine simlarity, but here we scale it by 100 to strengthen highest probability after softmax
    softmax = self.softmax(scores)
    logprobs =(softmax.mm(Y_S)+1e-6).log()
    return logprobs

  def set_forward(self, x, is_feature = False):
    z_support, z_query  = self.parse_feature(x,is_feature)

    z_support   = z_support.contiguous().view( self.n_way* self.n_support, -1 )
    z_query     = z_query.contiguous().view( self.n_way* self.n_query, -1 )
    G, G_normalized = self.encode_training_set( z_support)

    y_s         = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
    Y_S         = utils.one_hot(y_s, self.n_way).cuda()
    f           = z_query
    logprobs = self.get_logprobs(f, G, G_normalized, Y_S)
    return logprobs

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
    y_query = y_query.cuda()

    logprobs = self.set_forward(x)
    loss = self.loss_fn(logprobs, y_query)

    return logprobs, loss

  def cuda(self):
    super(MatchingNet, self).cuda()
    self.FCE = self.FCE.cuda()
    return self

# --- Fully contextual embedding function adopted in Matchingnet ---
class FullyContextualEmbedding(nn.Module):
  def __init__(self, feat_dim):
    super(FullyContextualEmbedding, self).__init__()
    self.lstmcell = backbone.LSTMCell(feat_dim*2, feat_dim)
    self.softmax = nn.Softmax(dim=1)
    self.c_0 = torch.zeros(1, feat_dim)
    self.feat_dim = feat_dim

  def forward(self, f, G):
    h = f
    c = self.c_0.expand_as(f)
    G_T = G.transpose(0,1)
    K = G.size(0) #Tuna to be comfirmed
    for k in range(K):
      logit_a = h.mm(G_T)
      a = self.softmax(logit_a)
      r = a.mm(G)
      x = torch.cat((f, r),1)

      h, c = self.lstmcell(x, (h, c))
      h = h + f
    return h

  def cuda(self):
    super(FullyContextualEmbedding, self).cuda()
    self.c_0 = self.c_0.cuda()
    return self

