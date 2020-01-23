# This code is modified from https://github.com/floodsung/LearningToCompare_FSL

from methods import backbone
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import utils

class RelationNet(MetaTemplate):
  def __init__(self, model_func,  n_way, n_support, tf_path=None, loss_type = 'mse'):
    super(RelationNet, self).__init__(model_func,  n_way, n_support, flatten=False, tf_path=tf_path)

    # loss function
    self.loss_type = loss_type  #'softmax' or 'mse'
    if self.loss_type == 'mse':
      self.loss_fn = nn.MSELoss()
    else:
      self.loss_fn = nn.CrossEntropyLoss()

    # metric function
    self.relation_module = RelationModule( self.feat_dim , 8, self.loss_type ) #relation net features are not pooled, so self.feat_dim is [dim, w, h]
    self.method = 'RelationNet'

  def set_forward(self,x,is_feature = False):

    # get features
    z_support, z_query  = self.parse_feature(x,is_feature)
    z_support   = z_support.contiguous()
    z_proto     = z_support.view( self.n_way, self.n_support, *self.feat_dim ).mean(1)
    z_query     = z_query.contiguous().view( self.n_way* self.n_query, *self.feat_dim )

    # get relations with metric function
    z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query* self.n_way,1,1,1,1)
    z_query_ext = z_query.unsqueeze(0).repeat( self.n_way,1,1,1,1)
    z_query_ext = torch.transpose(z_query_ext,0,1)
    extend_final_feat_dim = self.feat_dim.copy()
    extend_final_feat_dim[0] *= 2
    relation_pairs = torch.cat((z_proto_ext,z_query_ext),2).view(-1, *extend_final_feat_dim)
    relations = self.relation_module(relation_pairs).view(-1, self.n_way)
    return relations

  def set_forward_loss(self, x):
    y = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))

    scores = self.set_forward(x)
    if self.loss_type == 'mse':
      y_oh = utils.one_hot(y, self.n_way)
      y_oh = y_oh.cuda()
      loss = self.loss_fn(scores, y_oh)
    else:
      y = y.cuda()
      loss = self.loss_fn(scores, y)
    return scores, loss

# --- Convolution block used in the relation module ---
class RelationConvBlock(nn.Module):
  maml = False
  def __init__(self, indim, outdim, padding = 0):
    super(RelationConvBlock, self).__init__()
    self.indim  = indim
    self.outdim = outdim
    if self.maml:
      self.C      = backbone.Conv2d_fw(indim, outdim, 3, padding=padding)
      self.BN     = backbone.BatchNorm2d_fw(outdim, momentum=1, track_running_stats=False)
    else:
      self.C      = nn.Conv2d(indim, outdim, 3, padding = padding )
      self.BN     = nn.BatchNorm2d(outdim, momentum=1, affine=True, track_running_stats=False)
    self.relu   = nn.ReLU()
    self.pool   = nn.MaxPool2d(2)

    self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]

    for layer in self.parametrized_layers:
      backbone.init_layer(layer)

    self.trunk = nn.Sequential(*self.parametrized_layers)

  def forward(self,x):
    out = self.trunk(x)
    return out

# --- Relation module adopted in RelationNet ---
class RelationModule(nn.Module):
  maml = False
  def __init__(self,input_size,hidden_size, loss_type = 'mse'):
    super(RelationModule, self).__init__()

    self.loss_type = loss_type
    padding = 1 if ( input_size[1] <10 ) and ( input_size[2] <10 ) else 0 # when using Resnet, conv map without avgpooling is 7x7, need padding in block to do pooling

    self.layer1 = RelationConvBlock(input_size[0]*2, input_size[0], padding = padding )
    self.layer2 = RelationConvBlock(input_size[0], input_size[0], padding = padding )

    shrink_s = lambda s: int((int((s- 2 + 2*padding)/2)-2 + 2*padding)/2)

    if self.maml:
      self.fc1 = backbone.Linear_fw( input_size[0]* shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size )
      self.fc2 = backbone.Linear_fw( hidden_size,1)
    else:
      self.fc1 = nn.Linear( input_size[0]* shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size )
      self.fc2 = nn.Linear( hidden_size,1)

  def forward(self,x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.view(out.size(0),-1)
    out = F.relu(self.fc1(out))
    if self.loss_type == 'mse':
      out = torch.sigmoid(self.fc2(out))
    elif self.loss_type == 'softmax':
      out = self.fc2(out)

    return out
