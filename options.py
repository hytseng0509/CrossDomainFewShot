import numpy as np
import os
import glob
import argparse

def parse_args(script):
  parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
  parser.add_argument('--dataset', default='miniImagenet', help='cub/cars/places/plantae, specify multi for training with multiple domains')
  parser.add_argument('--testset', default='cub', help='cub/cars/places/plantae, valid only when dataset=multi')
  parser.add_argument('--model', default='Conv4', help='model: Conv{4|6} / ResNet{10|18|34}')
  parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training')
  parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ')
  parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support')
  parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ')
  parser.add_argument('--name'        , default='tmp', type=str, help='')
  parser.add_argument('--save_dir'    , default='./output', type=str, help='')
  parser.add_argument('--data_dir'    , default='./filelists', type=str, help='')

  if script == 'train':
    parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline')
    parser.add_argument('--save_freq'   , default=25, type=int, help='Save frequency')
    parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
    parser.add_argument('--stop_epoch'  , default=500, type=int, help ='Stopping epoch')
    parser.add_argument('--resume'      , default='', type=str, help='continue from previous trained model with largest epoch')
    parser.add_argument('--resume_epoch', default=-1, type=int, help='')
    parser.add_argument('--warmup'      , default='gg3b0', type=str, help='continue from baseline, neglected if resume is true')
  elif script == 'save_features':
    parser.add_argument('--split'       , default='novel', help='base/val/novel')
    parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
  elif script == 'test':
    parser.add_argument('--split'       , default='novel', help='base/val/novel')
    parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1') #please match the one used in save_features
    parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
  else:
    raise ValueError('Unknown script')

  return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
  assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
  return assign_file

def get_resume_file(checkpoint_dir, resume_epoch=-1):
  filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
  if len(filelist) == 0:
    return None

  filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
  epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
  max_epoch = np.max(epochs)
  epoch = max_epoch if resume_epoch == -1 else resume_epoch
  resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
  return resume_file

def get_best_file(checkpoint_dir):
  best_file = os.path.join(checkpoint_dir, 'best_model.tar')
  if os.path.isfile(best_file):
    return best_file
  else:
    return get_resume_file(checkpoint_dir)
