import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
#import json
import random
from subprocess import call

cwd = os.getcwd()
source_path = join(cwd,'source/Plantae')
data_path = join(cwd,'images')
if not os.path.exists(data_path):
    os.makedirs(data_path)
savedir = './'
dataset_list = ['base','val','novel']


folder_list = [f for f in listdir(source_path) if isdir(join(source_path, f))]
#folder_list.sort()
folder_list_count = np.array([len(listdir(join(source_path, f))) for f in folder_list])
folder_list_idx = np.argsort(folder_list_count)
folder_list = np.array(folder_list)[folder_list_idx[-200:]].tolist()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    source_folder_path = join(source_path, folder)
    folder_path = join(data_path, folder)
    classfile_list_all.append( [ cf for cf in listdir(source_folder_path) if (isfile(join(source_folder_path,cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])
    classfile_list_all[i] = classfile_list_all[i][:min(len(classfile_list_all[i]), 600)]

    call('mkdir ' + folder_path, shell=True)
    for cf in classfile_list_all[i]:
      call('cp ' + join(source_folder_path, cf) + ' ' + join(folder_path, cf), shell=True)
    classfile_list_all[i] = [join(folder_path, cf) for cf in classfile_list_all[i]]

for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'base' in dataset:
            if (i%2 == 0):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset:
            if (i%4 == 1):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'novel' in dataset:
            if (i%4 == 3):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
