import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

import sys
from data_utils.indoor3d_util import collect_point_label

def normalized(data_label):
    data = data_label[:,0:6]
    max_room_x = max(data[:,0])
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])
    data_label[:, 0] = (data_label[:, 0]/max_room_x) * 10
    data_label[:, 1] = (data_label[:, 1]/max_room_y) * 10
    data_label[:, 2] = (data_label[:, 2]/max_room_z) * 10
    
    return data_label

# data = np.load('/home/wanghe/workspace/Pointnet_Pointnet2_pytorch/data/stanford_indoor3d/Area_2_hallway_1.npy')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # ./data_utils
DATA_PATH = os.path.join(BASE_DIR, 'data/south/trainval_fullarea/train_area')
scenes = sorted(os.listdir(DATA_PATH))


output_folder = os.path.join(BASE_DIR, 'data/south/prepared_data/trainval_fullarea/train_area')    # data/stsanford_indoor3d
if not os.path.exists(output_folder):
    os.mkdir(output_folder)


for scene in scenes:
    scene_path = os.path.join(DATA_PATH, scene)
    print(scene_path)
    data_label = np.loadtxt(scene_path)
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min
    data_label = normalized(data_label)
    with open(os.path.join(output_folder, scene), 'w') as f:
        np.savetxt(f, data_label)
    print('Already saved scene {}'.format(scene))



