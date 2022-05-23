# sys
import os
import sys
import numpy as np
import random
import pickle
import json
# torch
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# operation
from . import tools

class Feeder_alphapose(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition in both Halpe 136 and Halpe 68 datasets, ran on AlphaPose
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        sort_method: the method to sort the samples, 'area_sum', 'score' or 'random'
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move: If true, perform randomly but continuously changed transformation to input sequence
        window_size: The length of the output sequence
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 sort_method='area_sum',
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 num_person_out=2,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.num_person_out = num_person_out
        self.ignore_empty_sample = ignore_empty_sample
        self.sort_method = sort_method

        self.load_data()

    def load_data(self):
        # load file list
        self.sample_name = os.listdir(self.data_path)

        if self.debug:
            self.sample_name = self.sample_name[:2]

        # load label Json file
        with open(self.label_path, 'r') as f:
            label_info = json.load(f)
        
        sample_id = [name.split('.')[0] for name in self.sample_name]
        self.label = np.array(
            [label_info[id]['label'] for id in sample_id])
        self.has_skeleton = np.array(
            [label_info[id]['has_skeleton'] for id in sample_id])
        
        # ignore samples without skeleton
        if self.ignore_empty_sample:
            self.sample_name = [
              s for h, s in zip(self.has_skeleton, self.sample_name) if h]
            self.label = self.label[self.has_skeleton]

        # output data shape
        self.N = len(self.sample_name)
        self.C = 3 # x, y, score
        self.T = 300
        self.V = 68
        self.M = self.num_person_out

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output data shape (C, T, V, M) for each sample
        # get the data from Json file
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)
        
        data_numpy = np.zeros((self.C, self.T, self.V, self.M))
        
        # Get all values of the 'idx' key in the video_info
        idx_list = np.array([v['idx'] for v in video_info])

        # Seperate the dicts from video_info into a dictionary of {idx: [dicts]}
        idx_dict = {idx: [] for idx in idx_list}
        for k, v  in zip(idx_list, video_info):
            idx_dict[k].append(v)
        
        # Get only the unique idx values
        idx_list = np.unique(idx_list)
        
        if self.sort_method == 'area_sum':
            # print(idx_list)
            idx_list = np.take(idx_list, np.argsort(
              [sum([(v['box'][2] * v['box'][3]) for v in idx_dict[idx]]) for idx in idx_list])[::-1])
            # Reduce the idx_list to the first num_person_out
            idx_list = idx_list[:self.M]
            # print(idx_list)
            # Add to the data_numpy
            for i, idx in enumerate(idx_list):
                for pose_info in idx_dict[idx]:
                    keypoints = np.array(pose_info['keypoints'])
                    frame_id = int(pose_info['image_id'].split('.')[0])
                    data_numpy[0, frame_id, :, i] = (keypoints[0::3] - pose_info['box'][0])/pose_info['box'][2]
                    data_numpy[1, frame_id, :, i] = (keypoints[1::3] - pose_info['box'][1])/pose_info['box'][3]
                    data_numpy[2, frame_id, :, i] = keypoints[2::3]
        elif self.sort_method == 'score':
            idx_list = np.take(idx_list, np.argsort(
              [sum([sum(v['keypoints'][2::3]) for v in idx_dict[idx]]) for idx in idx_list])[::-1])
            # Reduce the idx_list to the first num_person_out
            idx_list = idx_list[:self.M]
            # Add to the data_numpy
            for i, idx in enumerate(idx_list):
                for pose_info in idx_dict[idx]:
                    keypoints = pose_info['keypoints']
                    frame_id = int(pose_info['image_id'].split('.')[0])
                    data_numpy[0, frame_id, :, i] = (keypoints[0::3] - pose_info['box'][0])/pose_info['box'][2]
                    data_numpy[1, frame_id, :, i] = (keypoints[1::3] - pose_info['box'][1])/pose_info['box'][3]
                    data_numpy[2, frame_id, :, i] = keypoints[2::3]
        
        #centralize the data
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # data augmentation
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, self.label[index]
        
    def top_k(self, score, top_k):
        assert (all(self.label >= 0))

        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def top_k_by_category(self, score, top_k):
        assert (all(self.label >= 0))
        return tools.top_k_by_category(self.label, score, top_k)

    def calculate_recall_precision(self, score):
        assert (all(self.label >= 0))
        return tools.calculate_recall_precision(self.label, score)
