import os
import cv2
import argparse
import numpy as np
import pandas as pd


from glob import glob
from tqdm import tqdm
from time import time

import batchgenerators
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.utilities.data_splitting import get_split_deterministic_stratified
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.augmentations.crop_and_pad_augmentations import get_lbs_for_random_crop

from core.utils import helpers
from core.dataset.common import get_patients, get_train_transform


class LungSegRegDataLoader(DataLoader):
    def __init__(self,
                 data_path,
                 csv_path,
                 data,
                 batch_size,
                 patch_size,
                 num_threads_in_multithreaded,
                 seed_for_shuffle=1234,
                 return_incomplete=False,
                 shuffle=True,
                 infinite=True):
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)

        self.data_path = data_path
        self.patch_size = patch_size
        self.input_channel = 1
        self.output_channel = 1
        self.indices = list(range(len(data)))
        self.df = pd.read_csv(csv_path)

    def load_class(self, patient_id):
        row = self.df[self.df["patient_name"]==patient_id]
        return row["type"]

    def load_patient(self, patient_id):
        try:
            data = np.load("{}/{}_x.npy".format(self.data_path, patient_id), mmap_mode='r')
            seg = np.load("{}/{}_y.npy".format(self.data_path, patient_id), mmap_mode='r') / 255.
            prior = np.load("{}/{}_template.npy".format(self.data_path, patient_id), mmap_mode='r')
        except:
            data = np.load("{}/{}_x.npy".format(self.data_path, patient_id), mmap_mode='r+')
            seg = np.load("{}/{}_y.npy".format(self.data_path, patient_id), mmap_mode='r+') / 255.
            prior = np.load("{}/{}_template.npy".format(self.data_path, patient_id), mmap_mode='r+')
        data = np.expand_dims(data, axis=0)
        seg = np.expand_dims(seg, axis=0)
        prior = np.expand_dims(prior, axis=0)
        cls = self.load_class(patient_id)
        return data, seg, prior, cls

    def crop(self, data, seg, prior, crop_size):
        lbs = get_lbs_for_random_crop(crop_size, data.shape, margins=[0, 0, 0])
        data_return = data[:, :, lbs[0]:lbs[0]+crop_size[0], lbs[1]:lbs[1]+crop_size[1], lbs[2]:lbs[2]+crop_size[2]]
        seg_return = seg[:, :, lbs[0]:lbs[0]+crop_size[0], lbs[1]:lbs[1]+crop_size[1], lbs[2]:lbs[2]+crop_size[2]]
        prior_return = prior[:, :, lbs[0]:lbs[0]+crop_size[0], lbs[1]:lbs[1]+crop_size[1], lbs[2]:lbs[2]+crop_size[2]]
        return data_return, seg_return, prior_return

    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for data and seg
        data = np.zeros((self.batch_size, self.input_channel, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, self.output_channel, *self.patch_size), dtype=np.float32)
        prior = np.zeros((self.batch_size, self.input_channel, *self.patch_size), dtype=np.float32)
        classes = []

        patient_names = []

        for i, j in enumerate(patients_for_batch):
            patient_data, patient_seg, patient_prior, cls = self.load_patient(j)
            patient_data, patient_seg, patient_prior = self.crop(data=np.expand_dims(patient_data, axis=0), # +b, c, w, h, z
                                                                 seg=np.expand_dims(patient_seg, axis=0),
                                                                 prior=np.expand_dims(patient_prior, axis=0),
                                                                 crop_size=self.patch_size)
            data[i] = patient_data
            seg[i] = patient_seg
            prior[i] = patient_prior
            classes.append(cls)
            patient_names.append(j)

        # data = np.transpose(data, (0, 2, 3, 4, 1))
        # seg = np.transpose(seg, (0, 2, 3, 4, 1))

        return {'data': data, 'seg': seg, 'prior': prior, 'names': patient_names, 'class': classes}


def get_generator(args, split=True):
    """
    obtain data generators for training data and validation data
    :param args:
    :return:
    """
    patients = get_patients(args.data_path)

    print("[x] found %d patients" % len(patients))
    patch_size = (args.patch_x, args.patch_y, args.patch_z)

    df = pd.read_csv(args.csv_path)
    classes = df.type.values.tolist()

    if split:
        train_patients, val_patients = get_split_deterministic_stratified(patients, classes, fold=args.cv, num_splits=args.cv_max, random_state=12345)
        dataloader_train = LungSegRegDataLoader(args.data_path, args.csv_path, train_patients, args.batch_size, patch_size, 1)
        dataloader_validation = LungSegRegDataLoader(args.data_path, args.csv_path, val_patients, args.batch_size, patch_size, 1)

        if args.n_workers > 1:
            # use data augmentation shown in tr_transforms to augment training data
            # use plain test data without augmentation for testing data
            #tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms, num_processes=args.n_workers, num_cached_per_queue=5, pin_memory=False)
            tr_gen = MultiThreadedAugmenter(dataloader_train, None, num_processes=args.n_workers, num_cached_per_queue=5, pin_memory=False)
            val_gen = MultiThreadedAugmenter(dataloader_validation, None,
                                             num_processes=args.n_workers,
                                             num_cached_per_queue=5,
                                             pin_memory=False)

            tr_gen.restart()
            val_gen.restart()
        else:
            tr_gen = SingleThreadedAugmenter(dataloader_train, None)
            val_gen = SingleThreadedAugmenter(dataloader_train, None)
        return tr_gen, val_gen, train_patients, val_patients
    else:
        dataloader = LungSegRegDataLoader(args.data_path, patients, args.batch_size, patch_size, 1)
        data_gen = SingleThreadedAugmenter(dataloader, None)
        return data_gen, patients


# def test():
#      # txt file path
#     patients = get_patients('/media/zhaochen/data/COVID/data_sh/numpy')
#     print(len(patients))
#
#     train_patients, val_patients = get_split_deterministic(patients, fold=0, num_splits=10, random_state=12345)
#
#     patch_size = (192, 192, 32)
#     batch_size = 2
#
#     # npy path
#     data_path = '/home/zhaochen/Desktop/femur/experiments/exp_s0_cv4/10000/'
#     num_threads_in_multithreaded = 2
#
#     dataloader = get_patients(data_path, train_patients, batch_size, patch_size, 1)
#
#     batch = next(dataloader)
#     print(batch)
#                        #LungSegRegDataLoader(data_path, train_patients, batch_size, patch_size, num_threads_in_multithreaded)
#     dataloader_train = LungSegRegDataLoader(data_path, train_patients, batch_size, patch_size, 1)
#     dataloader_validation = LungSegRegDataLoader(data_path, val_patients, batch_size, patch_size, 1)
#
#     tr_transforms = get_train_transform(patch_size, spatial=False)
#
#     tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms,
#                                     num_processes=num_threads_in_multithreaded,
#                                     num_cached_per_queue=3,
#                                     pin_memory=False)
#     val_gen = MultiThreadedAugmenter(dataloader_validation, None,
#                                      num_processes=num_threads_in_multithreaded,
#                                      num_cached_per_queue=3,
#                                      pin_memory=False)
#
#     tr_gen.restart()
#     val_gen.restart()
#
#     num_batches_per_epoch = 100
#     num_validation_batches_per_epoch = 3
#     num_epochs = 5
#     # let's run this to get a time on how long it takes
#     time_per_epoch = []
#     start = time()
#     for epoch in range(num_epochs):
#         start_epoch = time()
#         for b in range(num_batches_per_epoch):
#             batch = next(tr_gen)
#             # do network training here with this batch
#
#         for b in range(num_validation_batches_per_epoch):
#             batch = next(val_gen)
#             # run validation here
#         end_epoch = time()
#         time_per_epoch.append(end_epoch - start_epoch)
#     end = time()
#     total_time = end - start
#     print("Running %d epochs took a total of %.2f seconds with time per epoch being %s" %
#           (num_epochs, total_time, str(time_per_epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--patch_x', type=int, default=192)
    parser.add_argument('--patch_y', type=int, default=192)
    parser.add_argument('--patch_z', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='/media/zhaochen/data/covid_seg/data/data_sh_all/numpy')
    parser.add_argument('--dicom_path', type=str, default='/media/zhaochen/data/covid_seg/data/data_sh_all/images')
    parser.add_argument('--csv_path', type=str, default='/media/zhaochen/data/covid_seg/data/data_sh_all/classification.csv')
    # cpu
    parser.add_argument('--n_workers', type=int, default=6)

    # CV
    parser.add_argument('--cv', type=int, default=0)  # cross validation, CV=5
    parser.add_argument('--cv_max', type=int, default=5)

    # train
    parser.add_argument('--train', type=helpers.str2bool, default=True)

    args = parser.parse_args()

    data_gen_tr, data_gen_val, train_patients, val_patients = get_generator(args, split=args.train)

    for epoch in tqdm(range(10)):
        print("[x] epoch: %d, training" % epoch)
        epoch_loss = 0.
        for mini_batch in range(len(train_patients)//args.batch_size):
            batch_data = next(data_gen_tr)