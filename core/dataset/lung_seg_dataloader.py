import os
import cv2
import numpy as np


from glob import glob
from time import time

import batchgenerators
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.utils import pad_nd_image


from core.dataset.common import get_patients, get_train_transform


class LungSegDataloader(DataLoader):
    def __init__(self,
                 data_path,
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

    def load_patient(self, patient_id):
        try:
            data = np.load("{}/{}_x.npy".format(self.data_path, patient_id), mmap_mode='r')
            seg = np.load("{}/{}_y.npy".format(self.data_path, patient_id), mmap_mode='r') / 255.
        except:
            data = np.load("{}/{}_x.npy".format(self.data_path, patient_id), mmap_mode='r+')
            seg = np.load("{}/{}_y.npy".format(self.data_path, patient_id), mmap_mode='r+') / 255.
        data = np.expand_dims(data, axis=0)
        seg = np.expand_dims(seg, axis=0)
        return data, seg

    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for data and seg
        data = np.zeros((self.batch_size, self.input_channel, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, self.output_channel, *self.patch_size), dtype=np.float32)

        patient_names = []

        for i, j in enumerate(patients_for_batch):
            patient_data, patient_seg = self.load_patient(j)
            patient_data, patient_seg = crop(data=np.expand_dims(patient_data, axis=0), # +b, c, w, h, z
                                             seg=np.expand_dims(patient_seg, axis=0),
                                             crop_size=self.patch_size,
                                             crop_type="random")
            data[i] = patient_data
            seg[i] = patient_seg
            patient_names.append(j)

        # data = np.transpose(data, (0, 2, 3, 4, 1))
        # seg = np.transpose(seg, (0, 2, 3, 4, 1))

        return {'data': data, 'seg': seg, 'names': patient_names}


def get_generator(args, split=True):
    """
    obtain data generators for training data and validation data
    :param args:
    :return:
    """
    patients = get_patients(args.data_path)

    print("[x] found %d patients" % len(patients))
    patch_size = (args.patch_x, args.patch_y, args.patch_z)

    if split:
        train_patients, val_patients = get_split_deterministic(patients, fold=args.cv, num_splits=args.cv_max, random_state=12345)

        dataloader_train = LungSegDataloader(args.data_path, train_patients, args.batch_size, patch_size, 1)

        dataloader_validation = LungSegDataloader(args.data_path, val_patients, args.batch_size, patch_size, 1)

        tr_transforms = get_train_transform(patch_size)

        if args.n_workers > 1:
            # use data augmentation shown in tr_transforms to augment training data
            tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms,
                                            num_processes=args.n_workers, num_cached_per_queue=5, pin_memory=False)
            # use plain test data without augmentation for testing data
            val_gen = MultiThreadedAugmenter(dataloader_validation, None,
                                             num_processes=args.n_workers, num_cached_per_queue=5, pin_memory=False)

            tr_gen.restart()
            val_gen.restart()
        else:
            tr_gen = SingleThreadedAugmenter(dataloader_train, tr_transforms)
            val_gen = SingleThreadedAugmenter(dataloader_train, None)
        return tr_gen, val_gen, train_patients, val_patients
    else:
        dataloader = LungSegDataloader(args.data_path, patients, args.batch_size, patch_size, 1)
        data_gen = SingleThreadedAugmenter(dataloader, None)
        return data_gen, patients


if __name__ == '__main__':

    data_root = '/media/zhaochen/data/covid/data_sh/numpy'

    patients = get_patients(data_root)
    print(len(patients))

    train_patients, val_patients = get_split_deterministic(patients, fold=0, num_splits=5, random_state=12345)

    patch_size = (384, 384, 16)
    batch_size = 2

    num_threads_in_multithreaded = 2

    dataloader = LungSegDataloader(data_root, train_patients, batch_size, patch_size, 1)

    batch = next(dataloader)
    print(batch)
                       #LungSegDataloader(data_path, train_patients, batch_size, patch_size, num_threads_in_multithreaded)
    dataloader_train = LungSegDataloader(data_root, train_patients, batch_size, patch_size, 1)

    dataloader_validation = LungSegDataloader(data_root, val_patients, batch_size, patch_size, 1)

    tr_transforms = get_train_transform(patch_size)

    tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms,
                                    num_processes=num_threads_in_multithreaded,
                                    num_cached_per_queue=3,
                                    pin_memory=False)
    val_gen = MultiThreadedAugmenter(dataloader_validation, None,
                                     num_processes=num_threads_in_multithreaded,
                                     num_cached_per_queue=3,
                                     pin_memory=False)

    tr_gen.restart()
    val_gen.restart()

    num_batches_per_epoch = 100
    num_validation_batches_per_epoch = 3
    num_epochs = 5
    # let's run this to get a time on how long it takes
    time_per_epoch = []
    start = time()
    for epoch in range(num_epochs):
        start_epoch = time()
        for b in range(num_batches_per_epoch):
            batch = next(tr_gen)
            # do network training here with this batch

        for b in range(num_validation_batches_per_epoch):
            batch = next(val_gen)
            # run validation here
        end_epoch = time()
        time_per_epoch.append(end_epoch - start_epoch)
    end = time()
    total_time = end - start
    print("Running %d epochs took a total of %.2f seconds with time per epoch being %s" %
          (num_epochs, total_time, str(time_per_epoch)))

