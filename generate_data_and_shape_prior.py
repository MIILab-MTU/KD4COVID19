import numpy as np
import argparse
import os
import cv2
import pydicom as dicom
import SimpleITK as sitk
import matplotlib.pyplot as plt

from core.utils.visualize import DataVisualizer
from core.utils.lung_segmentation_threshold import segment_morphology2

from glob import glob
from tqdm import tqdm


def load_dicoms(dicom_file_path):
    dicom_files = glob(dicom_file_path)
    print("{}, # dicoms {}".format(dicom_file_path, len(dicom_files)))
    slices = [dicom.read_file(dcm) for dcm in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    image = np.stack([s.pixel_array for s in slices], axis=-1)

    print("content date {}, min = {}, max = {}".format(dicom.read_file(dicom_files[0]).ContentDate, np.min(image), np.max(image)))

    return image


def load_dicoms_itk(dicom_file_path):
    dicom_files = glob(dicom_file_path)
    print("{}, # dicoms {}".format(dicom_file_path, len(dicom_files)))
    slices = [dicom.read_file(dcm) for dcm in dicom_files]
    idxs = np.argsort([slice.ImagePositionPatient[2] for slice in slices])
    image = np.stack([np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(dicom_files[idx]))) for idx in idxs], axis=-1)

    print("content date {}, min = {}, max = {}".format(dicom.read_file(dicom_files[0]).ContentDate, np.min(image), np.max(image)))
    return image

'''
python3 generate_data_and_shape_prior.py \
  --dicom_path=/media/zhaochen/data/covid_seg/data/data_sh_covid/images\
  --label_path=/media/zhaochen/data/covid_seg/data/data_sh_covid/binary \
  --visualize_path=/media/zhaochen/data/covid_seg/data/data_sh_covid/visualize \
  --numpy_path=/media/zhaochen/data/covid_seg/data/data_sh_covid/numpy \
  --mode=1

python3 generate_data_and_shape_prior.py \
  --dicom_path=/media/zhaochen/data/covid_seg/data/data_sh_normal/images\
  --label_path=/media/zhaochen/data/covid_seg/data/data_sh_normal/binary \
  --visualize_path=/media/zhaochen/data/data/covid_seg/data/data_sh_normal/visualize \
  --numpy_path=/media/zhaochen/data/covid_seg/data/data_sh_normal/numpy \
  --mode=2

python3 generate_data_and_shape_prior.py \
  --dicom_path=/media/zhaochen/data/covid_seg/data/data_sh_normal_pneumonia/images\
  --label_path=/media/zhaochen/data/covid_seg/data/data_sh_normal_pneumonia/binary \
  --visualize_path=/media/zhaochen/data/covid_seg/data/data_sh_normal_pneumonia/visualize \
  --numpy_path=/media/zhaochen/data/covid_seg/data/data_sh_normal_pneumonia/numpy \
  --mode=1
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data_sh_normal_pneumonia
    parser.add_argument("--dicom_path", type=str, default="/home/zhaochen/Desktop/covid/data_sh_segmentation/images")
    parser.add_argument("--label_path", type=str, default="/home/zhaochen/Desktop/covid/data_sh_segmentation/binary")
    parser.add_argument("--visualize_path", type=str, default="/home/zhaochen/Desktop/data/covid/data_sh_segmentation/visualize")
    parser.add_argument("--numpy_path", type=str, default="/home/zhaochen/Desktop/covid/data_sh_segmentation/numpy")
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--mode", type=int,  default=1, choices=[1, 2])
    args = parser.parse_args()

    if not os.path.isdir(args.visualize_path):
        os.makedirs(args.visualize_path)

    if not os.path.isdir(args.numpy_path):
        os.makedirs(args.numpy_path)

    for patient_dir in tqdm(os.listdir(args.dicom_path)):
        if os.path.isdir(os.path.join(args.dicom_path, patient_dir)):
            print("-------------------------------------------------------")
            # original dicom file process
            if args.mode == 1:
                image = load_dicoms(os.path.join(args.dicom_path, patient_dir, "*.dcm"))
            else:
                image = load_dicoms_itk(os.path.join(args.dicom_path, patient_dir, "*.dcm"))
            print(image.shape)

            #image = np.interp(image, (image.min(), image.max()), (-2000, +4000))
            #print(f"after interp, min = {np.min(image)}, max = {np.max(image)}")

            # mask generation
            mask = segment_morphology2(image, volume_ratio=0.005, binary_threshold=-320)

            # label images
            label_image_path = os.path.join(args.label_path, patient_dir, "label")
            label_image_paths = glob(os.path.join(label_image_path, "*.png"))
            label_image_paths = sorted(label_image_paths)[::-1]
            label = np.zeros((512, 512, len(label_image_paths)))

            for i in range(len(label_image_paths)):
                label[:, :, i] = cv2.imread(label_image_paths[i], cv2.IMREAD_GRAYSCALE)

            if not os.path.isdir(os.path.join(args.visualize_path, patient_dir)):
                os.makedirs(os.path.join(args.visualize_path, patient_dir))

            for i in range(image.shape[2]):
                plt.imsave(fname=os.path.join(args.visualize_path, patient_dir, "x_{}.png".format(i)), arr=image[:, :, i], cmap="gray")
                plt.imsave(fname=os.path.join(args.visualize_path, patient_dir, "y_{}.png".format(i)), arr=label[:, :, i], cmap="gray")
                plt.imsave(fname=os.path.join(args.visualize_path, patient_dir, "template_{}.png".format(i)), arr=mask[:, :, i], cmap="gray")

            shape = image.shape
            crop_size = args.crop_size
            # image = (image - np.min(image)) / (np.max(image) - np.min(image)) # new
            image = image[crop_size: shape[0] - crop_size, crop_size: shape[1] - crop_size, :]
            label = label[crop_size: shape[0] - crop_size, crop_size: shape[1] - crop_size, :]
            mask = mask[crop_size: shape[0] - crop_size, crop_size: shape[1] - crop_size, :]

            # visualize
            dv = DataVisualizer([image, label, mask], save_path=os.path.join(args.visualize_path, "{}.png".format(patient_dir)))
            # dv.visualize_np(image.shape[2], patch_col=512, patch_row=512)
            dv.visualize(shape[2])

            # save to numpy
            np.save(file=os.path.join(args.numpy_path, "{}_x.npy".format(patient_dir)), arr=image)
            np.save(file=os.path.join(args.numpy_path, "{}_y.npy".format(patient_dir)), arr=label)
            np.save(file=os.path.join(args.numpy_path, "{}_template.npy".format(patient_dir)), arr=mask)

            print("image.shape = {}, label.shape = {}, mask.shape = {}".format(image.shape, label.shape, mask.shape))
