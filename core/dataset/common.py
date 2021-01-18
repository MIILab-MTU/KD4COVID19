import numpy as np
import os
import pydicom as dicom

from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms import Compose

from glob import glob


def get_pixel_spacing(dicom_path):
    patient_dicom_files = glob(os.path.join(dicom_path, "*.dcm"))
    pixel_spacing = dicom.read_file(patient_dicom_files[0]).PixelSpacing
    slice_thickness = dicom.read_file(patient_dicom_files[0]).SliceThickness
    return [float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)]


def get_patients(root_dir):
    image_files = glob(root_dir+"/*_x.npy")
    patient_names = []
    for image_file in image_files:
        patient_names.append(image_file[image_file.rfind("/")+1: image_file.rfind("_x")])
    return patient_names


def get_train_transform(patch_size, spatial=True,
                        rotation_angle=15,
                        elastic_deform=(0, 0.25),
                        scale_factor=(0.75, 1.25),
                        augmentation_prob=0.1):
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    if spatial:
        tr_transforms.append(
            SpatialTransform_2(
                patch_size,
                patch_center_dist_from_border=0, #[i // 2 for i in patch_size]
                do_elastic_deform=False, deformation_scale=elastic_deform,
                do_rotation=False,
                angle_x=(- rotation_angle / 360. * 2 * np.pi, rotation_angle / 360. * 2 * np.pi),
                angle_y=(- rotation_angle / 360. * 2 * np.pi, rotation_angle / 360. * 2 * np.pi),
                angle_z=(- rotation_angle / 360. * 2 * np.pi, rotation_angle / 360. * 2 * np.pi),
                do_scale=False, scale=scale_factor,
                border_mode_data='constant', border_cval_data=0,
                border_mode_seg='constant', border_cval_seg=0,
                order_seg=1, order_data=1,
                random_crop=True,
                p_el_per_sample=augmentation_prob,
                p_rot_per_sample=augmentation_prob,
                p_scale_per_sample=augmentation_prob
            )
        )

        # now we mirror along all axes
        tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness transform for 15% of samples
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5),
                                                            per_channel=True,
                                                            p_per_sample=augmentation_prob))

    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # we can also invert the image, apply the transform and then invert back
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))

    # Gaussian Noise
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

    # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
    # thus make the seg_model more robust to it
    #tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,  p_per_channel=0.5, p_per_sample=0.15))

    # new TODO
    #tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.5, 2), per_channel=True, p_per_sample=0.15))
    #tr_transforms.append(ContrastAugmentationTransform(contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, p_per_sample=0.15))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms