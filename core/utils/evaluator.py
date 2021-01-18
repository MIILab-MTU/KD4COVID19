import os
import sys

from skimage import filters
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import measure, exposure
from tqdm import tqdm

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from core.utils import lookup_tables



def print_metrics(itr, **kargs):
    print("*** Round {}  ====> ".format(itr))
    for name, value in kargs.items():
        print ("{} : {}, ".format(name, value)),
    print("")
    sys.stdout.flush()



def AUC_ROC(true_vessel_arr, pred_vessel_arr):
    """
    Area under the ROC curve with x axis flipped
    """
    """
    fpr, tpr, _ = roc_curve(true_vessel_arr, pred_vessel_arr)
    try:
        AUC_ROC = roc_auc_score(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    except:
        AUC_ROC = 0.
    return AUC_ROC, fpr, tpr
    """
    try:
        AUC_ROC = roc_auc_score(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    except:
        AUC_ROC = 0.
    return AUC_ROC


def AUC_PR(true_vessel_img, pred_vessel_img):
    """
    Precision-recall curve
    """
    """
    precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(),  pos_label=1)
    try:
        AUC_prec_rec = auc(recall, precision)
    except:
        AUC_prec_rec = 0.
    return AUC_prec_rec, precision, recall
    """
    try:
        precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(), pos_label=1)
        AUC_prec_rec = auc(recall, precision)
    except:
        AUC_prec_rec = 0.
    return AUC_prec_rec


def best_f1_threshold(precision, recall, thresholds):
    best_f1 = -1
    for index in range(len(precision)):
        curr_f1 = 2. * precision[index] * recall[index] / (precision[index] + recall[index])
        if best_f1 < curr_f1:
            best_f1 = curr_f1
            best_threshold = thresholds[index]

    return best_f1, best_threshold


def threshold_by_otsu_local(pred_vessesl, flatten=True, window=128, stride=32):
    assert len(pred_vessesl.shape)==2
    binary_vessel = np.zeros_like(pred_vessesl, dtype=np.uint8)
    for sw_x in range(0, pred_vessesl.shape[0]-window+1, stride):
        for sw_y in range(0, pred_vessesl.shape[1]-window+1, stride):
            local_image = pred_vessesl[sw_x: sw_x + window, sw_y: sw_y + window]
            if np.max(local_image) != np.min(local_image):
                threshold = filters.threshold_otsu(local_image)
                local_bin = np.zeros(shape=[window, window], dtype=np.uint8)
                local_bin[local_image > threshold] = 1
                binary_vessel[sw_x: sw_x+window, sw_y: sw_y+window] += local_bin

    binary_vessel = np.clip(binary_vessel, 0, 1)

    if flatten:
        return binary_vessel.flatten()
    else:
        return binary_vessel


def threshold_by_otsu(pred_vessels, flatten=True):
    # cut by otsu threshold
    threshold = filters.threshold_otsu(pred_vessels)
    pred_vessels_bin = np.zeros(pred_vessels.shape)
    pred_vessels_bin[pred_vessels >= threshold] = 1

    if flatten:
        return pred_vessels_bin.flatten()
    else:
        return pred_vessels_bin


def threshold_by_f1(true_vessels, generated, masks, flatten=True, f1_score=False):
    vessels_in_mask, generated_in_mask = pixel_values_in_mask(true_vessels, generated, masks)
    precision, recall, thresholds = precision_recall_curve(vessels_in_mask.flatten(), generated_in_mask.flatten(),
                                                           pos_label=1)
    best_f1, best_threshold = best_f1_threshold(precision, recall, thresholds)

    pred_vessels_bin = np.zeros(generated.shape)
    pred_vessels_bin[generated >= best_threshold] = 1

    if flatten:
        if f1_score:
            return pred_vessels_bin[masks == 1].flatten(), best_f1
        else:
            return pred_vessels_bin[masks == 1].flatten()
    else:
        if f1_score:
            return pred_vessels_bin, best_f1
        else:
            return pred_vessels_bin


def misc_measures(true_vessels, pred_vessels, masks):
    thresholded_vessel_arr, f1_score = threshold_by_f1(true_vessels, pred_vessels, masks, f1_score=True)
    true_vessel_arr = true_vessels[masks == 1].flatten()

    cm = confusion_matrix(true_vessel_arr, thresholded_vessel_arr)
    acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    sensitivity = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
    specificity = 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
    return f1_score, acc, sensitivity, specificity


def misc_measures_in_train(true_vessel_arr, pred_vessel_arr):
    true_vessel_arr = true_vessel_arr.astype(np.bool)
    pred_vessel_arr = pred_vessel_arr.astype(np.bool)

    cm = confusion_matrix(true_vessel_arr, pred_vessel_arr)
    try:
        acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    except:
        acc = 0.

    try:
        sensitivity = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
    except:
        sensitivity = 0.

    try:
        specificity = 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
    except:
        specificity = 0.
    return acc, sensitivity, specificity


def dice_coefficient(true_vessels, pred_vessels, masks):
    thresholded_vessels = threshold_by_f1(true_vessels, pred_vessels, masks, flatten=False)

    true_vessels = true_vessels.astype(np.bool)
    thresholded_vessels = thresholded_vessels.astype(np.bool)

    intersection = np.count_nonzero(true_vessels & thresholded_vessels)

    size1 = np.count_nonzero(true_vessels)
    size2 = np.count_nonzero(thresholded_vessels)

    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def img_dice(pred_vessel, true_vessel):
    threshold = filters.threshold_otsu(pred_vessel)
    pred_vessels_bin = np.zeros(pred_vessel.shape)
    pred_vessels_bin[pred_vessel >= threshold] = 1
    dice_coeff = dice_coefficient_in_train(true_vessel.flatten(), pred_vessels_bin.flatten())
    return dice_coeff


def vessel_similarity(segmented_vessel_0, segmented_vessel_1):
    try:
        threshold_0 = filters.threshold_otsu(segmented_vessel_0)
        threshold_1 = filters.threshold_otsu(segmented_vessel_1)
        segmented_vessel_0_bin = np.zeros(segmented_vessel_0.shape)
        segmented_vessel_1_bin = np.zeros(segmented_vessel_1.shape)
        segmented_vessel_0_bin[segmented_vessel_0 > threshold_0] = 1
        segmented_vessel_1_bin[segmented_vessel_1 > threshold_1] = 1
        dice_coeff = dice_coefficient_in_train(segmented_vessel_0_bin.flatten(), segmented_vessel_1_bin.flatten())
        return dice_coeff
    except:
        return 0.


def dice_coefficient_in_train(true_vessel_arr, pred_vessel_arr):
    true_vessel_arr = true_vessel_arr.astype(np.bool)
    pred_vessel_arr = pred_vessel_arr.astype(np.bool)

    intersection = np.count_nonzero(true_vessel_arr & pred_vessel_arr)

    size1 = np.count_nonzero(true_vessel_arr)
    size2 = np.count_nonzero(pred_vessel_arr)

    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def operating_pts_human_experts(gt_vessels, pred_vessels, masks):
    gt_vessels_in_mask, pred_vessels_in_mask = pixel_values_in_mask(gt_vessels, pred_vessels, masks, split_by_img=True)

    n = gt_vessels_in_mask.shape[0]
    op_pts_roc, op_pts_pr = [], []
    for i in range(n):
        cm = confusion_matrix(gt_vessels_in_mask[i], pred_vessels_in_mask[i])
        fpr = 1 - 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
        tpr = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
        prec = 1. * cm[1, 1] / (cm[0, 1] + cm[1, 1])
        recall = tpr
        op_pts_roc.append((fpr, tpr))
        op_pts_pr.append((recall, prec))

    return op_pts_roc, op_pts_pr


def pixel_values_in_mask(true_vessels, pred_vessels, masks, split_by_img=False):
    assert np.max(pred_vessels) <= 1.0 and np.min(pred_vessels) >= 0.0
    assert np.max(true_vessels) == 1.0 and np.min(true_vessels) == 0.0
    assert np.max(masks) == 1.0 and np.min(masks) == 0.0
    assert pred_vessels.shape[0] == true_vessels.shape[0] and masks.shape[0] == true_vessels.shape[0]
    assert pred_vessels.shape[1] == true_vessels.shape[1] and masks.shape[1] == true_vessels.shape[1]
    assert pred_vessels.shape[2] == true_vessels.shape[2] and masks.shape[2] == true_vessels.shape[2]

    if split_by_img:
        n = pred_vessels.shape[0]
        return np.array([true_vessels[i, ...][masks[i, ...] == 1].flatten() for i in range(n)]), np.array(
            [pred_vessels[i, ...][masks[i, ...] == 1].flatten() for i in range(n)])
    else:
        return true_vessels[masks == 1].flatten(), pred_vessels[masks == 1].flatten()


def remain_in_mask(imgs, masks):
    imgs[masks == 0] = 0
    return imgs


def crop_to_original(imgs, ori_shape):
    pred_shape = imgs.shape
    assert len(pred_shape) < 4

    if ori_shape == pred_shape:
        return imgs
    else:
        if len(imgs.shape) > 2:
            ori_h, ori_w = ori_shape[1], ori_shape[2]
            pred_h, pred_w = pred_shape[1], pred_shape[2]
            return imgs[:, (pred_h - ori_h) // 2:(pred_h - ori_h) // 2 + ori_h,
                   (pred_w - ori_w) // 2:(pred_w - ori_w) // 2 + ori_w]
        else:
            ori_h, ori_w = ori_shape[0], ori_shape[1]
            pred_h, pred_w = pred_shape[0], pred_shape[1]
            return imgs[(pred_h - ori_h) // 2:(pred_h - ori_h) // 2 + ori_h,
                   (pred_w - ori_w) // 2:(pred_w - ori_w) // 2 + ori_w]


def difference_map(ori_vessel, pred_vessel, mask):
    # ori_vessel : an RGB image

    thresholded_vessel = threshold_by_f1(np.expand_dims(ori_vessel, axis=0), np.expand_dims(pred_vessel, axis=0),
                                         np.expand_dims(mask, axis=0), flatten=False)

    thresholded_vessel = np.squeeze(thresholded_vessel, axis=0)
    diff_map = np.zeros((ori_vessel.shape[0], ori_vessel.shape[1], 3))
    diff_map[(ori_vessel == 1) & (thresholded_vessel == 1)] = (0, 255, 0)  # Green (overlapping)
    diff_map[(ori_vessel == 1) & (thresholded_vessel != 1)] = (255, 0, 0)  # Red (false negative, missing in pred)
    diff_map[(ori_vessel != 1) & (thresholded_vessel == 1)] = (0, 0, 255)  # Blue (false positive)

    # compute dice coefficient for a given image
    overlap = len(diff_map[(ori_vessel == 1) & (thresholded_vessel == 1)])
    fn = len(diff_map[(ori_vessel == 1) & (thresholded_vessel != 1)])
    fp = len(diff_map[(ori_vessel != 1) & (thresholded_vessel == 1)])

    return diff_map, 2. * overlap / (2 * overlap + fn + fp)


def metric_single_img(pred_vessel, true_vessel):
    assert len(pred_vessel.shape) == 3
    assert len(true_vessel.shape) == 3

    pred_vessel_vec = pred_vessel.flatten()
    true_vessel_vec = true_vessel.flatten()

    """
    auc_roc, fpr, tpr = AUC_ROC(true_vessel_vec, pred_vessel_vec)
    auc_pr, precision, recall = AUC_PR(true_vessel_vec, pred_vessel_vec)
    """

    auc_roc = AUC_ROC(true_vessel_vec, pred_vessel_vec)
    auc_pr = AUC_PR(true_vessel_vec, pred_vessel_vec)

    # pred_vessel = exposure.equalize_hist(pred_vessel)
    #binary_vessels = threshold_by_otsu_local(pred_vessel, flatten=False, window=256, stride=64)
    binary_vessels = threshold_by_otsu(pred_vessel, flatten=False)
    binary_vessels_vec = binary_vessels.flatten()

    dice_coeff = dice_coefficient_in_train(true_vessel_vec, binary_vessels_vec)
    acc, sensitivity, specificity = misc_measures_in_train(true_vessel_vec, binary_vessels_vec)

    return binary_vessels, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity


def evaluate_single_image(pred_image, true_image):
    #assert len(pred_image.shape) == 4
    #assert len(true_image.shape) == 4

    pred_image_vec = pred_image.flatten()
    true_image_vec = true_image.flatten()

    auc_roc = AUC_ROC(true_image_vec, pred_image_vec)
    auc_pr = AUC_PR(true_image_vec, pred_image_vec)

    try:
        binary_image = threshold_by_otsu(pred_image, flatten=False)

    except:
        binary_image = np.zeros_like(true_image)
    binary_image_vec = binary_image.flatten()

    dice_coeff = dice_coefficient_in_train(true_image_vec, binary_image_vec)
    acc, sensitivity, specificity = misc_measures_in_train(true_image_vec, binary_image_vec)

    return binary_image, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity


def evaluate_single_image_distance(pred_image, true_image, spacing=[1, 1, 1]):
    hd = hausdorff_distance(pred_image, true_image, spacing)
    asd = average_surface_distance(pred_image, true_image, spacing)
    return hd, asd


def binary_image(pred_image):
    bin_image = threshold_by_otsu(pred_image, flatten=False)
    return bin_image


def hausdorff_distance(predictions, labels, spacing=[1, 1, 1]):
    """
    calculate hausdorff distance from prediction verse labels
    :param predictions: 3D Array
    :param labels: 3D Array
    :param spacing: voxel spacing
    :return:
    """
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    pred_img = sitk.GetImageFromArray(predictions)
    pred_img.SetSpacing(spacing)
    lab_img = sitk.GetImageFromArray(np.asarray(labels, dtype=np.float64))
    lab_img.SetSpacing(spacing)
    hausdorff_distance_filter.Execute(pred_img, lab_img)
    hd = hausdorff_distance_filter.GetHausdorffDistance()
    return hd


def average_surface_distance(predictions, labels, spacing=[1, 1, 1]):
    """
    calculate average surface distance
    :param predictions:
    :param labels:
    :param spacing:
    :return:
    """
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    pred_img = sitk.GetImageFromArray(np.asarray(predictions, np.float64))
    pred_img.SetSpacing(spacing)
    lab_img = sitk.GetImageFromArray(np.asarray(labels, np.float64))
    lab_img.SetSpacing(spacing)
    hausdorff_distance_filter.Execute(pred_img, lab_img)
    asd = hausdorff_distance_filter.GetAverageHausdorffDistance()
    return asd

def _assert_is_numpy_array(name, array):
    """Raises an exception if `array` is not a numpy array."""
    if not isinstance(array, np.ndarray):
        raise ValueError("The argument {!r} should be a numpy array, not a "
                         "{}".format(name, type(array)))


def _check_nd_numpy_array(name, array, num_dims):
    """Raises an exception if `array` is not a `num_dims`-D numpy array."""
    if len(array.shape) != num_dims:
        raise ValueError("The argument {!r} should be a {}D array, not of "
                         "shape {}".format(name, num_dims, array.shape))


def _check_2d_numpy_array(name, array):
    _check_nd_numpy_array(name, array, num_dims=2)


def _check_3d_numpy_array(name, array):
    _check_nd_numpy_array(name, array, num_dims=3)


def _assert_is_bool_numpy_array(name, array):
    _assert_is_numpy_array(name, array)
    if array.dtype != np.bool:
        raise ValueError("The argument {!r} should be a numpy array of type bool, "
                         "not {}".format(name, array.dtype))


def _compute_bounding_box(mask):
    """Computes the bounding box of the masks.

    This function generalizes to arbitrary number of dimensions great or equal
    to 1.

    Args:
      mask: The 2D or 3D numpy mask, where '0' means background and non-zero means
        foreground.

    Returns:
      A tuple:
       - The coordinates of the first point of the bounding box (smallest on all
         axes), or `None` if the mask contains only zeros.
       - The coordinates of the second point of the bounding box (greatest on all
         axes), or `None` if the mask contains only zeros.
    """
    num_dims = len(mask.shape)
    bbox_min = np.zeros(num_dims, np.int64)
    bbox_max = np.zeros(num_dims, np.int64)

    # max projection to the x0-axis
    proj_0 = np.amax(mask, axis=tuple(range(num_dims))[1:])
    idx_nonzero_0 = np.nonzero(proj_0)[0]
    if len(idx_nonzero_0) == 0:  # pylint: disable=g-explicit-length-test
        return None, None

    bbox_min[0] = np.min(idx_nonzero_0)
    bbox_max[0] = np.max(idx_nonzero_0)

    # max projection to the i-th-axis for i in {1, ..., num_dims - 1}
    for axis in range(1, num_dims):
        max_over_axes = list(range(num_dims))  # Python 3 compatible
        max_over_axes.pop(axis)  # Remove the i-th dimension from the max
        max_over_axes = tuple(max_over_axes)  # numpy expects a tuple of ints
        proj = np.amax(mask, axis=max_over_axes)
        idx_nonzero = np.nonzero(proj)[0]
        bbox_min[axis] = np.min(idx_nonzero)
        bbox_max[axis] = np.max(idx_nonzero)

    return bbox_min, bbox_max


def _crop_to_bounding_box(mask, bbox_min, bbox_max):
    """Crops a 2D or 3D mask to the bounding box specified by `bbox_{min,max}`."""
    # we need to zeropad the cropped region with 1 voxel at the lower,
    # the right (and the back on 3D) sides. This is required to obtain the
    # "full" convolution result with the 2x2 (or 2x2x2 in 3D) kernel.
    # bounding box.
    cropmask = np.zeros((bbox_max - bbox_min) + 2, np.uint8)

    num_dims = len(mask.shape)
    # pyformat: disable
    if num_dims == 2:
        cropmask[0:-1, 0:-1] = mask[bbox_min[0]:bbox_max[0] + 1,
                               bbox_min[1]:bbox_max[1] + 1]
    elif num_dims == 3:
        cropmask[0:-1, 0:-1, 0:-1] = mask[bbox_min[0]:bbox_max[0] + 1,
                                     bbox_min[1]:bbox_max[1] + 1,
                                     bbox_min[2]:bbox_max[2] + 1]
    # pyformat: enable
    else:
        assert False

    return cropmask


def _sort_distances_surfels(distances, surfel_areas):
    """Sorts the two list with respect to the tuple of (distance, surfel_area).

    Args:
      distances: The distances from A to B (e.g. `distances_gt_to_pred`).
      surfel_areas: The surfel areas for A (e.g. `surfel_areas_gt`).

    Returns:
      A tuple of the sorted (distances, surfel_areas).
    """
    sorted_surfels = np.array(sorted(zip(distances, surfel_areas)))
    return sorted_surfels[:, 0], sorted_surfels[:, 1]


def compute_surface_distances(mask_gt,
                              mask_pred,
                              spacing_mm):
    """Computes closest distances from all surface points to the other surface.

    This function can be applied to 2D or 3D tensors. For 2D, both masks must be
    2D and `spacing_mm` must be a 2-element list. For 3D, both masks must be 3D
    and `spacing_mm` must be a 3-element list. The description is done for the 2D
    case, and the formulation for the 3D case is present is parenthesis,
    introduced by "resp.".

    Finds all contour elements (resp surface elements "surfels" in 3D) in the
    ground truth mask `mask_gt` and the predicted mask `mask_pred`, computes their
    length in mm (resp. area in mm^2) and the distance to the closest point on the
    other contour (resp. surface). It returns two sorted lists of distances
    together with the corresponding contour lengths (resp. surfel areas). If one
    of the masks is empty, the corresponding lists are empty and all distances in
    the other list are `inf`.

    Args:
      mask_gt: 2-dim (resp. 3-dim) bool Numpy array. The ground truth mask.
      mask_pred: 2-dim (resp. 3-dim) bool Numpy array. The predicted mask.
      spacing_mm: 2-element (resp. 3-element) list-like structure. Voxel spacing
        in x0 anx x1 (resp. x0, x1 and x2) directions.

    Returns:
      A dict with:
      "distances_gt_to_pred": 1-dim numpy array of type float. The distances in mm
          from all ground truth surface elements to the predicted surface,
          sorted from smallest to largest.
      "distances_pred_to_gt": 1-dim numpy array of type float. The distances in mm
          from all predicted surface elements to the ground truth surface,
          sorted from smallest to largest.
      "surfel_areas_gt": 1-dim numpy array of type float. The length of the
        of the ground truth contours in mm (resp. the surface elements area in
        mm^2) in the same order as distances_gt_to_pred.
      "surfel_areas_pred": 1-dim numpy array of type float. The length of the
        of the predicted contours in mm (resp. the surface elements area in
        mm^2) in the same order as distances_gt_to_pred.

    Raises:
      ValueError: If the masks and the `spacing_mm` arguments are of incompatible
        shape or type. Or if the masks are not 2D or 3D.
    """
    # The terms used in this function are for the 3D case. In particular, surface
    # in 2D stands for contours in 3D. The surface elements in 3D correspond to
    # the line elements in 2D.

    _assert_is_bool_numpy_array("mask_gt", mask_gt)
    _assert_is_bool_numpy_array("mask_pred", mask_pred)

    if not len(mask_gt.shape) == len(mask_pred.shape) == len(spacing_mm):
        raise ValueError("The arguments must be of compatible shape. Got mask_gt "
                         "with {} dimensions ({}) and mask_pred with {} dimensions "
                         "({}), while the spacing_mm was {} elements.".format(
            len(mask_gt.shape),
            mask_gt.shape, len(mask_pred.shape), mask_pred.shape,
            len(spacing_mm)))

    num_dims = len(spacing_mm)
    if num_dims == 2:
        _check_2d_numpy_array("mask_gt", mask_gt)
        _check_2d_numpy_array("mask_pred", mask_pred)

        # compute the area for all 16 possible surface elements
        # (given a 2x2 neighbourhood) according to the spacing_mm
        neighbour_code_to_surface_area = (
            lookup_tables.create_table_neighbour_code_to_contour_length(spacing_mm))
        kernel = lookup_tables.ENCODE_NEIGHBOURHOOD_2D_KERNEL
        full_true_neighbours = 0b1111
    elif num_dims == 3:
        _check_3d_numpy_array("mask_gt", mask_gt)
        _check_3d_numpy_array("mask_pred", mask_pred)

        # compute the area for all 256 possible surface elements
        # (given a 2x2x2 neighbourhood) according to the spacing_mm
        neighbour_code_to_surface_area = (
            lookup_tables.create_table_neighbour_code_to_surface_area(spacing_mm))
        kernel = lookup_tables.ENCODE_NEIGHBOURHOOD_3D_KERNEL
        full_true_neighbours = 0b11111111
    else:
        raise ValueError("Only 2D and 3D masks are supported, not "
                         "{}D.".format(num_dims))

    # compute the bounding box of the masks to trim the volume to the smallest
    # possible processing subvolume
    bbox_min, bbox_max = _compute_bounding_box(mask_gt | mask_pred)
    # Both the min/max bbox are None at the same time, so we only check one.
    if bbox_min is None:
        return {
            "distances_gt_to_pred": np.array([]),
            "distances_pred_to_gt": np.array([]),
            "surfel_areas_gt": np.array([]),
            "surfel_areas_pred": np.array([]),
        }

    # crop the processing subvolume.
    cropmask_gt = _crop_to_bounding_box(mask_gt, bbox_min, bbox_max)
    cropmask_pred = _crop_to_bounding_box(mask_pred, bbox_min, bbox_max)

    # compute the neighbour code (local binary pattern) for each voxel
    # the resulting arrays are spacially shifted by minus half a voxel in each
    # axis.
    # i.e. the points are located at the corners of the original voxels
    neighbour_code_map_gt = ndimage.filters.correlate(
        cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbour_code_map_pred = ndimage.filters.correlate(
        cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0)

    # create masks with the surface voxels
    borders_gt = ((neighbour_code_map_gt != 0) &
                  (neighbour_code_map_gt != full_true_neighbours))
    borders_pred = ((neighbour_code_map_pred != 0) &
                    (neighbour_code_map_pred != full_true_neighbours))

    # compute the distance transform (closest distance of each voxel to the
    # surface voxels)
    if borders_gt.any():
        distmap_gt = ndimage.morphology.distance_transform_edt(
            ~borders_gt, sampling=spacing_mm)
    else:
        distmap_gt = np.Inf * np.ones(borders_gt.shape)

    if borders_pred.any():
        distmap_pred = ndimage.morphology.distance_transform_edt(
            ~borders_pred, sampling=spacing_mm)
    else:
        distmap_pred = np.Inf * np.ones(borders_pred.shape)

    # compute the area of each surface element
    surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
    surface_area_map_pred = neighbour_code_to_surface_area[
        neighbour_code_map_pred]

    # create a list of all surface elements with distance and area
    distances_gt_to_pred = distmap_pred[borders_gt]
    distances_pred_to_gt = distmap_gt[borders_pred]
    surfel_areas_gt = surface_area_map_gt[borders_gt]
    surfel_areas_pred = surface_area_map_pred[borders_pred]

    # sort them by distance
    if distances_gt_to_pred.shape != (0,):
        distances_gt_to_pred, surfel_areas_gt = _sort_distances_surfels(
            distances_gt_to_pred, surfel_areas_gt)

    if distances_pred_to_gt.shape != (0,):
        distances_pred_to_gt, surfel_areas_pred = _sort_distances_surfels(
            distances_pred_to_gt, surfel_areas_pred)

    return {
        "distances_gt_to_pred": distances_gt_to_pred,
        "distances_pred_to_gt": distances_pred_to_gt,
        "surfel_areas_gt": surfel_areas_gt,
        "surfel_areas_pred": surfel_areas_pred,
    }


def compute_average_surface_distance(surface_distances):
    """Returns the average surface distance.

    Computes the average surface distances by correctly taking the area of each
    surface element into account. Call compute_surface_distances(...) before, to
    obtain the `surface_distances` dict.

    Args:
      surface_distances: dict with "distances_gt_to_pred", "distances_pred_to_gt"
      "surfel_areas_gt", "surfel_areas_pred" created by
      compute_surface_distances()

    Returns:
      A tuple with two float values:
        - the average distance (in mm) from the ground truth surface to the
          predicted surface
        - the average distance from the predicted surface to the ground truth
          surface.
    """
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    average_distance_gt_to_pred = (
            np.sum(distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt))
    average_distance_pred_to_gt = (
            np.sum(distances_pred_to_gt * surfel_areas_pred) /
            np.sum(surfel_areas_pred))
    return (average_distance_gt_to_pred, average_distance_pred_to_gt)


def compute_robust_hausdorff(surface_distances, percent):
    """Computes the robust Hausdorff distance.

    Computes the robust Hausdorff distance. "Robust", because it uses the
    `percent` percentile of the distances instead of the maximum distance. The
    percentage is computed by correctly taking the area of each surface element
    into account.

    Args:
      surface_distances: dict with "distances_gt_to_pred", "distances_pred_to_gt"
        "surfel_areas_gt", "surfel_areas_pred" created by
        compute_surface_distances()
      percent: a float value between 0 and 100.

    Returns:
      a float value. The robust Hausdorff distance in mm.
    """
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    if len(distances_gt_to_pred) > 0:  # pylint: disable=g-explicit-length-test
        surfel_areas_cum_gt = np.cumsum(surfel_areas_gt) / np.sum(surfel_areas_gt)
        idx = np.searchsorted(surfel_areas_cum_gt, percent / 100.0)
        perc_distance_gt_to_pred = distances_gt_to_pred[
            min(idx, len(distances_gt_to_pred) - 1)]
    else:
        perc_distance_gt_to_pred = np.Inf

    if len(distances_pred_to_gt) > 0:  # pylint: disable=g-explicit-length-test
        surfel_areas_cum_pred = (np.cumsum(surfel_areas_pred) /
                                 np.sum(surfel_areas_pred))
        idx = np.searchsorted(surfel_areas_cum_pred, percent / 100.0)
        perc_distance_pred_to_gt = distances_pred_to_gt[
            min(idx, len(distances_pred_to_gt) - 1)]
    else:
        perc_distance_pred_to_gt = np.Inf

    return max(perc_distance_gt_to_pred, perc_distance_pred_to_gt)


def compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm):
    """Computes the overlap of the surfaces at a specified tolerance.

    Computes the overlap of the ground truth surface with the predicted surface
    and vice versa allowing a specified tolerance (maximum surface-to-surface
    distance that is regarded as overlapping). The overlapping fraction is
    computed by correctly taking the area of each surface element into account.

    Args:
      surface_distances: dict with "distances_gt_to_pred", "distances_pred_to_gt"
        "surfel_areas_gt", "surfel_areas_pred" created by
        compute_surface_distances()
      tolerance_mm: a float value. The tolerance in mm

    Returns:
      A tuple of two float values. The overlap fraction in [0.0, 1.0] of the
      ground truth surface with the predicted surface and vice versa.
    """
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    rel_overlap_gt = (
            np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm]) /
            np.sum(surfel_areas_gt))
    rel_overlap_pred = (
            np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm]) /
            np.sum(surfel_areas_pred))
    return (rel_overlap_gt, rel_overlap_pred)


def compute_surface_dice_at_tolerance(surface_distances, tolerance_mm):
    """Computes the _surface_ DICE coefficient at a specified tolerance.

    Computes the _surface_ DICE coefficient at a specified tolerance. Not to be
    confused with the standard _volumetric_ DICE coefficient. The surface DICE
    measures the overlap of two surfaces instead of two volumes. A surface
    element is counted as overlapping (or touching), when the closest distance to
    the other surface is less or equal to the specified tolerance. The DICE
    coefficient is in the range between 0.0 (no overlap) to 1.0 (perfect overlap).

    Args:
      surface_distances: dict with "distances_gt_to_pred", "distances_pred_to_gt"
        "surfel_areas_gt", "surfel_areas_pred" created by
        compute_surface_distances()
      tolerance_mm: a float value. The tolerance in mm

    Returns:
      A float value. The surface DICE coefficient in [0.0, 1.0].
    """
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    overlap_gt = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
    overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
    surface_dice = (overlap_gt + overlap_pred) / (
            np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))
    return surface_dice
