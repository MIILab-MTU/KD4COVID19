import numpy as np
import matplotlib.pyplot as plt

from skimage import measure, morphology, segmentation
from sklearn.cluster import KMeans
from skimage import exposure


def segment_morphology(images,
                       binary_threshold=-400,
                       erosion_kernel=(2, 2, 2),
                       dilation_kernel=(4, 4, 4),
                       closing_kearnel=(4, 4, 4)):
    masks = images.copy()
    masks = masks < binary_threshold
    # clear border
    for c in range(masks.shape[0]):
        masks[c, :, :] = segmentation.clear_border(masks[c, :, :])

    # keep 2 largest connected graph
    labels = measure.label(masks)
    regions = measure.regionprops(labels)
    labels = [(r.area, r.label) for r in regions]

    if len(labels) > 2:
        labels.sort(reverse=True)
        max_area = labels[2][0]
        for r in regions:
            if r.area < max_area:
                for c in r.coords:
                    masks[c[0], c[1], c[2]] = 0

    # erosion
    masks = morphology.erosion(masks, selem=np.ones(erosion_kernel))
    # closing
    masks = morphology.closing(masks, selem=np.ones(closing_kearnel))
    # dilation
    masks = morphology.dilation(masks, selem=np.ones(dilation_kernel))
    return masks

def segment_morphology2(images, volume_ratio=0.005,
                       binary_threshold=-400,
                       erosion_kernel=(2, 2, 2),
                       dilation_kernel=(4, 4, 4),
                       closing_kearnel=(4, 4, 4)):
    masks = images.copy()
    masks = masks < binary_threshold
    # clear border
    for c in range(masks.shape[0]):
        masks[c, :, :] = segmentation.clear_border(masks[c, :, :])

    # keep 2 largest connected graph
    labels = measure.label(masks)
    regions = measure.regionprops(labels)
    labels = [(r.area, r.label) for r in regions]

    if len(labels) > 2:
        for r in regions:
            if r.area < np.prod(images.shape) * volume_ratio:
                for c in r.coords:
                    masks[c[0], c[1], c[2]] = 0

    # erosion
    masks = morphology.erosion(masks, selem=np.ones(erosion_kernel))
    # closing
    masks = morphology.closing(masks, selem=np.ones(closing_kearnel))
    # dilation
    masks = morphology.dilation(masks, selem=np.ones(dilation_kernel))
    return masks