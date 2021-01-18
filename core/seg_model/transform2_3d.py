import tensorflow as tf
import glob
from PIL import Image
import cv2
import numpy as np



def mgrid(*args, **kwargs):
    """
    create orthogonal grid
    similar to np.mgrid
    Parameters
    ----------
    args : int
        number of points on each axis
    low : float
        minimum coordinate value
    high : float
        maximum coordinate value
    Returns
    -------
    grid : tf.Tensor [len(args), args[0], ...]
        orthogonal grid
    """
    low = kwargs.pop("low", -1)
    high = kwargs.pop("high", 1)
    low = tf.to_float(low)
    high = tf.to_float(high)
    coords = (tf.linspace(low, high, arg) for arg in args)
    # 生成meshgrid
    grid = tf.stack(tf.meshgrid(*coords, indexing='ij'))
    return grid


def batch_warp3d(imgs, mappings):
    """
    warp image using mapping function
    I(x) -> I(phi(x))
    phi: mapping function
    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, zlen, n_channel]
    mapping : tf.Tensor
        grids representing mapping function
        [n_batch, xlen, ylen, zlen, 3]
    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, zlen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    coords = tf.reshape(mappings, [n_batch, 3, -1])
    x_coords = tf.slice(coords, [0, 0, 0], [-1, 1, -1])
    y_coords = tf.slice(coords, [0, 1, 0], [-1, 1, -1])
    z_coords = tf.slice(coords, [0, 2, 0], [-1, 1, -1])
    x_coords_flat = tf.reshape(x_coords, [-1])
    y_coords_flat = tf.reshape(y_coords, [-1])
    z_coords_flat = tf.reshape(z_coords, [-1])

    output = _interpolate3d(imgs, x_coords_flat, y_coords_flat, z_coords_flat)

    return output


def _repeat(base_indices, n_repeats):
    #  _repeat(tf.range(n_batch) * xlen * ylen, ylen * xlen)
    # 这里只是一些值，代表batch，用于后面的索引啦
    # (N, 1) * (1, n_repeats) = (N, n_repeat), and flatten to 1-D
    base_indices = tf.matmul(
        tf.reshape(base_indices, [-1, 1]),
        tf.ones([1, n_repeats], dtype='int32'))
    return tf.reshape(base_indices, [-1])  # flatten to 1D


def _interpolate3d(imgs, x, y, z):
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    zlen = tf.shape(imgs)[3]
    n_channel = tf.shape(imgs)[4]

    x = tf.to_float(x)
    y = tf.to_float(y)
    z = tf.to_float(z)
    xlen_f = tf.to_float(xlen)
    ylen_f = tf.to_float(ylen)
    zlen_f = tf.to_float(zlen)
    zero = tf.zeros([], dtype='int32')
    max_x = tf.cast(xlen - 1, 'int32')
    max_y = tf.cast(ylen - 1, 'int32')
    max_z = tf.cast(zlen - 1, 'int32')

    # scale indices from [-1, 1] to [0, xlen/ylen/zlen]
    x = (x + 1.) * (xlen_f - 1.) * 0.5
    y = (y + 1.) * (ylen_f - 1.) * 0.5
    z = (z + 1.) * (zlen_f - 1.) * 0.5

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    z0 = tf.cast(tf.floor(z), 'int32')
    z1 = z0 + 1

    # 做个clip, 所有值都落在 0和 max_x
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    z0 = tf.clip_by_value(z0, zero, max_z)
    z1 = tf.clip_by_value(z1, zero, max_z)

    # what is the use?
    # (N, 1) * (1, n_repeats) = (N, n_repeat), and flatten to 1-D
    base = _repeat(tf.range(n_batch) * xlen * ylen * zlen,
                   xlen * ylen * zlen)  # tf.range(3: [0, 1, 2]
    base_x0 = base + x0 * ylen * zlen
    base_x1 = base + x1 * ylen * zlen
    base00 = base_x0 + y0 * zlen
    base01 = base_x0 + y1 * zlen
    base10 = base_x1 + y0 * zlen
    base11 = base_x1 + y1 * zlen
    index000 = base00 + z0
    index001 = base00 + z1
    index010 = base01 + z0
    index011 = base01 + z1
    index100 = base10 + z0
    index101 = base10 + z1
    index110 = base11 + z0
    index111 = base11 + z1

    # use indices to lookup pixels in the flat image and restore
    # n_channel dim
    imgs_flat = tf.reshape(imgs, [-1, n_channel])
    imgs_flat = tf.to_float(imgs_flat)
    I000 = tf.gather(imgs_flat, index000)
    I001 = tf.gather(imgs_flat, index001)
    I010 = tf.gather(imgs_flat, index010)
    I011 = tf.gather(imgs_flat, index011)
    I100 = tf.gather(imgs_flat, index100)
    I101 = tf.gather(imgs_flat, index101)
    I110 = tf.gather(imgs_flat, index110)
    I111 = tf.gather(imgs_flat, index111)

    # and finally calculate interpolated values
    dx = x - tf.to_float(x0)
    dy = y - tf.to_float(y0)
    dz = z - tf.to_float(z0)
    w000 = tf.expand_dims((1. - dx) * (1. - dy) * (1. - dz), 1)
    w001 = tf.expand_dims((1. - dx) * (1. - dy) * dz, 1)
    w010 = tf.expand_dims((1. - dx) * dy * (1. - dz), 1)
    w011 = tf.expand_dims((1. - dx) * dy * dz, 1)
    w100 = tf.expand_dims(dx * (1. - dy) * (1. - dz), 1)
    w101 = tf.expand_dims(dx * (1. - dy) * dz, 1)
    w110 = tf.expand_dims(dx * dy * (1. - dz), 1)
    w111 = tf.expand_dims(dx * dy * dz, 1)
    output = tf.add_n([w000 * I000, w001 * I001, w010 * I010, w011 * I011,
                       w100 * I100, w101 * I101, w110 * I110, w111 * I111])

    # reshape
    output = tf.reshape(output, [n_batch, xlen, ylen, zlen, n_channel])

    return output


def batch_mgrid(n_batch, *args, **kwargs):
    """
    create batch of orthogonal grids
    similar to np.mgrid
    Parameters
    ----------
    n_batch : int
        number of grids to create
    args : int
        number of points on each axis
    low : float
        minimum coordinate value
    high : float
        maximum coordinate value
    Returns
    -------
    grids : tf.Tensor [n_batch, len(args), args[0], ...]
        batch of orthogonal grids
    """
    grid = mgrid(*args, **kwargs)
    grid = tf.expand_dims(grid, 0)
    grids = tf.tile(grid, [n_batch] + [1 for _ in range(len(args) + 1)])
    return grids


def batch_affine_warp2d(imgs, theta):
    """
    affine transforms 2d images
    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, n_channel]
    theta : tf.Tensor
        parameters of affine transformation
        [n_batch, 6]
    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    theta = tf.reshape(theta, [-1, 2, 3])
    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 2])
    t = tf.slice(theta, [0, 0, 2], [-1, -1, -1])

    grids = batch_mgrid(n_batch, xlen, ylen)
    coords = tf.reshape(grids, [n_batch, 2, -1])

    T_g = tf.batch_matmul(matrix, coords) + t
    T_g = tf.reshape(T_g, [n_batch, 2, xlen, ylen])
    output = batch_warp2d(imgs, T_g)
    return output


def batch_affine_warp3d(imgs, theta):
    """
    affine transforms 3d images
    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, zlen, n_channel]
    theta : tf.Tensor
        parameters of affine transformation
        [n_batch, 12]
    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, zlen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    zlen = tf.shape(imgs)[3]
    theta = tf.reshape(theta, [-1, 3, 4])
    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 3])
    t = tf.slice(theta, [0, 0, 3], [-1, -1, -1])

    grids = batch_mgrid(n_batch, xlen, ylen, zlen)
    grids = tf.reshape(grids, [n_batch, 3, -1])
    # 生成mapping grid
    T_g = tf.matmul(matrix, grids) + t
    T_g = tf.reshape(T_g, [n_batch, 3, xlen, ylen, zlen])

    # 主要是sampling 操作
    output = batch_warp3d(imgs, T_g)
    return output


def read_image():
    f = glob.iglob(r'*.jpg')
    img = []
    # f is a generator
    for i in f:
        img.append(np.array(Image.open(i)))
        # (5, 28, 28)
    return img


if __name__ == '__main__':
    sess = tf.Session()
    imgg = read_image()
    d, h, w, c = np.shape(imgg)
    imgg = np.transpose(imgg, [1, 2, 0, 3])
    x_np = np.expand_dims(imgg, axis=0)
    x_tensor = tf.convert_to_tensor(x_np)

    identity_matrix = tf.to_float([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.1, 0]])
    identity_warped = batch_affine_warp3d(x_tensor, identity_matrix)




    img = identity_warped[0, :, :, :, :].eval(session=sess)
    print(np.shape(img))
    img0 = img[:, :, 2, :].astype("uint8")
    print(np.shape(img0))
    img0 = Image.fromarray(img0, 'RGB')
    img0.show()
    # with tf.Session() as sess:
    #
    #  print sess.run(identity_warped[0, :, :, :, 0].eval())
    # A = np.array([[1, 1, 2, 4], [3, 4, 8, 5]])
    #
    #
    # print(sess.run(tf.clip_by_value(A, 2, 5)))