import tensorflow as tf


def spatial_transformer_network_3d(input_fmap, theta, out_dims=None, **kwargs):
    """
    Spatial Transformer Network layer implementation as described in [1].
    The layer is composed of 3 elements:
    - localization_net: takes the original images as input and outputs
      the parameters of the affine transformation that should be applied
      to the input images.
    - affine_grid_generator: generates a grid of (x,y,z) coordinates that
      correspond to a set of points where the input should be sampled
      to produce the transformed output.
    - bilinear_sampler: takes as input the original image and the 3D grid
      and produces the output 3D images using bilinear interpolation.
    Input
    -----
    - input_fmap: output of the previous layer. Can be input if spatial
      transformer layer is at the beginning of architecture. Should be
      a tensor of shape (B, H, W, D, C).
    - theta: affine transform tensor of shape (B, 12). Permits cropping,
      translation and isotropic scaling. Initialize to identity matrix.
      It is the output of the localization network.
    Returns
    -------
    - out_fmap: transformed input feature map. Tensor of size (B, H, W, D, C).
    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
         (https://arxiv.org/abs/1506.02025)
    """
    # grab input dimensions
    B = tf.shape(input_fmap)[0]
    H = tf.shape(input_fmap)[1]
    W = tf.shape(input_fmap)[2]
    D = tf.shape(input_fmap)[3]

    # reshape theta to (B, 3, 4)
    theta = tf.reshape(theta, [B, 3, 4])

    # generate grids of same size or upsample/downsample if specified
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        out_D = out_dims[2]
        batch_grids = affine_grid_generator_3d(out_H, out_W, out_D, theta)
    else:
        batch_grids = affine_grid_generator_3d(H, W, D, theta)

    # 已经是source grid了， 得到的值是 [-1, 1]
    x_s = batch_grids[:, 0, :, :, :]
    y_s = batch_grids[:, 1, :, :, :]
    z_s = batch_grids[:, 2, :, :, :]

    # sample input with grid to get output
    out_fmap = bilinear_sampler_3d(input_fmap, x_s, y_s, z_s)

    return out_fmap


def get_pixel_value_3d(img, x, y, z):
    """
    Utility function to get pixel value for coordinate vectors x, y and z from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, D，C)
    Comment (JChen):
    I think the shape of x, y is non-flattened, and thus (B, H, W, D)
    即： 每个x上点的值都代表在目标像素x coor上的索引值！
    终于tm明白了！
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)  # ????  it should be tf.shape(img)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth = shape[3]

    # 这边代表了_repeat过程！
    batch_idx = tf.range(0, batch_size)  # 数字序列
    # (batch_size, 1, 1)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1))  # [[[[0]]],[[1]],[[2]] ...]
    # (batch_size, height, width, depth) , 就是每个点的值都代表在batch上的索引！！！
    b = tf.tile(batch_idx, (1, height,
                            width,
                            depth))  # [[[0, 0 .... width个0], [0, 0 .... width个0], .... height 个 [0, 0 .... width个0],[[1,..],..],[[2,..],..] ...]
    # b: (B, H, W, D), y: (B,H,W,D), x: (B,H,W,D)
    # indices: (B, H, W, D) -> (B, H, W, D, 4), 且第4维度的"值"是代表在batch, height, width, depth上进行索引。
    indices = tf.stack([b, y, x, z], 4)

    # 多维索引， 取值； indices 上 每个
    # gather_nd( (B, H, W, D, C), 多维索引！
    gather =  tf.gather_nd(img, indices)
    gather = tf.cast(gather, 'float32')
    return gather


def affine_grid_generator_3d(height, width, depth, theta):
    """
    This function returns a sampling grid, which when used with the bilinear sampler on the input feature
    map, will create an output feature map that is an affine transformation [1] of the input feature map.
    Input
    -----
    - height: desired height of grid/output. Used to downsample or upsample.
    - width: desired width of grid/output. Used to downsample or upsample.
    - depth: desired width of grid/output. Used to downsample or upsample.
    - theta: affine transform matrices of shape (num_batch, 3, 4).
      For each image in the batch, we have 12 theta parameters of
      the form (3x4) that define the affine transformation T.
    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 3, H, W, D).
      The 2nd dimension has 3 components: (x, y, z) which are the
      sampling points of the original image for each point in the
      target image.
    Note
    ----
    [1]: the affine transformation allows cropping, translation, and isotropic scaling.
    """
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid, target grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    z = tf.linspace(-1.0, 1.0, depth)
    x_t, y_t, z_t = tf.meshgrid(x, y, z)

    # flatten to 1-D
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])
    z_t_flat = tf.reshape(z_t, [-1])

    # reshape to [x_t, y_t, z_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, z_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)  # (1, 4, h*w*d)
    print(sampling_grid)
    # 第0维度复制num_batch次，第1维度复制1次，第2维度复制1次
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch_grid has shape (num_batch, 3, H*W*D)

    # reshape to (num_batch, 3, H, W, D)
    batch_grids = tf.reshape(batch_grids, [num_batch, 3, height, width, depth])

    return batch_grids


def bilinear_sampler_3d(img, x, y, z):
    """
    Performs bilinear sampling of the input images according to the normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input. To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y, z which is the output of affine_grid_generator.
            x, y, z should have the most values of [-1, 1]
            x_s, y_s, z : (B, H, W, D)
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
            Tensor of size (B, H, W, D, C)
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    D = tf.shape(img)[3]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    max_z = tf.cast(D - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale the value of x, y and z to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    z = tf.cast(z, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y, 'float32'))
    z = 0.5 * ((z + 1.0) * tf.cast(max_z, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i) and 2 nearest z-plane, make of 8 nearest corner points
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    z0 = tf.cast(tf.floor(z), 'int32')
    z1 = z0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    z0 = tf.clip_by_value(z0, zero, max_z)
    z1 = tf.clip_by_value(z1, zero, max_z)

    # get pixel value at corner coords Ia, ...
    # (B, H, W, D, 3), 这边的值要为整数，因为代表了下标。
    I000 = get_pixel_value_3d(img, x0, y0, z0)
    I001 = get_pixel_value_3d(img, x0, y0, z1)
    I010 = get_pixel_value_3d(img, x0, y1, z0)
    I011 = get_pixel_value_3d(img, x0, y1, z1)
    I100 = get_pixel_value_3d(img, x1, y0, z0)
    I101 = get_pixel_value_3d(img, x1, y0, z1)
    I110 = get_pixel_value_3d(img, x1, y1, z0)
    I111 = get_pixel_value_3d(img, x1, y1, z1)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    z0 = tf.cast(z0, 'float32')
    z1 = tf.cast(z1, 'float32')
    #
    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    we = ()
    #
    #
    # # add dimension for addition
    # wa = tf.expand_dims(wa, axis=3)  # (B,H,W,1)
    # wb = tf.expand_dims(wb, axis=3)
    # wc = tf.expand_dims(wc, axis=3)
    # wd = tf.expand_dims(wd, axis=3)

    dx = x - tf.to_float(x0)
    dy = y - tf.to_float(y0)
    dz = z - tf.to_float(z0)
    # (B, H, W, D, 1)
    w000 = tf.expand_dims((1. - dx) * (1. - dy) * (1. - dz), axis=4)
    w001 = tf.expand_dims((1. - dx) * (1. - dy) * dz, axis=4)
    w010 = tf.expand_dims((1. - dx) * dy * (1. - dz), axis=4)
    w011 = tf.expand_dims((1. - dx) * dy * dz, axis=4)
    w100 = tf.expand_dims(dx * (1. - dy) * (1. - dz), axis=4)
    w101 = tf.expand_dims(dx * (1. - dy) * dz, axis=4)
    w110 = tf.expand_dims(dx * dy * (1. - dz), axis=4)
    w111 = tf.expand_dims(dx * dy * dz, axis=4)

    # w000 = tf.cast(w000, 'float32')
    # w001 = tf.cast(w001, 'float32')
    # w010 = tf.cast(w010, 'float32')
    # w011 = tf.cast(w011, 'float32')
    # w100 = tf.cast(w100, 'float32')
    # w101 = tf.cast(w101, 'float32')
    # w110 = tf.cast(w110, 'float32')
    # w111 = tf.cast(w111, 'float32')

    # compute output, 一个列表元素相加, 距离加权对应位置像素，得到最后结果，(B, H, W, D, C)
    out = tf.add_n([w000 * I000, w001 * I001, w010 * I010, w011 * I011, w100 * I100, w101 * I101, w110 * I110, w111 * I111])

    return out


def read_image():
    img = np.random.uniform(size=(28,28,28))
    # f is a generator
    # for i in f:
    #     img.append(np.array(Image.open(i)))
    #     # (5, 28, 28, 1)
    #if np.shape(img[3])==None:
    # img = tf.expand_dims(img, -1)
    print(np.shape(img))
    return img

if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    import cv2
    import numpy as np
    import glob

    ########### 取消注释后，是验证2D
    # I = tf.to_float(np.array(Image.open('0_0.bmp')).reshape(1, 28, 28, 1))
    # # img = tf.to_float(np.arange(25).reshape(1, 5, 5, 1))
    # identity_matrix = tf.to_float([0.5, 0, 0, 0, 0.5, 0])
    # zoom_in_matrix = identity_matrix * 0.5
    # identity_warped = spatial_transformer_network(I, identity_matrix)
    # sess = tf.Session()
    # identity_warped = identity_warped.eval(session=sess)
    # img0 = np.reshape(identity_warped, (28, 28))
    # img0 = Image.fromarray(img0)
    #
    # img0.show()
    #
    # # zoom_in_warped = batch_affine_warp2d(img, zoom_in_matrix)
    # # with tf.Session() as sess:
    # #     print sess.run(img[0, :, :, 0])
    # #     print sess.run(identity_warped[0, :, :, 0])
    # #     print sess.run(zoom_in_warped[0, :, :, 0])

    ######### 3D 验证 -- MNIST
    sess = tf.Session()
    imgg = read_image()
    d, h, w = np.shape(imgg)
    imgg = np.transpose(imgg, [1, 2, 0])

    imgg = np.expand_dims(imgg, axis=0)
    imgg = np.expand_dims(imgg, axis=4)

    # RGB: x_np = np.reshape(imgg, (1, w, h, d, 3)).astype("float")
    # MNIST: (1, w, h, d, 1)
    x_np = imgg.astype("float")

    x_tensor = tf.convert_to_tensor(x_np)
    print(x_tensor)

    identity_matrix = tf.cast([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]], "float32")
    identity_warped = spatial_transformer_network_3d(x_tensor, identity_matrix)
    print("----------")
    print(identity_warped)
    img = identity_warped[0, :, :, 3, 0].eval(session=sess)
    # print(np.shape(img))

    img0 = Image.fromarray(img)
    img0.show()