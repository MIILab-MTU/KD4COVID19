import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers.python.layers import regularizers


def conv_block_3d(inputs, n_filters, l2_scale, kernel_size=[3, 3, 3]):
	"""
	Builds the conv block for MobileNets
	Apply successivly a 3D convolution + BatchNormalization + ReLu activation
	"""
	# Skip pointwise by setting num_outputs=Non
	net = slim.conv3d(inputs, n_filters,
					  kernel_size=kernel_size,
					  activation_fn=None,
					  weights_regularizer=regularizers.l2_regularizer(scale=l2_scale))
	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	return net


def conv_transpose_block_3d(inputs, n_filters, l2_scale, kernel_size=[3, 3, 3], stride=[2,2,2]):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	"""
	net = slim.conv3d_transpose(inputs, n_filters,
								kernel_size=kernel_size,
								stride=stride,
								activation_fn=None,
								weights_regularizer=regularizers.l2_regularizer(scale=l2_scale))
	net = tf.nn.relu(slim.batch_norm(net))
	return net


class Resnet4CLF(object):
	def __init__(self, inputs, n_filter, l2):
		self.inputs = inputs
		self.n_filter = n_filter
		self.l2 = l2
	
	def create_model(self):
		net = conv_block_3d(self.inputs, self.n_filter, self.l2, kernel_size=[3, 3, 3])
		net = slim.max_pool3d(net, [2, 2, 2], stride=2)

		net = conv_block_3d(net, self.n_filter*2, self.l2, kernel_size=[3, 3, 3])
		net = slim.max_pool3d(net, [2, 2, 2], stride=2)

		net = conv_block_3d(net, self.n_filter*4, self.l2, kernel_size=[3, 3, 3])
		net = slim.max_pool3d(net, [2, 2, 2], stride=2)

		net = conv_block_3d(net, self.n_filter*8, self.l2, kernel_size=[3, 3, 3])
		net = slim.max_pool3d(net, [2, 2, 2], stride=2)

		net = slim.flatten(net)
		fc = slim.fully_connected(net, 12)
		return fc


class STN(object):

	def __init__(self, inputs, n_filter, l2):
		self.inputs = inputs
		self.n_filter = n_filter
		self.l2 = l2
	
	def create_model(self):
		net = conv_block_3d(self.inputs, self.n_filter, self.l2, kernel_size=[3, 3, 3])
		net = slim.max_pool3d(net, [2, 2, 2], stride=2)

		net = conv_block_3d(net, self.n_filter*2, self.l2, kernel_size=[3, 3, 3])
		net = slim.max_pool3d(net, [2, 2, 2], stride=2)

		net = conv_block_3d(net, self.n_filter*4, self.l2, kernel_size=[3, 3, 3])
		net = slim.max_pool3d(net, [2, 2, 2], stride=2)

		net = conv_block_3d(net, self.n_filter*8, self.l2, kernel_size=[3, 3, 3])
		net = slim.max_pool3d(net, [2, 2, 2], stride=2)

		net = slim.flatten(net)
		fc = slim.fully_connected(net, 12)
		return fc


class VNet(object):

	def __init__(self, n_filter, inputs, output_channel, dropout_tensor, l2):
		self.n_filter = n_filter
		self.inputs = inputs
		self.l2 = l2
		self.output_channel = output_channel
		self.dropout_tensor = dropout_tensor

	def create_model(self):
		net = conv_block_3d(self.inputs, self.n_filter, self.l2, kernel_size=[3, 3, 3])
		net = conv_block_3d(net, self.n_filter, self.l2, kernel_size=[3, 3, 3])
		net = slim.max_pool3d(net, [2, 2, 2], stride=2)
		skip_1 = net
		print("skip_1, shape = {}".format(skip_1.get_shape()))

		net = conv_block_3d(net, self.n_filter * 2, self.l2, kernel_size=[3, 3, 3])
		net = conv_block_3d(net, self.n_filter * 2, self.l2, kernel_size=[3, 3, 3])
		net = slim.avg_pool3d(net, [2, 2, 2], stride=2)
		skip_2 = net
		print("skip_2, shape = {}".format(skip_2.get_shape()))

		net = conv_block_3d(net, self.n_filter * 4, self.l2, kernel_size=[3, 3, 3])
		net = conv_block_3d(net, self.n_filter * 4, self.l2, kernel_size=[3, 3, 3])
		net = conv_block_3d(net, self.n_filter * 4, self.l2, kernel_size=[3, 3, 3])
		net = slim.avg_pool3d(net, [2, 2, 2], stride=2)
		skip_3 = net
		print("skip_3, shape = {}".format(skip_3.get_shape()))

		# deepest convolutional layers
		net = conv_block_3d(net, self.n_filter * 8, self.l2, kernel_size=[3, 3, 3])
		net = conv_block_3d(net, self.n_filter * 8, self.l2, kernel_size=[3, 3, 3])
		net = conv_block_3d(net, self.n_filter * 8, self.l2, kernel_size=[3, 3, 3])

		net = tf.concat([net, skip_3], axis=4)
		print("concatenate_1, shape = {}".format(net.get_shape()))

		net = conv_transpose_block_3d(net, self.n_filter * 4, self.l2, kernel_size=[3, 3, 3])
		net = conv_block_3d(net, self.n_filter * 4, self.l2, kernel_size=[3, 3, 3])
		net = conv_block_3d(net, self.n_filter * 4, self.l2, kernel_size=[3, 3, 3])
		net = conv_block_3d(net, self.n_filter * 2, self.l2, kernel_size=[3, 3, 3])

		net = tf.concat([net, skip_2], axis=4)
		print("concatenate_2, shape = {}".format(net.get_shape()))

		net = conv_transpose_block_3d(net, self.n_filter * 2, self.l2, kernel_size=[3, 3, 3])
		net = conv_block_3d(net, self.n_filter * 2, self.l2, kernel_size=[3, 3, 3])
		net = conv_block_3d(net, self.n_filter, self.l2, kernel_size=[3, 3, 3])

		net = tf.concat([net, skip_1], axis=4)
		print("concatenate_3, shape = {}".format(net.get_shape()))

		net = conv_transpose_block_3d(net, self.n_filter, self.l2, kernel_size=[3, 3, 3])
		net = conv_block_3d(net, self.n_filter, self.l2, kernel_size=[3, 3, 3])
		net = conv_block_3d(net, self.n_filter, self.l2, kernel_size=[3, 3, 3])
		print("transpose block final, shape = {}".format(net.get_shape()))

		#####################
		#      Softmax      #
		#####################
		net = tf.nn.dropout(net, keep_prob=self.dropout_tensor)  # new
		net = slim.conv3d(net, self.output_channel, [1, 1, 1],
						  activation_fn=None, scope='logits',
						  weights_regularizer=regularizers.l2_regularizer(scale=self.l2))
		return net


def count_params():
	total_parameters = 0
	for variable in tf.trainable_variables():
		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
	print("This seg_model has %d trainable parameters"% (total_parameters))


if __name__ == '__main__':

	x = tf.placeholder('float', [None, 128, 128, 16, 1])
	drop = tf.placeholder('float')
	vnet = VNet(16, x, 1, drop, 1.)
	y_pred = vnet.create_model()
	print(y_pred.get_shape())
	count_params()
	"""
	x = tf.placeholder('float', [None, 192, 192, 1])
	drop = tf.placeholder('float')
	msunet = MSUNet(32, x, 1, drop, 1.)
	y_pred = msunet.create_model()
	print(y_pred.get_shape())
	count_params()
	"""
