import tensorflow as tf
import numpy as np

class PolicyValueNet():
	def __init__(self):
		self.state = None
		self.valid_actions = None
		self.log_probs = None
		self.value = None
		self.build_model()

	def eval_probs(self, game):
		state = game.get_state()[np.newaxis, :]
		valid_actions = game.get_valid_actions()
		log_probs = self.log_probs.eval(feed_dict={self.state:state, self.is_training:False})[0, valid_actions]
		probs = np.exp(log_probs)
		return zip(valid_actions, probs)

	def get_log_probs(self):
		return self.log_probs

	def eval_value(self, game):
		state = game.get_state()[np.newaxis, :]
		return self.value.eval(feed_dict={self.state:state, self.is_training:False})[0, 0]

	def get_value(self):
		return self.value

	def build_model(self):
		with tf.variable_scope('policy_value_net'):
			# input layer
			self.state = tf.placeholder(tf.float32, [None, 2, 6, 7], name='state')
			self.is_training = tf.placeholder(tf.bool, name='training')

			x = tf.reshape(self.state, [-1, 6, 7, 2], name='x')
			# hidden layers 1, we got [batch_size, 6, 7, 32]
			h_conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3],
				padding="same")
			h_conv1_bn = tf.layers.batch_normalization(inputs=h_conv1, training=self.is_training)
			h_conv1_output = tf.nn.relu(h_conv1_bn)
			# hidden layers 2, we got [batch_size, 6, 7, 64]
			h_conv2 = tf.layers.conv2d(inputs=h_conv1_output, filters=64, kernel_size=[3, 3],
				padding="same")
			h_conv2_bn = tf.layers.batch_normalization(inputs=h_conv2, training=self.is_training)
			h_conv2_output = tf.nn.relu(h_conv2_bn)
			# hidden layers 3, we got [batch_size, 6, 7, 128]
			h_conv3 = tf.layers.conv2d(inputs=h_conv2_output, filters=128, kernel_size=[3, 3],
				padding="same")
			h_conv3_bn = tf.layers.batch_normalization(inputs=h_conv3, training=self.is_training)
			h_conv3_output = tf.nn.relu(h_conv3_bn)

			# policy layers
			h_policy = tf.layers.conv2d(inputs=h_conv3_output, filters=2, kernel_size=[1, 1],
				padding="same")
			h_policy_bn = tf.layers.batch_normalization(inputs=h_policy, training=self.is_training)
			h_policy_output = tf.nn.relu(h_policy_bn)
			flat_p = tf.reshape(h_policy_output, [-1, 6*7*2])
			self.log_probs = tf.layers.dense(inputs=flat_p, units=7,
				activation=tf.nn.log_softmax)

			# value layers
			h_value = tf.layers.conv2d(inputs=h_conv3_output, filters=1, kernel_size=[1, 1],
				padding="same")
			h_value_bn = tf.layers.batch_normalization(inputs=h_value, training=self.is_training)
			h_value_output = tf.nn.relu(h_value_bn)
			flat_v = tf.reshape(h_value_output, [-1, 6*7*1])
			h_v1 = tf.layers.dense(inputs=flat_v, units=64, activation=tf.nn.relu)
			self.value = tf.layers.dense(inputs=h_v1, units=1, activation=tf.nn.tanh)
