import tensorflow as tf
from ops import *
import pdb


class G_Res_cifar10(object):
    def __init__(self, args, scope):
        self.args = args
        self.scope = scope
        self.ch = 256
        self.bottom_width = 4
        self.initializer = tf.glorot_uniform_initializer(np.sqrt(2))
        self.initializer_sc = tf.glorot_uniform_initializer()

    def __call__(self, z, is_training, update_collection):
        def _add_basic_block(x_in, in_channels, out_channels, strides, scope):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                learnable_sc = (in_channels != out_channels) or (strides != 1)
                
                h = bn(x_in, self.args.momentum, self.args.epsilon, "bn_1", is_training)
                h = tf.nn.relu(h)
                if strides == 2:
                    h = conv(upsampling(h), out_channels, 3, 1, "SAME", self.initializer, "conv_2",
                             update_collection=update_collection)
                else:
                    h = conv(h, out_channels, 3, 1, "SAME", self.initializer, "conv_2", update_collection=update_collection)
                h = bn(h, self.args.momentum, self.args.epsilon, "bn_3", is_training)
                h = tf.nn.relu(h)
                out = conv(h, out_channels, 3, 1, "SAME", self.initializer, "conv_4", update_collection=update_collection)

                if learnable_sc is True:
                    if strides == 2:
                        shortcut = conv(upsampling(x_in), out_channels, 1, 1, "VALID", self.initializer_sc, "shortcut_conv",
                                        update_collection=update_collection)
                    else:
                        shortcut = conv(x_in, out_channels, 1, 1, "VALID", self.initializer_sc, "shortcut_conv",
                                        update_collection=update_collection)
                else:
                    shortcut = x_in

            return (out + shortcut)

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            n_batch = z.get_shape().as_list()[0]
            dense_1 = dense(z, self.ch * self.bottom_width ** 2, self.initializer_sc, "dense_1", update_collection)
            dense_1 = tf.reshape(dense_1, [n_batch, self.bottom_width, self.bottom_width, self.ch])
            block_2 = _add_basic_block(dense_1, self.ch, self.ch, 2, "block_2")
            block_3 = _add_basic_block(block_2, self.ch, self.ch, 2, "block_3")
            block_4 = _add_basic_block(block_3, self.ch, self.ch, 2, "block_4")
            bn_5 = bn(block_4, self.args.momentum, self.args.epsilon, "bn_5", is_training)
            bn_5 = tf.nn.relu(bn_5)
            conv_6 = tf.nn.tanh(conv(bn_5, 3, 3, 1, "SAME", self.initializer_sc, "conv_6", update_collection))

        return conv_6
