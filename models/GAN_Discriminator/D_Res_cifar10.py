import tensorflow as tf
from ops import *

class D_Res_cifar10(object):
    def __init__(self, args, scope):
        self.args = args
        self.scope = scope
        self.ch = 128
        self.initializer = tf.glorot_uniform_initializer(np.sqrt(2))
        self.initializer_sc = tf.glorot_uniform_initializer()

    def __call__(self, x, is_training, update_collection):
        def _add_optimized_block(x_in, out_channels, scope):
            with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
                h = conv(x_in, out_channels, 3, 1, "SAME", self.initializer, "sn_opconv_1", sn=True,
                         update_collection=update_collection)
                h = tf.nn.relu(h)
                h = conv(h, out_channels, 3, 1, "SAME", self.initializer, "sn_opconv_2", sn=True,
                         update_collection=update_collection)
                out = downsampling(h)

                shortcut = conv(downsampling(x_in), out_channels, 1, 1, "VALID", self.initializer_sc, "shorcut_sn_opconv",
                                sn = True, update_collection=update_collection)

            return (out + shortcut)

        def _add_basic_block(x_in, in_channels, out_channels, strides, scope):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                learnable_sc = (in_channels != out_channels) or (strides != 1)
                h = tf.nn.relu(x_in)
                h = conv(h, in_channels, 3, 1, "SAME", self.initializer, "sn_conv_1", sn=True,
                         update_collection=update_collection)
                h = tf.nn.relu(h)
                out = conv(h, out_channels, 3, 1, "SAME", self.initializer, "sn_conv_2", sn=True,
                           update_collection=update_collection)
                if strides == 2:
                    out = downsampling(out)

                if learnable_sc is True:
                    shortcut = conv(x_in, out_channels, 1, 1, "VALID", self.initializer_sc, "shortcut_sn_conv", sn=True,
                                    update_collection=update_collection)
                    if strides == 2:
                        shortcut = downsampling(shortcut)

                else:
                    shortcut = x_in

            return (out + shortcut)

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            block_1 = _add_optimized_block(x, self.ch, "block_1")
            block_2 = _add_basic_block(block_1, self.ch, self.ch, 2, "block_2")
            block_3 = _add_basic_block(block_2, self.ch, self.ch, 1, "block_3")
            block_4 = _add_basic_block(block_3, self.ch, self.ch, 1, "block_4")
            block_4 = tf.nn.relu(block_4)
            block_4 = tf.reduce_mean(block_4, axis=(1, 2))
            dense_5 = dense(block_4, 1, self.initializer_sc, "dense_5", update_collection, sn = True)

        return dense_5
