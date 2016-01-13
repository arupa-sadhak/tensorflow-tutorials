import tensorflow as tf
import helper
import elements

from architecture import Architecture

class Cifar10Architecture(Architecture):
    def forward(self, images, dropratio_tensor, reuse=False):
        conv1 = tf.nn.elu( elements._convolution('layer1_conv', images, kernel_shape=[5, 5, 3, 64], reuse=reuse), name='layer1_conv')
        pool1 = elements._pooling('layer1_pool', conv1, kernel_shape=[1, 3, 3, 1], strides=[1, 2, 2, 1], reuse=reuse)

        conv2 = tf.nn.elu( elements._convolution('layer2_conv', pool1,  kernel_shape=[5, 5, 64, 64], reuse=reuse), name='layer2_conv')
        pool2 = elements._pooling('layer2_pool', conv2, kernel_shape=[1, 3, 3, 1], strides=[1, 2, 2, 1], reuse=reuse)

	conv3 = tf.nn.elu( elements._convolution('layer3_conv', pool2,  kernel_shape=[5, 5, 64, 64], reuse=reuse), name='layer3_conv')
        pool3 = elements._pooling('layer3_pool', conv3, kernel_shape=[1, 3, 3, 1], strides=[1, 2, 2, 1], reuse=reuse)

        local4 = tf.nn.elu( elements._dropout('layer4_dropout', pool3, hidden_nodes=512, keep_prob_tensor=dropratio_tensor, stddev=0.04, wd=0.004, reuse=reuse), name='layer4_dropout')
        logits = elements._fullconnect('layer5_linear', local4, hidden_nodes=10, stddev=1/192.0, wd=0.0, reuse=reuse)

        if reuse:
            return logits

        with tf.name_scope('summary_images') as scope:
            helper._activation_summary(images)
            tf.image_summary('input/images', images, max_images=3)
            helper._multichannel_image_summary('conv1/output', conv1)
            helper._multichannel_image_summary('conv2/output', conv2)
            helper._multichannel_image_summary('conv3/output', conv3)
        return logits

