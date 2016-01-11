import tensorflow as tf
import helper
import elements

from architecture import Architecture

class Cifar10Architecture(Architecture):
    def forward(self, images):
        helper._activation_summary(images)

        conv1 = tf.nn.elu( elements._convolution('conv1', images, kernel_shape=[5, 5, 3, 64]), name='conv1')
        helper._multichannel_image_summary('conv1/output', conv1)
        pool1 = elements._pooling('pool1', conv1, kernel_shape=[1, 3, 3, 1], strides=[1, 2, 2, 1])
        norm1 = elements._local_response_normalization('norm1', pool1)

        conv2 = tf.nn.elu( elements._convolution('conv2', norm1,  kernel_shape=[5, 5, 64, 64]), name='conv2')
        helper._multichannel_image_summary('conv2/output', conv2)
        norm2 = elements._local_response_normalization('norm2', conv2)
        pool2 = elements._pooling('pool2', norm2, kernel_shape=[1, 3, 3, 1], strides=[1, 2, 2, 1])
        
        local3 = tf.nn.elu( elements._fullconnect('local3', pool2,  hidden_nodes=384, stddev=0.04, wd=0.004), name='local3')
        local4 = tf.nn.elu( elements._fullconnect('local4', local3, hidden_nodes=192, stddev=0.04, wd=0.004), name='local4')
        logits = elements._fullconnect('linear', local4, hidden_nodes=10, stddev=1/192.0, wd=0.0)
        return logits

