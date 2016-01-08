import tensorflow as tf
import network

class ImageClassification(object):
    def __init__(self, options, session):
        self._options = options
        self._session = session

    def forward(self, images):
        opts = self._options

        conv1 = network._convolution('conv1', images, kernel_shape=[5, 5, 3, 64])
        pool1 = network._pooling('pool1', conv1, kernel_shape=[1, 3, 3, 1], strides=[1, 2, 2, 1])
        norm1 = network._local_response_normalization('norm1', pool1) 
        conv2 = network._convolution('conv2', norm1,  kernel_shape=[5, 5, 64, 64])
        norm2 = network._local_response_normalization('norm2', conv2)
        pool2 = network._pooling('pool2', norm2, kernel_shape=[1, 3, 3, 1], strides=[1, 2, 2, 1])
        local3 = network._fullconnect_relu('local3', pool2,  hidden_nodes=384)
        local4 = network._fullconnect_relu('local4', local3, hidden_nodes=192)
        logits = network._fullconnect_linear('linear', local4, hidden_nodes=10, stddev=1/192.0, wd=0.0)

        return logits

        
        

