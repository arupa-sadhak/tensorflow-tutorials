import tensorflow as tf
import helper

def _convolution(name, input_tensor, kernel_shape, stddev=1e-4, wd=0.0):
    with tf.variable_scope(name) as scope:
        kernel = helper._variable_with_weight_decay('weights', shape=kernel_shape, stddev=stddev, wd=wd)
        biases = helper._variable('biases', [kernel_shape[-1]], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')
        output_tensor = tf.nn.bias_add(conv, biases)
        helper._activation_summary(output_tensor)
    return output_tensor

def _pooling(name, input_tensor, kernel_shape, strides):
    with tf.variable_scope(name) as scope:
        output_tensor = tf.nn.max_pool(input_tensor, ksize=kernel_shape, strides=strides, padding='SAME', name=scope.name)
        helper._activation_summary(output_tensor)
    return output_tensor

def _local_response_normalization(name, input_tensor, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75):
    with tf.variable_scope(name) as scope:
        output_tensor = tf.nn.lrn(input_tensor, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name=scope.name)
        helper._activation_summary(output_tensor)
    return output_tensor

def _fullconnect(name, input_tensor, hidden_nodes, stddev=0.04, wd=0.0):
    with tf.variable_scope(name) as scope:
        input_tensor_size = reduce(lambda x,y: x*y, input_tensor.get_shape()[1:].as_list())
        weights = helper._variable_with_weight_decay('weights', shape=[input_tensor_size, hidden_nodes], stddev=stddev, wd=wd)
        biases = helper._variable('biases', [hidden_nodes], tf.constant_initializer(0.0))

        reshaped = tf.reshape(input_tensor, [-1, input_tensor_size])
        output_tensor = tf.nn.xw_plus_b(reshaped, weights, biases, name=scope.name)
        helper._activation_summary(output_tensor)
    return output_tensor

