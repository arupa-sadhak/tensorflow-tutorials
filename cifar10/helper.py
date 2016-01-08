import re
import tensorflow as tf

def _variable(name, shape, initializer, device='/gpu:0'):
    with tf.device(device):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd, device='/gpu:0'):
    var = _variable(name, shape, tf.truncated_normal_initializer(stddev=stddev), device)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _convolution(name, input_tensor, kernel_shape, stddev=1e-4, wd=0.0):
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=kernel_shape, stddev=stddev, wd=wd)
        biases = _variable('biases', [kernel_shape[-1]], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        output_tensor = tf.nn.relu(bias, name=scope.name)
        _activation_summary(output_tensor)
    return output_tensor

def _pooling(name, input_tensor, kernel_shape, strides):
    with tf.variable_scope(name) as scope:
        output_tensor = tf.nn.max_pool(input_tensor, ksize=kernel_shape, strides=strides, padding='SAME', name=scope.name)
        _activation_summary(output_tensor)
    return output_tensor

def _local_response_normalization(name, input_tensor, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75):
    with tf.variable_scope(name) as scope:
        output_tensor = tf.nn.lrn(input_tensor, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name=scope.name)
        _activation_summary(output_tensor)
    return output_tensor

def _fullconnect_relu(name, input_tensor, hidden_nodes, stddev=0.04, wd=0.004):
    with tf.variable_scope(name) as scope:
        input_tensor_size = reduce(lambda x,y: x*y, input_tensor.get_shape()[1:].as_list())
        weights = _variable_with_weight_decay('weights', shape=[input_tensor_size, hidden_nodes], stddev=stddev, wd=wd)
        biases = _variable('biases', [hidden_nodes], tf.constant_initializer(0.1))

        reshaped = tf.reshape(input_tensor, [-1, input_tensor_size])
        output_tensor = tf.nn.relu_layer(reshaped, weights, biases, name=scope.name)
        _activation_summary(output_tensor)
    return output_tensor

def _fullconnect_linear(name, input_tensor, hidden_nodes, stddev=0.04, wd=0.0):
    with tf.variable_scope(name) as scope:
        input_tensor_size = reduce(lambda x,y: x*y, input_tensor.get_shape()[1:].as_list())
        weights = _variable_with_weight_decay('weights', shape=[input_tensor_size, hidden_nodes], stddev=stddev, wd=wd)
        biases = _variable('biases', [hidden_nodes], tf.constant_initializer(0.0))

        output_tensor = tf.nn.xw_plus_b(input_tensor, weights, biases, name=scope.name)
        _activation_summary(output_tensor)

    return output_tensor

