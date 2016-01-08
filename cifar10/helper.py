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


