import re
import tensorflow as tf

def _variable(name, shape, initializer, device='/cpu:0'):
    with tf.device(device):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd, device='/cpu:0'):
    var = _variable(name, shape, tf.truncated_normal_initializer(stddev=stddev), device)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _sparse_to_dense(labels, num_classes):
    sparse_labels = tf.reshape(labels, [-1, 1])
    batch_size = sparse_labels.get_shape[0]
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)
    return dense_labels

def _add_loss_summaries(loss):
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [loss])

    # Attach a scalar summmary to all individual losses and the total loss
    for l in losses + [loss]:
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op

