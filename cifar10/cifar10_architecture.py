import tensorflow as tf
import elements

class Cifar10Architecture(object):
    def __init__(self, options, session):
        self._options = options
        self._session = session

    def forward(self, images):
        conv1 = tf.nn.relu( elements._convolution('conv1', images, kernel_shape=[5, 5, 3, 64]), name='conv1')
        pool1 = elements._pooling('pool1', conv1, kernel_shape=[1, 3, 3, 1], strides=[1, 2, 2, 1])
        norm1 = elements._local_response_normalization('norm1', pool1)
        conv2 = tf.nn.relu( elements._convolution('conv2', norm1,  kernel_shape=[5, 5, 64, 64]), name='conv2')
        norm2 = elements._local_response_normalization('norm2', conv2)
        pool2 = elements._pooling('pool2', norm2, kernel_shape=[1, 3, 3, 1], strides=[1, 2, 2, 1])
        local3 = tf.nn.relu( elements._fullconnect('local3', pool2,  hidden_nodes=384, stddev=0.04, wd=0.004), name='local3')
        local4 = tf.nn.relu( elements._fullconnect('local4', local3, hidden_nodes=192, stddev=0.04, wd=0.004), name='local4')
        logits = elements._fullconnect('linear', local4, hidden_nodes=10, stddev=1/192.0, wd=0.0)
        return logits

    def classification(self, images):
        logits = self.forward(images)
        return tf.nn.softmax(logits, name='softmax')

    def loss(self, logits, labels):
        dense_labels = helper._sparse_to_dense(labels, num_classes=10)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, dense_labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def train(loss):
        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = helper._add_loss_summaries(loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer()
            grads = opt.compute_gradients(loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads)

        # Add histograms for trainable variables and gradients
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
        for grad, var in grads:
            if grad:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        with tf.control_dependencies([apply_gradient_op]):
            train_op = tf.no_op(name='train')
        return train_op

    def evaluate(self, images, labels, at_k=1):
        with tf.variable_scope('evaluate') as scope:
            num_samples = images.get_shape()[0]

            predictions = self.classification(images)
            corrects = tf.nn.in_top_k(predictions, labels, at_k)
            true_count += np.sum(corrects)

            accuracy = true_count / num_samples
        return accuracy


