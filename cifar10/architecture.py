import tensorflow as tf
import helper
import elements

class Architecture(object):
    def __init__(self, session):
        self._session = session

    def forward(self, images, reuse=False):
        pass

    def classification(self, logits):
        return tf.nn.softmax(logits, name='softmax')

    def loss(self, logits, labels):
        dense_labels = helper._sparse_to_dense(labels, num_classes=10)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, dense_labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return loss

    def accuracy(self, predictions, labels, at_k=1):
        num_samples = predictions.get_shape().as_list()[0]
        corrects = tf.cast( tf.nn.in_top_k(predictions, labels, at_k), tf.float32 )
        corrects_count = tf.reduce_sum(corrects)
        _accuracy = corrects_count / num_samples
        return _accuracy

    def train(self, logits, labels):
        predictions = self.classification( logits )
          
        with tf.name_scope('accuracy') as scope:
            accuracy = self.accuracy(predictions, labels, 1)
            accuracy_average_op = helper._add_scalar_average_summary('accuracy', accuracy)

        with tf.name_scope('updater') as scope:
            loss = self.loss(logits, labels)
            loss_averages_op = helper._add_loss_summaries(loss)
            with tf.control_dependencies([loss_averages_op, accuracy_average_op]):
                opt = tf.train.AdamOptimizer()
                grads = opt.compute_gradients(loss)
            apply_gradient_op = opt.apply_gradients(grads)

            for var in tf.trainable_variables():
                tf.histogram_summary(var.op.name, var)
            for grad, var in grads:
                if grad:
                    tf.histogram_summary(var.op.name + '/gradients', grad)

            with tf.control_dependencies([apply_gradient_op]):
                train_op = tf.no_op(name='train')
        return train_op, loss, accuracy        


