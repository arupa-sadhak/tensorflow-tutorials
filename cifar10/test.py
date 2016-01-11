import images
filenames = ['../datas/label%d.txt'%_ for _ in range(10)]
reader = images.Reader(filenames, batch_size=128, width=32, height=32, filetype='png', min_after_dequeue=1)
batch = reader.batch

from cifar10_architecture import Cifar10Architecture

import time
import tensorflow as tf
with tf.Session() as sess:
    network = Cifar10Architecture(sess)
    train_op, loss, accuracy = network.train(batch.images, batch.labels)
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('/tmp/cifar10', graph_def=sess.graph_def)

    sess.run( tf.initialize_all_variables() )
    reader.start(sess)
    
    try:
        for step in range(10001):
            start_time = time.time()
            _ = sess.run([train_op, loss, accuracy])
            duration = time.time() - start_time
            
            if step%10 == 0:
                num_examples_per_step = 128
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                print '[training] step:%04d, loss:%.6f accuracy:%.6f (%.1f examples/sec; %.3f sec/batch)'%(step, _[1], _[2], examples_per_sec, sec_per_batch)
                
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        reader.stop()
    
    reader.join()

