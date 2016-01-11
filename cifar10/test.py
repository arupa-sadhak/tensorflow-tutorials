import images
batch_size=128
filenames = ['../datas/label%d.txt'%_ for _ in range(10)]
reader = images.Reader()
reader.attach('train', filenames, batch_size=batch_size, width=32, height=32, filetype='png', shuffle=True, min_after_dequeue=1)
reader.attach('valid', ['../datas/valid.txt'], batch_size=batch_size, width=32, height=32, filetype='png', shuffle=True, min_after_dequeue=1)
batch = reader.batch

import helper
from cifar10_architecture import Cifar10Architecture

import math
import time
import tensorflow as tf
with tf.Session() as sess:
    network = Cifar10Architecture(sess)
    train_op, loss, accuracy = network.train(batch['train'].images, batch['train'].labels)
    valid_accuracy_op = network.accuracy( network.classification( network.forward(batch['valid'].images, reuse=True) ), batch['valid'].labels )
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
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                print '[training] step:%04d, loss:%.6f accuracy:%.6f (%.1f examples/sec; %.3f sec/batch)'%(step, _[1], _[2], examples_per_sec, sec_per_batch)
              
            if step % 1000 == 0:
                start_time = time.time()
                num_valid_step = int( math.floor(10000.0 / batch_size) )
                valid_accuracy = 0.0
                for valid_step in range(num_valid_step):
                    valid_accuracy += sess.run(valid_accuracy_op)
                duration = time.time() - start_time
                valid_accuracy /= num_valid_step
                tf.scalar_summary('valid_accuracy', valid_accuracy)
                print '[validation] step:%03d, accuracy:%.6f (duration: %.1f sec; %.1f examples/sec; %.3f sec/batch)'%(step, valid_accuracy, duration, num_valid_step*batch_size/duration, duration/num_valid_step) 
 
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
 
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        reader.stop()
    
    reader.join()

