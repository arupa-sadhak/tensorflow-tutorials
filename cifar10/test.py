import images
import helper
from cifar10_architecture import Cifar10Architecture

import os
import math
import time
import tensorflow as tf

def main(args):
    batch_size=args.batch_size
    num_validation_images=10000.0
    filenames = ['../datas/label%d.txt'%_ for _ in range(10)]
    reader = images.Reader()
    reader.attach('train', filenames, batch_size=batch_size, width=32, height=32, filetype='png', shuffle=True, min_after_dequeue=1)
    reader.attach('valid', ['../datas/valid.txt'], batch_size=batch_size, width=32, height=32, filetype='png', shuffle=False, read_threads=5, min_after_dequeue=1)
    batch = reader.batch

    with tf.Session() as sess:
        network = Cifar10Architecture(sess)
        dropratio = tf.placeholder("float")
        with tf.name_scope('network') as scope:
            logits = network.forward(batch['train'].images, dropratio)
        train_op, loss, accuracy = network.train(logits, batch['train'].labels)

        with tf.name_scope('validation') as scope:
            valid_accuracy_op = network.accuracy( network.classification( network.forward(batch['valid'].images, dropratio, reuse=True) ), batch['valid'].labels )

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(args.summary_path, graph_def=sess.graph_def)

        sess.run( tf.initialize_all_variables() )
        reader.start(sess)
    
        try:
            for step in range(args.max_iter):
                start_time = time.time()
                _ = sess.run([train_op, loss, accuracy], feed_dict={dropratio:args.dropratio})
                duration = time.time() - start_time
            
                if step % 10 == 0:
                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    print '[training] step:%04d, loss:%.6f accuracy:%.6f (%.1f examples/sec; %.3f sec/batch)'%(step, _[1], _[2], examples_per_sec, sec_per_batch)
              
                if step % 1000 == 0:
                    start_time = time.time()
                    num_valid_step = int( math.floor(num_validation_images / batch_size) )
                    valid_accuracy = 0.0
                    for valid_step in range(num_valid_step):
                        valid_accuracy += sess.run(valid_accuracy_op, feed_dict={dropratio:1.0})
                    duration = time.time() - start_time
                    valid_accuracy /= num_valid_step
                    tf.scalar_summary('accuracy (validation)', valid_accuracy)
                    print '[validation] step:%03d, accuracy:%.6f (duration: %.1f sec; %.1f examples/sec; %.3f sec/batch)'%(step, valid_accuracy, duration, num_valid_step*batch_size/duration, duration/num_valid_step) 

                    summary = tf.Summary()
                    summary.ParseFromString( sess.run(summary_op, feed_dict={dropratio:0.5}) )
                    summary.value.add(tag='accuracy (validation)', simple_value=valid_accuracy)
                    summary_writer.add_summary(summary, step)
                elif step % 50 == 0:
                    summary_str = sess.run(summary_op, feed_dict={dropratio:0.5})
                    summary_writer.add_summary(summary_str, step)

                if step % 1000 == 0 or (step+1) == args.max_iter:
                    checkpoint_path = os.path.join(args.model_path, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step) 
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            reader.stop()
    
        reader.join()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--summary-path',       type=str,   required=True, help='/tmp/tensorflow/cifar10/description...')
    parser.add_argument('-t', '--model-path',         type=str,   required=True, help='./models/cifar10/description...')
    parser.add_argument('-d', '--dropratio',          type=float, default=0.5,   help='default:0.5')
    parser.add_argument('-b', '--batch-size',         type=int,   default=128,   help='default:128')
    parser.add_argument('-m', '--max-iter',           type=int,   default=10001, help='default:10001')
    args = parser.parse_args()

    from tensorflow.python.platform import gfile
    if gfile.Exists(args.model_path):
        gfile.DeleteRecursively(args.model_path)
    gfile.MakeDirs(args.model_path)
    
    main(args)
