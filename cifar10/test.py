import images
filenames = ['../datas/label0.txt', '../datas/label1.txt']
reader = images.Reader(filenames, batch_size=8, width=32, height=32, filetype='png', min_after_dequeue=1)
batch = reader.batch

import tensorflow as tf
from cifar10_architecture import Cifar10Architecture

with tf.Session() as sess:
    reader.start(sess)
    try:
        for i in range(10):
            _ = sess.run([batch.images, batch.labels])
            print _[1]
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        reader.stop()
    
    reader.join()


