import tensorflow as tf

image = tf.image.decode_jpeg(tf.read_file('../datas/sample.jpg'), channels=3)
image = tf.cast( tf.image.resize_images(image, 32, 32), tf.float32 )
images = tf.reshape(image, [-1, 32, 32, 3])

from cifar10_architecture import Cifar10Architecture

with tf.Session() as sess:
    nn = Cifar10Architecture({}, sess)
    output = nn.classification(images)
    values, indices = tf.nn.top_k(output, 1)

    sess.run( tf.initialize_all_variables() )
    prediction = sess.run( [output, indices] )
    print prediction

