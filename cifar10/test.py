import tensorflow as tf

image = tf.image.decode_jpeg(tf.read_file('../datas/sample.jpg'), channels=3)
image = tf.cast( tf.image.resize_images(image, 32, 32), tf.float32 )
images = tf.train.batch([image], 1)

from image_classification import ImageClassification

with tf.Session() as sess:
    nn = ImageClassification({}, sess)
    output = nn.forward(images)
    
    sess.run( tf.initialize_all_variables() )
    prediction = sess.run([output])
    print prediction

