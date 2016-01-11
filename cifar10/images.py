import tensorflow as tf

def _read_csv_format(filename_queue, width, height, channels=3, filetype='png'):
    reader = tf.TextLineReader()
    key, line = reader.read(filename_queue)
    _ = tf.decode_csv(line, record_defaults=[[''], [0]])

    bytestream = tf.read_file(_[0])
    if filetype is 'png':
        image = tf.image.decode_png(bytestream, channels=channels)
    elif filetype in ['jpg', 'jpeg']:
        image = tf.image.decode_jpeg(bytestream, channels=channels)

    # TODO: image variation...

    image = tf.image.resize_images(image, height, width)
    label = tf.cast(_[1], tf.int32)
    image = tf.cast(image, tf.float32)
    return image, label

def _pipeline(filenames, batch_size, width, height, channels=3, filetype='png', shuffle=True, read_threads=None, min_after_dequeue=None, num_epochs=None):
    read_threads = read_threads if read_threads else len(filenames)
    min_after_dequeue = min_after_dequeue if min_after_dequeue else batch_size
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
    red_list = [_read_csv_format(filename_queue, width, height, channels, filetype) for _ in range(read_threads)]
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch_join(red_list, batch_size, capacity, min_after_dequeue)
    return image_batch, label_batch

class Reader(object):
    def __init__(self, filenames, batch_size, width, height, channels=3, filetype='png', shuffle=True, read_threads=None, min_after_dequeue=None, num_epochs=None):
        images, labels = _pipeline(filenames, batch_size, width, height, channels, filetype, shuffle, read_threads, min_after_dequeue, num_epochs)
        class Batch(object):
            pass
        self.batch = Batch()
        self.batch.images = images
        self.batch.labels = labels

    def start(self, sess):
        self.sess = sess
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def is_stop(self):
        return self.coord.should_stop()

    def stop(self):
        self.coord.request_stop()

    def join(self):
        self.coord.join(self.threads)
    

