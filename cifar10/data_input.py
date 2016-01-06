import os
import tarfile

from six.moves import urllib
import cPickle

import numpy as np

SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/'

def maybe_download(filename, work_directory):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros( (num_labels, num_classes) )
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def extract_datas(filename, one_hot=False):
    print('Extracting', filename)
    with tarfile.open(filename) as tar:
        train = {'data':np.array([]), 'labels':[], 'batch_label':[], 'filenames':[]}
        for i in range(1, 6):
            inner_filename = 'cifar-10-batches-py/data_batch_%d'%i
            print('Loading', inner_filename)
            data = cPickle.load( tar.extractfile( inner_filename ) )
            for key in train.keys():
                if key is 'data':
                    train[key] = np.concatenate( (train[key], data[key]), axis=0 ) if not i==1 else data[key]
                else:
                    train[key] += data[key]
        inner_filename='cifar-10-batches-py/test_batch'
        print('Loading', inner_filename)
        test = cPickle.load( tar.extractfile(inner_filename) )
        train['labels'] = dense_to_one_hot(np.array(train['labels'])) if one_hot else np.array( train['labels'] ) 
        test['labels'] =  dense_to_one_hot(np.array(test['labels'])) if one_hot else np.array( test['labels'] )
    return {'train':train, 'test':test}

class DataSet(object):
    def __init__(self, images, labels, one_hot=False):
        self._num_examples = images.shape[0]

        # convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0/255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()

    local_file = maybe_download('cifar-10-python.tar.gz', train_dir)
    datas = extract_datas(local_file, one_hot)

    data_sets.names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    data_sets.train = DataSet(datas['train']['data'], datas['train']['labels'])
    data_sets.test  = DataSet(datas['test']['data'],  datas['test']['labels'])

    return data_sets
