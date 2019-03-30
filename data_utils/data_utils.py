from keras.datasets import cifar10, cifar100, fashion_mnist
import numpy as np
import pdb

class data_manager(object):
    def __init__(self, dataset, is_training = True):
        self.self = self
        self.dataset = dataset
        self.is_training = is_training
        
    def __call__(self):
        if self.dataset == 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            y_train = y_train[:, 0]
            y_test = y_test[:, 0]
        elif self.dataset == 'cifar100':
            (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode = 'coarse')
            y_train = y_train[:, 0]
            y_test = y_test[:, 0]
        elif self.dataset == 'fashion_mnist':
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        else:
            pass
        x_train = (x_train-127.5)/127.5

        return x_train,  y_train
        
class batch_manager(object):
    def __init__(self, data, labels, batch_size):
        self.self = self
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.next_batch_pointer = 0
        assert len(data) == len(labels), "should be len(data) == len(labels)"

    def shuffle_samples(self):
        num_samples = len(self.data)
        indices = np.random.permutation(np.arange(num_samples//4))
        self.data = np.split(self.data, 4)
        self.labels = np.split(self.labels, 4)
        self.data = self.data[indices]
        self.labels = self.labels[indices]
        self.data = np.concatenate(self.data, axis = 0)
        self.labels = np.concatenate(self.labels, axis =0)

    def get_next_batch(self):
        num_sample_left = len(self.data) - self.next_batch_pointer
        if num_sample_left >= self.batch_size:
            batch_xs = self.data[self.next_batch_pointer: self.next_batch_pointer + self.batch_size]
            batch_ys = self.labels[self.next_batch_pointer: self.next_batch_pointer + self.batch_size]
            self.next_batch_pointer += self.batch_size
        else:
            batch_xs_1 = self.data[self.next_batch_pointer:len(self.data)]
            batch_ys_1 = self.labels[self.next_batch_pointer:len(self.data)]
            self.shuffle_samples()
            batch_xs_2 = self.data[0: self.batch_size - num_sample_left]
            batch_ys_2 = self.labels[0: self.batch_size - num_sample_left]
            batch_xs = np.concatenate((batch_xs_1, batch_xs_2))
            batch_ys = np.concatenate((batch_ys_1, batch_ys_2))
            self.next_batch_pointer = self.batch_size - num_sample_left

        return batch_xs, batch_ys
