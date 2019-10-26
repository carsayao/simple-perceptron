# Help from http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/

from mlxtend.data import loadlocal_mnist
import numpy as np
import os
import sys

# Number of inputs
INPUTS = 785
# Number of neurons
NEURONS = 10
# Number of examples to train
SAMPLES = 60000
# Number of test images
SAMPLES_T = 10000
# Number of epochs
EPOCHS = 3
# Learning Rate
LR = 0.001
# Array of epochs to store correct %
CORRECT = []
# Relative paths
train_images_raw = '/../MNIST/rawdata/train-images.idx3-ubyte'
train_labels_raw = '/../MNIST/rawdata/train-labels.idx1-ubyte'
test_images_raw = '/../MNIST/rawdata/t10k-images.idx3-ubyte'
test_labels_raw = '/../MNIST/rawdata/t10k-labels.idx1-ubyte'

def main():

    path = os.path.dirname(os.path.realpath(__file__))

    def load_train_data():
        x, y = loadlocal_mnist(
                images_path = path + train_images_raw,
                labels_path = path + train_labels_raw)
        return x, y
    
    def load_test_data():
        x, y = loadlocal_mnist(
                images_path = path + test_images_raw,
                labels_path = path + test_labels_raw)
        return x, y
    
    def convert_targets_train(array):
        target_array = np.zeros((SAMPLES, NEURONS), dtype=int)
        for t in range(SAMPLES):
            target_array[t][int(array[t])] = 1
        return target_array

    def convert_targets_test(array):
        target_array = np.zeros((SAMPLES_T, NEURONS), dtype=int)
        for t in range(SAMPLES_T):
            target_array[t][int(array[t])] = 1
        return target_array

    # print('Dimensions: %s x %s' % (x.shape[0], x.shape[1]))

    # print('digits:  0 1 2 3 4 5 6 7 8 9')
    # print('labels: %s' % np.unique(y))
    # print('Class distribution: %s' % np.bincount(y))

    # Check if csv files already exist
    train_images_csv = path + '/../MNIST/train_images.csv'
    train_labels_csv = path + '/../MNIST/train_labels.csv'
    train_images_dat = path + '/../MNIST/train_images.dat'
    train_labels_dat = path + '/../MNIST/train_labels.dat'

    test_images_csv = path + '/../MNIST/test_images.csv'
    test_labels_csv = path + '/../MNIST/test_labels.csv'
    test_images_dat = path + '/../MNIST/test_images.dat'
    test_labels_dat = path + '/../MNIST/test_labels.dat'

    train_images, train_labels = load_train_data()
    train_labels = convert_targets_train(train_labels)

    test_images, test_labels = load_test_data()
    test_labels = convert_targets_test(test_labels)

    # Normalize and add bias
    train_images = train_images / 255
    train_images = np.c_[train_images, np.ones(SAMPLES)]
    test_images = test_images / 255
    test_images = np.c_[test_images, np.ones(SAMPLES_T)]

    # print(train_images)
    print('dim train_images: %s x %s' % (train_images.shape[0], train_images.shape[1]))
    print('dim train_labels: %s x %s' % (train_labels.shape[0], train_labels.shape[1]))
    print('dim test_images: %s x %s' % (test_images.shape[0], test_images.shape[1]))
    print('dim test_labels: %s x %s' % (test_labels.shape[0], test_images.shape[1]))
    

    # exists = os.path.isfile(train_images_csv)
    # if exists:
    #     print('csv files already exist')
    # else:
    try:
        print('saving csvs...')
        np.savetxt(fname=train_images_csv, X=train_images,
                   delimiter=',', fmt='%f')
        np.savetxt(fname=train_labels_csv, X=train_labels,
                   delimiter=',', fmt='%d')

        np.savetxt(fname=test_images_csv, X=test_images,
                   delimiter=',', fmt='%f')
        np.savetxt(fname=test_labels_csv, X=test_labels,
                   delimiter=',', fmt='%d')

        print('saving dats...')
        fp0 = np.memmap(train_images_dat, dtype='float32',
                       mode='w+', shape=(SAMPLES,INPUTS))
        fp0[:] = train_images[:]
        fp1 = np.memmap(train_labels_dat, dtype='int8',
                       mode='w+', shape=(SAMPLES,NEURONS))
        fp1[:] = train_labels[:]

        fp2 = np.memmap(test_images_dat, dtype='float32',
                       mode='w+', shape=(SAMPLES_T,INPUTS))
        fp2[:] = test_images[:]
        fp3 = np.memmap(test_labels_dat, dtype='int8',
                       mode='w+', shape=(SAMPLES_T,NEURONS))
        fp3[:] = test_labels[:]
    except OSError as e:
        print('files not saved!')
        print(e.strerror)
        sys.exit(0)
    print('save successful!')

    newfp0 = np.memmap(train_images_dat, dtype='float32',
                       mode='r+', shape=(SAMPLES,INPUTS))
    np.set_printoptions(threshold=sys.maxsize)
    print(newfp0[0])
    newfp1 = np.memmap(train_labels_dat, dtype='int8',
                       mode='r+', shape=(SAMPLES,NEURONS))

    newfp2 = np.memmap(test_images_dat, dtype='float32',
                       mode='r+', shape=(SAMPLES_T,INPUTS))
    newfp3 = np.memmap(test_labels_dat, dtype='int8',
                       mode='r+', shape=(SAMPLES_T,NEURONS))

    train_images_csv_out = path + '/../MNIST/rawdata/train_images_out.csv'
    train_labels_csv_out = path + '/../MNIST/rawdata/train_labels_out.csv'
    np.savetxt(fname=train_images_csv_out, X=newfp0,
               delimiter=',', fmt='%f')
    np.savetxt(fname=train_labels_csv_out, X=newfp1,
               delimiter=',', fmt='%d')

    test_images_csv_out = path + '/../MNIST/rawdata/test_images_out.csv'
    test_labels_csv_out = path + '/../MNIST/rawdata/test_labels_out.csv'
    np.savetxt(fname=test_images_csv_out, X=newfp2,
               delimiter=',', fmt='%f')
    np.savetxt(fname=test_labels_csv_out, X=newfp3,
               delimiter=',', fmt='%d')


if __name__ == "__main__":
    main()
