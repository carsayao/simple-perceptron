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
# Number of epochs
EPOCHS = 3
# Learning Rate
LR = 0.001
# Array of epochs to store correct %
CORRECT = []

def main():

    path = os.path.dirname(os.path.realpath(__file__))
    x, y = loadlocal_mnist(
            images_path = path + '/../../MNIST/rawdata/train-images.idx3-ubyte',
            labels_path = path + '/../../MNIST/rawdata/train-labels.idx1-ubyte')

    print('Dimensions: %s x %s' % (x.shape[0], x.shape[1]))
    print('\n1st row', x[0])

    # print('digits:  0 1 2 3 4 5 6 7 8 9')
    # print('labels: %s' % np.unique(y))
    # print('Class distribution: %s' % np.bincount(y))

    # Check if csv files already exist
    train_images_csv = path + '/../../MNIST/train_images.csv'
    train_labels_csv = path + '/../../MNIST/train_labels_array.csv'

    target_array = np.zeros((SAMPLES, NEURONS), dtype=int)
    for t in range(SAMPLES):
        target_array[t][int(y[t])] = 1

    # exists = os.path.isfile(train_images_csv)
    # if exists:
    #     print('csv files already exist')
    # else:
    try:
        print('saving as csvs...')
        np.savetxt(fname=train_images_csv, X=x, delimiter=',', fmt='%d')
        np.savetxt(fname=train_labels_csv, X=target_array, delimiter=',', fmt='%d')
    except OSError as e:
        print('files not saved!')
        print(e.strerror)
        sys.exit(0)
    else:
        print('save successful!')

if __name__ == "__main__":
    main()
