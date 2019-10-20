from mlxtend.data import loadlocal_mnist
import numpy as np
import os
import sys


def main():

    def bias():
        b = np.array([])
        for i in range(0,60000):
            b = np.append(b, [1])
        #print('dim b: %s' % b.shape[0])
        return b

    path = os.path.dirname(os.path.realpath(__file__))
    train_images_csv = path + '/../../MNIST/train_images.csv'
    # train_labels_csv = path + '/../../MNIST/train-labels.idx1-ubyte'

    v = np.array([])
    print('loading array...')
    v = np.loadtxt(train_images_csv, delimiter=',')
    print('dividing array...')
    v = v / 255
    b = bias()
    print(b)
    np.hstack((v, b))

    # Check if csv files already exist
    norm_images_csv = path + '/../../MNIST/train_images_normalized.csv'
    exists = os.path.isfile(norm_images_csv)
    if exists:
        print('normalized csv files already exist')
        sys.exit(0)
    try:
        print('saving normalized data as csvs...')
        np.savetxt(fname=norm_images_csv, X=v, delimiter=',', fmt='%f')
    except OSError as e:
        print('files not saved!')
        print(e.strerror)
        sys.exit(0)
    else:
        print('save successful!')

if __name__ == "__main__":
    main()
