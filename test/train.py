import numpy as np
import os
import sys
import random

MAXROW = 10

path = os.path.dirname(os.path.realpath(__file__))

def main():

    def load():
        images = path + '/../../MNIST/train_images_normalized.csv'
        labels = path + '/../../MNIST/train_labels.csv'

        x = np.array([])
        t = np.array([])
        print('loading input and target arrays...')
        x = np.loadtxt(images, delimiter=',', max_rows=MAXROW)
        print('dim x: %s x %s' % (x.shape[0],x.shape[1]))
        t = np.loadtxt(labels, delimiter=',', max_rows=MAXROW)
        print('dim t: %s' % (t.shape[0]))
        return x, t

    def init_weights():
        w = np.array([])
        random.seed(a=1)
        for i in range(0,785):
            w = np.append(w, random.randrange(-5,5) / 100)



    x, t = load()
    w = init_weights()
    print(x)
    print(t)
    print(w)
    output = path + '/../../MNIST/output.csv'
    #np.savetxt(fname=output, X=x, delimiter=',', fmt='%f')

    # # Check if csv files already exist
    # norm_images = path + '/../../MNIST/train_images_normalized_10000.csv'
    # exists = os.path.isfile(norm_images)
    # if exists:
    #     print('normalized csv files already exist')
    # try:
    #     print('saving normalized data as csvs...')
    #     np.savetxt(fname=norm_images, X=v, delimiter=',', fmt='%f')
    # except OSError as e:
    #     print('files not saved!')
    #     print(e.strerror)
    #     sys.exit(0)
    # else:
    #     print('save successful!')

if __name__ == "__main__":
    main()
