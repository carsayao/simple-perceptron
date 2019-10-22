import numpy as np
import os
import sys
import random

INPUTS = 785
NEURONS = 10
SAMPLES = 10

path = os.path.dirname(os.path.realpath(__file__))

def main():

    def load():
        images = path + '/../../MNIST/train_images_normalized.csv'
        labels = path + '/../../MNIST/train_labels.csv'

        x = np.array([])
        t = np.array([])
        print('loading input and target arrays...')
        x = np.loadtxt(images, delimiter=',', max_rows=SAMPLES)
        print('dim x: %s x %s' % (x.shape[0],x.shape[1]))
        t = np.loadtxt(labels, delimiter=',', max_rows=SAMPLES)
        print('dim t: %s' % (t.shape[0]))
        return x, t

    def init_weights():
        w = np.ones((SAMPLES,INPUTS))
        print('w',w)
        print('w[0]',w[0])
        for i in range(0,10):
            print(i)
        random.seed(a=1)
        for i in range(0,INPUTS):
            for j in range(0,SAMPLES):
                print(i,j)
                # w[i] = np.append(w[i], random.randrange(-5,5) / 100)
                #w[i][j] = random.randrange(-5,5) / 100
        return w

    def init_neurons():
        n = np.array(np.zeros(NEURONS))
        return n

    n = init_neurons()
    x, t = load()
    w = init_weights()
    print('n',n)
    print('x',x)
    print('t',t)
    print('dim w: %s' % w.shape[0])
    print('w[9]',w[9])
    print('x[0]',x[0])
    print(w.size, x[0].size)
    print('x[9]',x[9])
    for i in range(0,INPUTS):      # input vectors
        for j in range(0,NEURONS): # neurons
            n[j] = n[j]+w[i][j]*x[i][j]
    print(activation)

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
