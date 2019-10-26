import numpy as np
import random
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

if sys.argv[1] == 'test1':
    for i in range(int(sys.argv[2])):
        ran = np.random.randint(-50, high=50, size=(785, 10)) / 1000
    print('test1')
# print(ran)

if sys.argv[1] == 'test2':
    for i in range(int(sys.argv[2])):
        weights = np.ones((INPUTS,NEURONS))
        # print('dim weights: %s x %s' % (weights.shape[0],weights.shape[1]))
        random.seed(a=1)
        for i in range(0,INPUTS):
            for j in range(0,NEURONS):
                weights[i][j] = random.randrange(-5,5) / 100
    print('test1')

if sys.argv[1] == 'test index':
    con = []
    print(con)
    con = np.zeros((NEURONS,NEURONS))
    print(con)
    # confusion matrix
    # c = [[0 for x in range(10)] for n in range(10)]
    c = np.zeros((NEURONS,NEURONS))
    print(c)
    # prediction at 3
    y = np.array([[1,0,0,0,0,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0,0,0],
                  [1,0,0,0,0,0,0,0,0,0],
                  [1,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0,0,1,0],
                  [0,1,0,0,0,0,0,0,0,0]])
    # target at 4
    t = np.array([[0,0,1,0,0,0,0,0,0,0],
                  [0,0,0,1,0,0,0,0,0,0],
                  [0,0,1,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,1,0,0,1,0],
                  [0,0,0,0,0,0,1,0,1,0],
                  [0,0,0,0,0,0,0,1,1,0],
                  [0,0,0,0,0,0,0,0,1,0],
                  [0,0,0,1,0,0,0,0,0,0]])
    # index_y = np.where(y[0] == 1)[0][0]
    # index_t = np.where(t[0] == 1)[0][0]
    # print(index_t, index_y)
    # c[index_t][index_y] = 1
    c = np.dot(np.transpose(t),y)
    print(c)
    print(np.trace(c))
    num = 100
    print(f"{num:03d}")

if sys.argv[1] == 'test ref':
    listA = [0]
    listB = listA
    listB.append(1)
    print(listA, listB)


if sys.argv[1] == 'test weights':
    # dim w: 4 x 3
    weights = np.array([[.1,.2,.3],[.4,.5,.6],[.7,.8,.9],[1,1.1,1.2]])
    # dim i: 5 x 4
    inputs = np.array([[.4,.5,.6,.7],[.1,.2,.3,.4],[.7,.8,.9,.01],[.4,.4,.3,.4],[.3,.3,.4,.9]])
    # dim n: 5 x 3
    weights -= LR*(np.transpose(inputs), neuron-targets)

if sys.argv[1] == 'test loop':
    for x in range(NEURONS):
        print(x)

if sys.argv[1] == 'test memmap':
    np.set_printoptions(threshold=sys.maxsize)
    path = os.path.dirname(os.path.realpath(__file__))
    filename = path + '/../../MNIST/train_labels_array.csv'
    memmap = path + '/../../MNIST/train_labels_array.dat'

    # Load from csv
    array = np.loadtxt(filename, delimiter=',', max_rows=SAMPLES)
    # Save to dat
    fp = np.memmap(memmap, dtype='float32', mode='w+', shape=(SAMPLES,NEURONS))
    fp[:] = array[:]
    # Load from dat
    newfp = np.memmap(memmap, dtype='float32', mode='r+', shape=(SAMPLES,NEURONS))
    newfp_csv = path + '/../../MNIST/train_labels_array_from_dat.csv'
    np.savetxt(fname=newfp_csv, X=newfp, delimiter=',', fmt='%f')

if sys.argv[1] == 'test size':
    arr = np.zeros((20,10))
    print(arr.size)
    print(arr.shape)
    print(int(arr.shape[0]))
    num = int(arr.shape[0])
    print(num)
    print(arr[1].size)