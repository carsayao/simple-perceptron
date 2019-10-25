import numpy as np
import random
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

m = np.array([1,2,3,4,30,1,4])
i = np.where(m == m.max())
print(i)


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
