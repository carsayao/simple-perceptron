import numpy as np
import os
import sys
import random

# Number of inputs
INPUTS = 785
# Number of neurons
NEURONS = 10
# Number of examples to test
SAMPLES = 25
# Number of epochs
EPOCHS = 5

path = os.path.dirname(os.path.realpath(__file__))

def main():

    def load():
        images = path + '/../../MNIST/train_images_normalized.csv'
        labels = path + '/../../MNIST/train_labels.csv'

        inputs = np.array([])
        targets = np.array([])
        print('loading input and target arrays...')
        inputs = np.loadtxt(images, delimiter=',', max_rows=SAMPLES)
        targets = np.loadtxt(labels, delimiter=',', max_rows=SAMPLES)
        return inputs, targets

    def init_weights():
        weights = np.ones((SAMPLES,INPUTS))
        random.seed(a=1)
        for i in range(0,SAMPLES):
            for j in range(0,INPUTS):
                weights[i][j] = random.randrange(-5,5) / 100
        return weights

    def init_neurons():
        neuron = np.array(np.zeros((SAMPLES,NEURONS)))
        return neuron


    # test1 = np.array([0,1,0])
    # test2 = np.array([1,0,0])
    # print(test1-test2)
    neuron = init_neurons()
    inputs, targets = load()
    weights = init_weights()
    #print('neuron',neuron)
    #print('inputs',inputs)
    #print('targets',targets)
    #print('targets',np.transpose(targets))
    print('dim neuron: %s x %s' % (neuron.shape[0],neuron.shape[1]))
    print('dim inputs: %s x %s' % (inputs.shape[0],inputs.shape[1]))
    print('dim targets: %s' % targets.shape[0])
    print('dim weights: %s x %s' % (weights.shape[0],weights.shape[1]))
    #print('weights[9]',weights[9])
    #print('weights',weights)
    activations = np.dot(inputs,np.transpose(weights))

# for each epoch
#     for each sample
#         calculate array of neurons
#         highest is prediction
#         if prediction is wrong
#             weight update

    # each neuron update per sample
    neuron[0][0] = inputs[0][0]*np.transpose(weights)[0][0]
    print(neuron[0][0])


    #activations = np.where(activations>0,1,0)
    print('inputs * weights^T')
    print('dim activations: %s x %s' % (activations.shape[0],activations.shape[1]))
    #weights -= 0.25*np.dot(np.transpose(inputs),activations-np.transpose(targets))

    #print('inputs[0]',inputs[0])
    #print('inputs[9]',inputs[9])

    # for i in range(SAMPLES):      # input vectors
    #     for j in range(NEURONS): # neurons
    #         activation[i][j] = 0
    #         for k in range(INPUTS):
    #             activation[i][j] += x[k][j] * x[i][k]
    #         print('i',i,'j',j,'\tn[j]',n[j],'w[i][j]',w[i][j],'x[i][j]',x[i][j])
    #         neuron[j] = weights[i][j]*x[i][j]

    #print(activation)

    output = path + '/../../MNIST/output.csv'
    np.savetxt(fname=output, X=activations, delimiter=',', fmt='%f')

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
