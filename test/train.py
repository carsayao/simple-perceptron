import numpy as np
import os
import sys
import random

# Number of inputs
INPUTS = 785
# Number of neurons
NEURONS = 10
# Number of examples to train
SAMPLES = 60000
# Number of epochs
# EPOCHS = int(sys.argv[2])
EPOCHS = 10
# Learning Rate
# LR = float(sys.argv[1])
LR = 0.01
# Array of epochs to store correct %
CORRECT = []

path = os.path.dirname(os.path.realpath(__file__))
path = path + '/../MNIST/'

def main():

    def load():
        # images = path + '/../../MNIST/train_images_normalized.csv'
        # labels = path + '/../../MNIST/train_labels_array.csv'
        train_images_dat = path + 'train_images.dat'
        train_labels_dat = path + 'train_labels.dat'

        inputs = np.array([])
        targets = np.array([])
        print('loading input and target arrays...')
        # inputs = np.loadtxt(images, delimiter=',', max_rows=SAMPLES)
        # targets = np.loadtxt(labels, delimiter=',', max_rows=SAMPLES)

        inputs = np.memmap(train_images_dat, dtype='float32',
                        mode='r+', shape=(SAMPLES,INPUTS))
        targets = np.memmap(train_labels_dat, dtype='int8',
                        mode='r+', shape=(SAMPLES,NEURONS))

        return inputs, targets

    def init_weights():
        weights = np.random.randint(-50,high=50,size=(INPUTS,NEURONS)) / 1000
        print("weights[0]\n", weights[0])
        return weights

    def init_neurons():
        neuron = np.array(np.zeros((SAMPLES,NEURONS)))
        return neuron

    neuron = init_neurons()
    inputs, targets = load()
    weights = init_weights()

    print('\n')
    
    print('dim neuron: %s x %s' % (neuron.shape[0],neuron.shape[1]))
    print('dim inputs: %s x %s' % (inputs.shape[0],inputs.shape[1]))
    print('dim targets: %s x %s' % (targets.shape[0],targets.shape[1]))
    print('dim weights: %s x %s' % (weights.shape[0],weights.shape[1]))

    # Algorithm

    # for each epoch
    #     for each sample
    #         calculate array of neurons
    #         highest is prediction
    #         if prediction is wrong
    #             weight update

    confusion = []
    CORRECT.append(EPOCHS)
    CORRECT.append(LR)
    print('running through epochs...')
    for e in range(EPOCHS):
        confusion = []
        print('Epoch: %s lr: %s' % (e+1, LR))
        # print("  calculating neurons...")
        neuron = np.dot(inputs,weights)
        # print("  activation function...")
        for N in range(SAMPLES):
            np.set_printoptions(threshold=sys.maxsize)
            neuron[N] = np.where(neuron[N]>=np.amax(neuron[N]),1,0)
        # weight update
        # print("  weight update...")
        weights -= LR*np.dot(np.transpose(inputs), neuron-targets)
        # print("  confusion matrix...")
        confusion = np.dot(np.transpose(targets),neuron)
        accuracy = np.trace(confusion)/SAMPLES
        CORRECT.append(accuracy)
        print('  %:', round(accuracy*100,4))
    
    #print('saving to csv...')
    save_stats = path+'train_lr'+f"{int(LR*1000):03d}"+'-e'+f"{e+1:03d}"+'.csv'
    np.savetxt(fname=save_stats, X=CORRECT, delimiter=',')

    # save_weights = path + 'saved_weights.csv'
    # np.savetxt(fname=save_weights, X=weights, delimiter=',', fmt='%f')

    try:
        print('saving weights as .dat...')
        save_weights_path = path + 'saved_weights.dat'
        weights_dat = np.memmap(save_weights_path, dtype='float32',
                                mode='w+', shape=(INPUTS,NEURONS))
        weights_dat[:] = weights[:]
    except OSError as e:
        print('weights not saved!')
        print(e.strerror)
        sys.exit(0)
    else:
        print('save successful!')

if __name__ == "__main__":
    main()
