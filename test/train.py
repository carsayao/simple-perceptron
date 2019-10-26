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
EPOCHS = 3
# Learning Rate
LR = 0.01
# Array of epochs to store correct %
CORRECT = []


path = os.path.dirname(os.path.realpath(__file__))

def main():

    def load():
        images = path + '/../../MNIST/train_images_normalized.csv'
        labels = path + '/../../MNIST/train_labels_array.csv'

        inputs = np.array([])
        targets = np.array([])
        print('loading input and target arrays...')
        inputs = np.loadtxt(images, delimiter=',', max_rows=SAMPLES)
        targets = np.loadtxt(labels, delimiter=',', max_rows=SAMPLES)

        return inputs, targets

    def init_weights():
        weights = np.random.randint(-50,high=50,size=(INPUTS,NEURONS)) / 1000
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
        # np.savetxt(fname=path+'/../../MNIST/test_neuron-calculation.csv', X=neuron, delimiter=',', fmt='%f')
        # print("  activation function...")
        for N in range(SAMPLES):
            np.set_printoptions(threshold=sys.maxsize)
            neuron[N] = np.where(neuron[N]>=np.amax(neuron[N]),1,0)
            #print('epoch e', e, 'sample s', s, 'neuron[s]', neuron[s], targets[s], neuron[s][n]-targets[s][n])
        
        # np.savetxt(fname=path+'/../../MNIST/test_neuron-activation-function.csv', X=neuron, delimiter=',', fmt='%f')
        # weight update
        # print("  weight update...")
        weights -= LR*np.dot(np.transpose(inputs), neuron-targets)
        # np.savetxt(fname=path+'/../../MNIST/test_weight-update.csv', X=weights, delimiter=',', fmt='%f')
        
        # print("  confusion matrix...")
        # confusion = init_confusion(neuron, targets)
        confusion = np.dot(np.transpose(targets),neuron)
        accuracy = np.trace(confusion)/SAMPLES
        CORRECT.append(accuracy)
        print('  %:', round(accuracy*100),4)
        # save_confusion = path + '/../../MNIST/saved_train_confusion3-' + str(e) + '.csv'
        # np.savetxt(fname=save_confusion, X=confusion, delimiter=',', fmt='%f')

    
    #print('saving to csv...')
    save_stats = path + '/../../MNIST/train_e' + f"{e+1:03d}" + '-' + f"{int(LR*1000):03d}" + '.csv'
    np.savetxt(fname=save_stats, X=CORRECT, delimiter=',')

    #save_weights = path + '/../../MNIST/saved_weights.csv'
    #np.savetxt(fname=save_weights, X=weights, delimiter=',', fmt='%f')

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
