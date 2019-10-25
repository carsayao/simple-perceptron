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
LR = 0.001
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

        # print('target_array')
        # for s in range(SAMPLES):
        #     print(s, target_array[s])
        #target_output = path + '/../../MNIST/target_array.csv'
        #np.savetxt(fname=target_output, X=target_array, delimiter=',', fmt='%f')

        return inputs, targets

    def init_weights():
        weights = np.random.randint(-50,high=50,size=(INPUTS,NEURONS)) / 1000
        return weights

    def init_neurons():
        neuron = np.array(np.zeros((SAMPLES,NEURONS)))
        return neuron

    def activation(neuron):
        for n in range(SAMPLES):
            neuron[n] = np.where(neuron[n]>=np.amax(neuron[n]),1,0)
            print('neuron\n',neuron)
        return neuron

    #def weight_update():

    #test = np.array([-.41,-.5,-.5])
    #test1 = np.array([0,1,0])
    test2 = np.array([0,1,0])
    #print((test1==test2).all())
# ge#t prediction, then use activation function
    #print('test\n',test,'\ntest1\n',test1,'\ntest2\n',test2)
    #result = np.where(test>=np.amax(test),1,0)
    result = np.where(test2==1,1,0)
    #test3 = result-test1
    #print('test1*test2',test1*test2)
    #print(test1==test2)
    #print((test1==test2)*1)
    #print(np.amax(test))
    #print(np.amax(test, keepdims=True))
# pr#ediction array - actual array
    #print(test-test1)
# pr#ediction activation - actual array
    #print(test2-test1)
    print(result)
    #print('a(test)-test1\n',test3)
    #print('\n')

    

    neuron = init_neurons()
    inputs, targets = load()
    weights = init_weights()

    print('\n')
    

    #print('neuron',neuron)
    #print('inputs',inputs)
    #print('targets',targets)
    #print('targets',np.transpose(targets))
    print('dim neuron: %s x %s' % (neuron.shape[0],neuron.shape[1]))
    print('dim inputs: %s x %s' % (inputs.shape[0],inputs.shape[1]))
    # print('dim inputs[0]: %s x %s' % inputs[0].shape[0])
    print('dim targets: %s x %s' % (targets.shape[0],targets.shape[1]))
    print('dim weights: %s x %s' % (weights.shape[0],weights.shape[1]))
    w_t = np.transpose(weights)
    print('dim weights: %s x %s' % (w_t.shape[0],w_t.shape[1]))
    #print('weights[9]',weights[9])
    #print('weights',weights)
# calculating activations
    #activations = np.dot(inputs,np.transpose(weights))

# for each epoch
#     for each sample
#         calculate array of neurons
#         highest is prediction
#         if prediction is wrong
#             weight update

    confusion = np.zeros((NEURONS, NEURONS))
    print('running through epochs...')
    for e in range(EPOCHS):
        correct = 0
        #print('\tsample:')
        for n in range(NEURONS):
            neuron = np.dot(inputs,weights)
            np.savetxt(fname=path+'/../../MNIST/output.csv', X=neuron, delimiter=',', fmt='%f')
            for N in range(SAMPLES):
                #print('\t', s)
                np.set_printoptions(threshold=sys.maxsize)
                neuron[N] = np.where(neuron[N]>=np.amax(neuron[N]),1,0)
            # for n in range(NEURONS):
                #neuron[s][n] = 0
                # each neuron update per sample
                # for i in range(INPUTS):
                # activation function
                #print('neuron[s]',neuron[s], np.where(neuron[s]>=np.amax(neuron[s]),1,0))
                #print('epoch e', e, 'sample s', s, 'neuron[s]', neuron[s], targets[s], neuron[s][n]-targets[s][n])

                # for p in range(NEURONS):
                #     if neuron[n][p] == 1:
                #         prediction = p
                #         break
                # for a in range(NEURONS):
                #     if targets[n][a] == 1:
                #         actual = a
                #         break

                #print('incorrect', prediction, actual)
                # confusion[prediction][actual] += 1
            
            np.savetxt(fname=path+'/../../MNIST/output1.csv', X=neuron, delimiter=',', fmt='%f')
            # weight update
            if (neuron[s] == targets[s]).all():
                print('correct', neuron[s], targets[s])
                correct += 1
                continue
            else:
                for n in range(NEURONS):
                    prediction = 0
                    actual = 0
                    for x in range(INPUTS):
                        #print('\tweights[s][x]',weights[s][x], '\tLR', LR, '\tneuron[s][n]',neuron[s][n],'\ttargets[s][n]',targets[s][n],'\tinputs[s][x]\t',inputs[s][x])
                        weights[s][x] -= LR*(neuron[s][n]-targets[s][n])*inputs[s][x]
                        #print('\t\tafter w_update',weights[s][x])
            #print('\n')
        print('correct/SAMPLES', correct, SAMPLES, correct/SAMPLES)
        CORRECT.append(correct/SAMPLES)
    print(confusion)
    print('\nLR:', LR, '\nCORRECT:',CORRECT, '\nEPOCHS:', EPOCHS)
    save_confusion = path + '/../../MNIST/saved_recall_confusion.csv'
    np.savetxt(fname=save_confusion, X=confusion, delimiter=',', fmt='%f')
    #save_weights = path + '/../../MNIST/saved_weights.csv'
    #np.savetxt(fname=save_weights, X=weights, delimiter=',', fmt='%f')
    #output = path + '/../../MNIST/activation_on_neurons.csv'
    #output1 = path + '/../../MNIST/activation.csv'
    #print('saving to csv...')
    #np.savetxt(fname=output, X=neuron, delimiter=',', fmt='%f')
    #np.savetxt(fname=output1, X=a, delimiter=',', fmt='%f')


    #activations = np.where(activations>0,1,0)
    #print('inputs * weights^T')
    #print('dim activations: %s x %s' % (activations.shape[0],activations.shape[1]))
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
