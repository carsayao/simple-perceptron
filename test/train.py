import numpy as np
import os
import sys
import random

INPUTS = 785
NEURONS = 10
SAMPLES = 25

path = os.path.dirname(os.path.realpath(__file__))

def main():

    def load():
        images = path + '/../../MNIST/train_images_normalized.csv'
        labels = path + '/../../MNIST/train_labels.csv'

        inputs = np.array([])
        t = np.array([])
        print('loading input and target arrays...')
        inputs = np.loadtxt(images, delimiter=',', max_rows=SAMPLES)
        print('dim inputs: %s x %s' % (inputs.shape[0],inputs.shape[1]))
        t = np.loadtxt(labels, delimiter=',', max_rows=SAMPLES)
        print('dim t: %s' % (t.shape[0]))
        return inputs, t

    def init_weights():
        weight = np.ones((SAMPLES,INPUTS))
        #print('weight',weight)
        print('dim weight: %s x %s' % (weight.shape[0],weight.shape[1]))
        #print('weight[0]',weight[0])
        random.seed(a=1)
        for i in range(0,SAMPLES):
            for j in range(0,INPUTS):
                #print(i,j)
                # weight[i] = np.append(weight[i], random.randrange(-5,5) / 100)
                weight[i][j] = random.randrange(-5,5) / 100
        return weight

    def init_neurons():
        neuron = np.array(np.zeros((SAMPLES,NEURONS)))
        return neuron


    # test1 = np.array([0,1,0])
    # test2 = np.array([1,0,0])
    # print(test1-test2)
    neuron = init_neurons()
    inputs, t = load()
    weight = init_weights()
    #print('neuron',neuron)
    #print('inputs',inputs)
    #print('t',t)
    #print('t',np.transpose(t))
    print('dim neuron: %s x %s' % (neuron.shape[0],neuron.shape[1]))
    print('dim inputs: %s x %s' % (inputs.shape[0],inputs.shape[1]))
    print('dim t: %s' % t.shape[0])
    print('dim weight: %s x %s' % (weight.shape[0],weight.shape[1]))
    #print('weight[9]',weight[9])
    #print('weight',weight)
    activations = np.dot(inputs,np.transpose(weight))

# for each epoch
#     for each sample
#         calculate array of neurons
#         highest is prediction
#         if prediction is wrong
#             weight update

    # each neuron update per sample
    neuron[0][0] = inputs[0][0]*np.transpose(weight)[0][0]
    print(neuron[0][0])


    #activations = np.where(activations>0,1,0)
    print('inputs * weight^T')
    print('dim activations: %s x %s' % (activations.shape[0],activations.shape[1]))
    #weight -= 0.25*np.dot(np.transpose(inputs),activations-np.transpose(t))

    #print('inputs[0]',inputs[0])
    #print('inputs[9]',inputs[9])

    # for i in range(SAMPLES):      # input vectors
    #     for j in range(NEURONS): # neurons
    #         activation[i][j] = 0
    #         for k in range(INPUTS):
    #             activation[i][j] += x[k][j] * x[i][k]
    #         print('i',i,'j',j,'\tn[j]',n[j],'w[i][j]',w[i][j],'x[i][j]',x[i][j])
    #         neuron[j] = weight[i][j]*x[i][j]

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
