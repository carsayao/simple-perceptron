import numpy as np

def main():
    v = np.array([[5,5,5],
                  [5,5,5]])
    a = np.eye(3)
    print('dim v: %s x %s' % (v.shape[0], v.shape[1]))
    print(v)
    print('dim a: %s x %s' % (a.shape[0], a.shape[1]))
    print(a,'\n')

    b = np.array(np.ones(2))
    print('dim b: %s' % (b.shape[0]))
    print(b)

    print(np.c_[v, b])

    c = np.array([[3],
                  [4],
                  [3]])
    #print(np.dot(v,c))
    x = np.array([-2,-2,-2,-2])
    y = np.array([3,3,3,3])
    activation = 0
    for i in range(0,4):
        activation += x[i]*y[i]
    print(activation)
    print(x*y)

    def activate(z):
        if z <= 0:
            return 0
        if z > 0:
            return 1
    print(activate(activation))

    def init_weights():
        w = np.array([])
        random.seed(a=1)
        for i in range(0,785):
            w = np.append(w, random.randrange(-5,5) / 100)
        return w

    w = init_weights()
    #def weight_update(ada, prediction, target):
    
    


if __name__ == "__main__":
    main()
