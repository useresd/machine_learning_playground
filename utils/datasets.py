import numpy as np

# The following data is generated and the source is taken from https://cs231n.github.io/neural-networks-case-study/
def make_spirals(sample_size=150, n_classes=3, n_features=2):
    N = int(sample_size / n_classes) # number of points per class
    D = n_features # dimensionality
    K = n_classes # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y