from importModule import *

# Données réelles :

def load_XZ(filename):
    with open(filename,"r") as f:
        f.readline()
        data =[ [x for x in l.split()] for l in f if len(l.split())>2]
    X = np.array(data)
    return X[:,:X.shape[1]-1], (X[:,X.shape[1]-1].astype(int)+1)/2

def load_Y(filename):
    with open(filename,"r") as f:
        f.readline()
        data =[ [x for x in l.split()] for l in f if len(l.split())>2]
    X = np.array(data)
    return X[:,:X.shape[1]-1].astype(int)


def genereWithoutMissing(Xmissing,Ymissing, Zmissing):
    X = []
    Y = []
    Z = []
    for i in range(Xmissing.shape[0]):
        if not '?' in Xmissing[i]:
            X.append(Xmissing[i])
            Y.append(Ymissing[i])
            Z.append(Zmissing[i])
    return (np.array(X).astype(int)), ((np.array(Y).astype(int)+1)/2), ((np.array(Z).astype(int)+1)/2)
