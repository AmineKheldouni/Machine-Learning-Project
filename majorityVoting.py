from importModule import *

#"Classifieur" par jugement majoritaire sur les labels donnés par les divers annotateurs

class MajorityVoting:
    def __init__(self):
        pass
    def fit(self, X, Y):
        pass
    def predict(self, Y,seuil):
        (N,T)=np.shape(Y)
        predictions = np.mean(Y,axis=1)
        predictions = predictions > seuil
        return predictions
    def score(self, Y, Z,seuil):
        return np.mean(self.predict(Y,seuil)==Z)
