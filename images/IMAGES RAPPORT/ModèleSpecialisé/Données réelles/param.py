
self.alpha = np.zeros((1,d))
self.beta = 0
alphaNew = np.random.rand(1,d)*10
betaNew = np.random.rand()*10
wNew = np.random.rand(d,T)*50
gammaNew = np.random.rand(1,T)*50

cpt_iter=0
LH = []
