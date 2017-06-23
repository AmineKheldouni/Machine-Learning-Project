
from importModule import *

def traite_zero(x):
    if x==0:
        return 1
    else:
        return x

class LearnCrowd:
    def __init__(self, T, N, d, l=0.):
        self.alpha = np.zeros((1,d)) # Poids des dimensions
        self.beta = 0
        self.w = np.random.rand(d,T) # Poids des labelleurs
        self.gamma = np.random.rand(1,T)
        self.lb = l

    def z_cond_x(self, X, alpha, beta):
        """renvoie la matrice z_cond_x : proba que le vrai label soit 0 ou 1 sachant la donnée (Rlog) (indépendant du modèle Bernoulli/Gaussien)"""
        z_cond_x = np.zeros((X.shape[0],2))
        sigm = lambda x:  1/(1+np.exp(x))
        sigm=np.vectorize(sigm)
        tmp_value = -np.dot(X,alpha.T)-beta
        tmpsigm = sigm(np.where(tmp_value>0,10,tmp_value))
        z_cond_x[:,1] = tmpsigm.ravel()
        z_cond_x[:,0] = 1-z_cond_x[:,1]
        return z_cond_x

    def y_cond_z(self, X, Y, gamma, w):
        """Renvoie la matrice y_cond_z,x ; proba d'attribution d'une annotation connaissant les données"""
        (N,d)=np.shape(X)
        (N,T)=np.shape(Y)

        eta = 1/(1+np.exp(-np.dot(X,w)-gamma)) # Taille : N,T

        y_cond_z = np.zeros((N,T,2))
        Z = np.tile(np.zeros(N).reshape(-1,1),(1,T))
        tmpPower = np.abs(Y-Z)
        y_cond_z[:,:,0] = ((1-eta)**tmpPower)*(eta**(1-tmpPower))

        Z = np.tile(np.ones(N).reshape(-1,1),(1,T))
        tmpPower = np.abs(Y-Z)
        y_cond_z[:,:,1] = ((1-eta)**tmpPower)*(eta**(1-tmpPower))

        return y_cond_z

    def expects_labels_Bernoulli(self, X, Y, alpha, beta, gamma, w):
        """calcule les probas des labels z pour chaque donnée -> taille (N,2)
        en multipliant sur tous les labelleurs : la proba que le vrai label soit 0 ou 1 et que le label du labelleur Yt soit celui obtenu sachant la donnée i
        donnée dans le modèle de Bernoulli par la matrice y_z_cond_x = y_cond_z_cond_x * z_cond_x = y_cond_z * z_cond_x"""

        (N,d)=np.shape(X)
        (N,T)=np.shape(Y)
        eta = 1/(1+np.exp(-np.dot(X,w)-gamma)) # Taille : N,T
        y_cond_z = self.y_cond_z(X, Y, gamma, w)
        mat_z_cond_x = self.z_cond_x(X, alpha, beta)
        results = np.multiply(np.prod(y_cond_z,axis=1),mat_z_cond_x)
        return results

    def likelihood(self, Pt, X, Y, model, alpha, beta, gamma, w):
        """renvoit la log-vraisemblance totale du modèle calculée grâce à Pt matrice des probas des vrais labels Z calculés à l'E-step,
        model=Bernoulli ou Gaussian"""

        (N,d)=np.shape(X)
        (N,T)=np.shape(Y)
        eta = 1/(1+np.exp(-np.dot(X,w)-gamma)) # Taille : N,T
        y_cond_z = self.y_cond_z(X, Y, gamma, w)
        mat_z_cond_x = self.z_cond_x(X, alpha, beta)
        esp = np.zeros((N,T))
        esp += np.multiply(Pt[:,1].reshape((-1,1)),np.log(y_cond_z[:,:,1]+pow(10,-10))+np.log(mat_z_cond_x[:,1]+pow(10,-10)).reshape((-1,1)))
        esp += np.multiply(Pt[:,0].reshape((-1,1)),np.log(y_cond_z[:,:,0]+pow(10,-10))+np.log(mat_z_cond_x[:,0]+pow(10,-10)).reshape((-1,1)))
        return np.sum(esp) - self.lb * np.linalg.norm(w)**2

    def grad_likelihood(self, Pt, X, Y, model, alpha, beta, gamma, w, BFGS=False):
        """Returns the partial derivatives of likelihood according to
        alpha, beta, gamma and w
        model=Bernoulli ou Gaussian"""

        (N,d)=np.shape(X)
        (N,T)=np.shape(Y)

        tmp_exp = np.exp(-np.dot(X,alpha.T)-beta)

        mat = np.multiply(np.multiply(Pt[:,1].reshape((-1,1)),tmp_exp) - Pt[:,0].reshape((-1,1)),1/(1+tmp_exp))
        grad_lh_alpha = T*np.sum(mat.reshape((-1,1))*X,axis=0) #Taille 1,d
        grad_lh_beta = T*np.sum(mat.reshape((-1,1)),axis=0)  #Taille 1

        tmp_exp_2 = np.exp(-np.dot(X,w)-gamma) # Taille : N,T
        etasigma = 1/(1+tmp_exp_2)

        tmp = np.zeros((N,T,2))
        for z in range(2):
            tmp[:,:,z]=np.abs(Y-z)

        mat2 = np.zeros((N,T))
        mat2 = np.multiply(np.multiply(Pt[:,1].reshape((-1,1)),np.multiply(tmp_exp_2,1-tmp[:,:,1])-tmp[:,:,1])+np.multiply(Pt[:,0].reshape((-1,1)),np.multiply(tmp_exp_2,1-tmp[:,:,0])-tmp[:,:,0]),etasigma)
        grad_lh_gamma = np.sum(mat2,axis=0)

        mat3 = np.zeros((N,d,T))
        mat3 = np.multiply(np.repeat(mat2[:,np.newaxis,:],d,axis=1),np.repeat(X[:,:,np.newaxis],T,axis=2))
        grad_lh_w = np.sum(mat3,axis=0) - self.lb * w #Taille d,T
        if (BFGS):
            Grad = np.concatenate((grad_lh_alpha.ravel(),np.array([grad_lh_beta]).ravel(),grad_lh_gamma.ravel(),grad_lh_w.ravel()),axis=0)
            return Grad.ravel()
        return (grad_lh_alpha, grad_lh_beta, grad_lh_gamma, grad_lh_w)


    def fit(self, X, Y, epsGrad=10**(-3), model="Bernoulli", eps = 10**(-3), max_iter=500, draw_convergence=False):
        N = X.shape[0]
        d = X.shape[1]
        T = Y.shape[1]

        #EM Algorithm

        #Initialization

        # ArtificialData Initialization
        alphaNew = np.random.rand(1,d)
        betaNew = np.random.rand()
        wNew = np.random.rand(d,T)
        gammaNew = np.random.rand(1,T)

        self.alpha = np.zeros((1,d))
        self.beta = 0

        # TrueData Initialization
        # alphaNew = np.random.rand(1,d)*10
        # betaNew = np.random.rand()*10
        # wNew = np.random.rand(d,T)*50
        # gammaNew = np.random.rand(1,T)*50

        cpt_iter=0
        LH = []

        Pt = self.expects_labels_Bernoulli(X, Y, self.alpha, self.beta, self.gamma, self.w)
        normGrad = np.linalg.norm(self.grad_likelihood(Pt, X, Y, model, self.alpha, self.beta, self.gamma, self.w))
        diffLH = 1

        while (cpt_iter < max_iter and diffLH):


            print("ITERATION N°",cpt_iter)

            self.alpha = alphaNew
            self.beta = betaNew
            self.gamma = gammaNew
            self.w = wNew


            # Expectation (E-step)
            Pt = self.expects_labels_Bernoulli(X, Y, self.alpha, self.beta, self.gamma, self.w)

            # Maximization (M-step)
            Galpha, Gbeta, Ggamma, Gw = self.grad_likelihood(Pt, X, Y, model, alphaNew, betaNew, gammaNew, wNew)
            normGrad = np.linalg.norm(Galpha)+np.linalg.norm(Gbeta)+np.linalg.norm(Ggamma)+np.linalg.norm(Gw)
            grad_desc_count = 0

            while (normGrad > epsGrad and grad_desc_count < 400):
                step = 0.0001/(1+grad_desc_count)**2
                alphaNew += step * Galpha
                betaNew += step * Gbeta
                gammaNew += step * Ggamma
                wNew += step * Gw
                Galpha, Gbeta, Ggamma, Gw = self.grad_likelihood(Pt, X, Y, model, alphaNew, betaNew, gammaNew, wNew)
                normGrad = np.linalg.norm(Galpha)+np.linalg.norm(Gbeta)+np.linalg.norm(Ggamma)+np.linalg.norm(Gw)
                grad_desc_count += 1
            cpt_iter+=1

            LH.append(self.likelihood(Pt, X, Y, model, alphaNew, betaNew, gammaNew, wNew))
            if (len(LH) >= 2):
                diffLH = LH[-1]-LH[-2]

        self.alpha = alphaNew
        self.beta = betaNew
        self.gamma = gammaNew
        self.w = wNew

        if draw_convergence:
            plt.plot(np.linspace(1,cpt_iter,cpt_iter),LH)
            plt.title("Convergence de l'EM")
            plt.xlabel("nombre d'itérations")
            plt.ylabel("log-vraisemblance")
            plt.show()


    def predict(self, X, seuil):
        #on prédit les vrais labels à partir des données X
        tmp_exp = np.exp(-np.dot(X,self.alpha.T)-self.beta) # Taille : N
        proba_class_1 = 1/(1+tmp_exp)
        labels_predicted = proba_class_1 > seuil
        bool2float = lambda x:float(x)
        bool2float=np.vectorize(bool2float)
        return bool2float(labels_predicted).ravel()

    def score(self, X, Z, seuil):
        # On connaît la vérité terrain
        return np.mean(self.predict(X,seuil)==Z)

    def fitBFGS(self, X, Y, epsGrad=10**(-3), model="Bernoulli", eps = 10**(-3), max_iter=40, draw_convergence=False):
        N = X.shape[0]
        d = X.shape[1]
        T = Y.shape[1]

        ### EM Algorithm ###

        #Initialization
        alphaNew = np.random.rand(1,d)
        betaNew = np.random.rand()
        wNew = np.random.rand(d,T)
        gammaNew = np.random.rand(1,T)
        self.alpha = np.zeros((1,d))
        self.beta = 0
        cpt_iter=0
        LH = []

        Pt = self.expects_labels_Bernoulli(X, Y, self.alpha, self.beta, self.gamma, self.w)
        normGrad = np.linalg.norm(self.grad_likelihood(Pt, X, Y, model, self.alpha, self.beta, self.gamma, self.w))
        diffLH = 1

        while (cpt_iter < max_iter and diffLH):

            self.alpha = alphaNew
            self.beta = betaNew
            self.gamma = gammaNew
            self.w = wNew

            # Expectation (E-step)
            Pt = self.expects_labels_Bernoulli(X, Y, self.alpha, self.beta, self.gamma, self.w)

            # Maximization (M-step)
            grad_desc_count = 0

            BFGSfunc = lambda vect : -self.likelihood(Pt, X, Y, model, vect[0:d].reshape((1,d)), float(vect[d:d+1]), vect[d+1:d+1+T].reshape((1,T)), vect[d+1+T:d+1+T+d*T].reshape((d,T)))
            BFGSJac = lambda vect : -self.grad_likelihood(Pt, X, Y, model, vect[0:d].reshape((1,d)), float(vect[d:d+1]), vect[d+1:d+1+T].reshape((1,T)), vect[d+1+T:d+1+T+d*T].reshape((d,T)),BFGS=True)

            Teta_init = np.concatenate((alphaNew.ravel(),np.array([betaNew]).ravel(),gammaNew.ravel(),wNew.ravel()),axis=0) #initial guess

            LH.append(-BFGSfunc(Teta_init))
            result = minimize(BFGSfunc, Teta_init, method='BFGS', jac = BFGSJac,\
                              options={'gtol': 1e-5, 'disp': True, 'maxiter': 1000})
            print(result.message)

            Teta = result.x
            normGrad = np.linalg.norm(BFGSJac(Teta))

            alphaNew = Teta[0:d].reshape((1,d))
            betaNew = float(Teta[d:d+1])
            gammaNew = Teta[d+1:d+1+T].reshape((1,T))
            wNew = Teta[d+1+T:d+1+T+d*T].reshape((d,T))

            cpt_iter+=1

        self.alpha = alphaNew
        self.beta = betaNew
        self.gamma = gammaNew
        self.w = wNew

        if draw_convergence:
            plt.plot(np.linspace(1,cpt_iter,cpt_iter),LH)
            plt.title("Convergence de l'EM")
            plt.xlabel("nombre d'itérations")
            plt.ylabel("log-vraisemblance")
            plt.show()
