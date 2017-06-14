
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
        """renvoie la matrice z_cond_x : proba que le vrai label soit 0 ou 1 sachant la donnée (Rlog) (indépendant du modèle Bernoulli/Gaussien"""
        z_cond_x = np.zeros((X.shape[0],2))
        sigm = lambda x:  1/(1+np.exp(x))
        sigm=np.vectorize(sigm)
        tmp_value = -np.dot(X,alpha.T)-beta
        tmpsigm = sigm(np.where(tmp_value>0,10,tmp_value))
        z_cond_x[:,1] = tmpsigm.ravel()
        z_cond_x[:,0] = 1-z_cond_x[:,1]
        return z_cond_x

    def y_cond_z(self, X, Y, gamma, w):

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

        #proba cond du label Yt du labelleur t pour la donnée i sachant le vrai label 0 ou 1 (Bernoulli)

        y_cond_z = self.y_cond_z(X, Y, gamma, w)
        mat_z_cond_x = self.z_cond_x(X, alpha, beta)

        results = np.multiply(np.prod(y_cond_z,axis=1),mat_z_cond_x)

        s = results[:,0] + results[:,1]
        results[:,0] = results[:,0]/s
        results[:,1] = results[:,1]/s

        # traite = lambda x:traite_zero(x)
        # traite = np.vectorize(traite)
        # sum = traite(sum)
        # results[:,0] = np.multiply(results[:,0],1/sum)
        # results[:,1] = np.multiply(results[:,1],1/sum)
        return results

    def likelihood(self, Pt, X, Y, model, alpha, beta, gamma, w):
        """renvoit la log-vraisemblance totale du modèle calculée grâce à Pt matrice des probas des vrais labels Z calculés à l'E-step,
        model=Bernoulli ou Gaussian"""

        (N,d)=np.shape(X)
        (N,T)=np.shape(Y)

        eta = 1/(1+np.exp(-np.dot(X,w)-gamma)) # Taille : N,T

        #proba cond du label Yt du labelleur t pour la donnée i sachant le vrai label 0 ou 1 (Bernoulli)

        y_cond_z = self.y_cond_z(X, Y, gamma, w)
        mat_z_cond_x = self.z_cond_x(X, alpha, beta)

        esp = np.zeros((N,T))
        esp += np.multiply(Pt[:,1].reshape((-1,1)),np.log(y_cond_z[:,:,1])+np.log(mat_z_cond_x[:,1]).reshape((-1,1)))
        esp += np.multiply(Pt[:,0].reshape((-1,1)),np.log(y_cond_z[:,:,0])+np.log(mat_z_cond_x[:,0]).reshape((-1,1)))

        return np.sum(esp) - self.lb * np.linalg.norm(w)**2

    def grad_likelihood(self, Pt, X, Y, model, alpha, beta, gamma, w):
        """Returns the partial derivatives of likelihood according to
        alpha, beta, gamma and w
        model=Bernoulli ou Gaussian"""

        (N,d)=np.shape(X)
        (N,T)=np.shape(Y)

        deltaPt = Pt[:,1]-Pt[:,0]
        deltaPt = deltaPt.reshape((N,1))
        tmp_exp = np.exp(-np.dot(X,alpha.T)-beta)

        grad_lh_alpha = np.sum(np.multiply(deltaPt*tmp_exp/((1+tmp_exp)**2),X) ,axis=0).reshape((1,d)) #Taille 1,d
        grad_lh_beta = np.sum(deltaPt*(tmp_exp,1/((1+tmp_exp)**2)))  #Taille 1
        tmp_exp_2 = np.exp(-np.dot(X,w)-gamma) # Taille : N,T
        etasigma = 1/(1+tmp_exp_2)

        grad_etasigma_gamma = etasigma*(1-etasigma) # Taille : N,T

        grad_etasigma_w = np.zeros((N,d,T))
        for t in range(0,T):
            grad_etasigma_w[:,:,t]= np.multiply((etasigma[:,t]*(1-etasigma[:,t])).reshape(N,1),X) #taille N,d,T

        grad_lh_etasigma = (-deltaPt)*((-1)**Y) # Taille : N,T

        grad_lh_gamma = np.sum(np.multiply(grad_lh_etasigma,grad_etasigma_gamma),axis=0).reshape((1,T)) #Taille 1,T
        grad_lh_w = np.sum(np.multiply(np.repeat(grad_lh_etasigma[:,np.newaxis,:],d,axis=1),grad_etasigma_w),axis=0).reshape((d,T)) #Taille d,T


        # "Zippage" des gradients en un grand vecteur
        Grad = np.concatenate((grad_lh_alpha.ravel(),np.array([grad_lh_beta]).ravel(),grad_lh_gamma.ravel(),grad_lh_w.ravel()),axis=0)

        return Grad.ravel()

    def grad_likelihood_2(self, Pt, X, Y, model, alpha, beta, gamma, w):
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

        # "Zippage" des gradients en un grand vecteur
        Grad = np.concatenate((grad_lh_alpha.ravel(),np.array([grad_lh_beta]).ravel(),grad_lh_gamma.ravel(),grad_lh_w.ravel()),axis=0)

        return Grad.ravel()

    def grad_likelihood_V2(self, Pt, X, Y, model, alpha, beta, gamma, w):
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

        # "Zippage" des gradients en un grand vecteur

        return (grad_lh_alpha, grad_lh_beta, grad_lh_gamma, grad_lh_w)

    def fitV2(self, X, Y, epsGrad=10**(-6), model="Bernoulli", eps = 10**(-8), max_iter=300, draw_convergence=False):
        N = X.shape[0]
        d = X.shape[1]
        T = Y.shape[1]

        #EM Algorithm

        #Initialization

        self.alpha = np.zeros((1,d))
        self.beta = 1
        alphaNew = np.random.rand(1,d)*(0.3)
        betaNew = 0.2
        wNew = np.random.rand(d,T)
        gammaNew = np.random.rand(1,T)

        cpt_iter=0
        LH = []
        #if model=="Bernoulli":
        Pt = self.expects_labels_Bernoulli(X, Y, self.alpha, self.beta, self.gamma, self.w)
        normGrad = np.linalg.norm(-self.grad_likelihood(Pt, X, Y, model, self.alpha, self.beta, self.gamma, self.w))

        while ( np.linalg.norm(self.alpha - alphaNew)**2 + np.linalg.norm(self.beta - betaNew)**2 > eps and cpt_iter < max_iter):
            cpt_iter+=1

            print("ITERATION N°",cpt_iter)

            self.alpha = alphaNew
            self.beta = betaNew
            self.gamma = gammaNew
            self.w = wNew
            normGrad = np.linalg.norm(-self.grad_likelihood(Pt, X, Y, model, self.alpha, self.beta, self.gamma, self.w))

            #print(self.alpha,self.beta,self.gamma,self.w)

            # Expectation (E-step)

            #if model=="Bernoulli":
            Pt = self.expects_labels_Bernoulli(X, Y, self.alpha, self.beta, self.gamma, self.w)

            Theta_init = np.concatenate((alphaNew.ravel(),np.array([betaNew]).ravel(),gammaNew.ravel(),wNew.ravel()),axis=0) #initial guess

            Gtheta = -self.grad_likelihood_2(Pt, X, Y, model, Theta_init[0:d].reshape((1,d)), float(Theta_init[d:d+1]), Theta_init[d+1:d+1+T].reshape((1,T)), Theta_init[d+1+T:d+1+T+d*T].reshape((d,T)))

            step = 0.01/(1+cpt_iter)**2
            Theta_init -= step*Gtheta

            Gtheta = -self.grad_likelihood_2(Pt, X, Y, model, Theta_init[0:d].reshape((1,d)), float(Theta_init[d:d+1]), Theta_init[d+1:d+1+T].reshape((1,T)), Theta_init[d+1+T:d+1+T+d*T].reshape((d,T)))

            print("NORME DU GRADIENT : ")
            normGrad = np.linalg.norm(Gtheta)
            print(normGrad)

            alphaNew = Theta_init[0:d].reshape((1,d))
            betaNew = float(Theta_init[d:d+1])
            gammaNew = Theta_init[d+1:d+1+T].reshape((1,T))
            wNew = Theta_init[d+1+T:d+1+T+d*T].reshape((d,T))
            LH.append(np.linalg.norm(self.alpha - alphaNew)**2 + np.linalg.norm(self.beta - betaNew)**2)

        self.alpha = alphaNew
        self.beta = betaNew
        self.gamma = gammaNew
        self.w = wNew
        #print("############ Test BFGS #######################")
        Teta_f = np.concatenate((self.alpha.ravel(),np.array([self.beta]).ravel(),self.gamma.ravel(),self.w.ravel()),axis=0)
        #print(abs(Teta_i-Teta_f))
        print("")
        if draw_convergence:
            plt.plot(np.linspace(1,cpt_iter,cpt_iter),LH)
            plt.title("Convergence de l'EM")
            plt.show()
    def fit(self, X, Y, epsGrad=10**(-5), model="Bernoulli", eps = 10**(-8), max_iter=100, draw_convergence=False):
        N = X.shape[0]
        d = X.shape[1]
        T = Y.shape[1]

        #EM Algorithm

        #Initialization

        self.alpha = np.zeros((1,d))
        self.beta = 0
        alphaNew = np.ones((1,d))*(0.4)
        betaNew = 0.1
        wNew = np.random.rand(d,T)
        gammaNew = np.random.rand(1,T)

        cpt_iter=0
        LH = []
        #if model=="Bernoulli":
        Pt = self.expects_labels_Bernoulli(X, Y, self.alpha, self.beta, self.gamma, self.w)
        normGrad = np.linalg.norm(self.grad_likelihood(Pt, X, Y, model, self.alpha, self.beta, self.gamma, self.w))

        # while ( np.linalg.norm(self.alpha - alphaNew)**2 + np.linalg.norm(self.beta - betaNew)**2 > eps and cpt_iter < max_iter):
        while ( normGrad > epsGrad and cpt_iter < max_iter):


            print("ITERATION N°",cpt_iter)
            print("Norme du gradient : ", normGrad)

            self.alpha = alphaNew
            self.beta = betaNew
            self.gamma = gammaNew
            self.w = wNew


            # Expectation (E-step)

            #if model=="Bernoulli":
            Pt = self.expects_labels_Bernoulli(X, Y, self.alpha, self.beta, self.gamma, self.w)
            #elif model=="Gaussian":
            #Pt = self.expects_labels_Gaussian(X, Y, self.alpha, self.beta, self.gamma, self.w)

            #print(Pt)
            # Maximization (M-step)

            # "Zippage" de self.alpha, self.beta, self.gamma, self.w en un grand vecteur Teta
            Galpha, Gbeta, Ggamma, Gw = self.grad_likelihood_V2(Pt, X, Y, model, alphaNew, betaNew, gammaNew, wNew)
            normGrad = np.linalg.norm(Galpha)+np.linalg.norm(Gbeta)+np.linalg.norm(Ggamma)+np.linalg.norm(Gw)

            step = 0.01/((cpt_iter+1))**2
            alphaNew += step * Galpha
            betaNew += step * Gbeta
            gammaNew += step * Ggamma
            wNew += step * Gw
            cpt_iter+=1

            LH.append(np.linalg.norm(self.alpha - alphaNew)**2 + np.linalg.norm(self.beta - betaNew)**2)

        self.alpha = alphaNew
        self.beta = betaNew
        self.gamma = gammaNew
        self.w = wNew
        #print("############ Test BFGS #######################")
        if draw_convergence:
            plt.plot(np.linspace(1,cpt_iter,cpt_iter),LH)
            plt.title("Convergence de l'EM")
            plt.show()


    def predict(self, X, seuil):
        #on prédit les vrais labels à partir des données X

        tmp_exp = np.exp(-np.dot(X,self.alpha.T)-self.beta) # Taille : N
        proba_class_1 = 1/(1+tmp_exp)
        labels_predicted = proba_class_1 > seuil
        bool2float = lambda x:float(x)
        bool2float=np.vectorize(bool2float)
        return bool2float(labels_predicted).ravel()

    def predictV3(self, X, Y, seuil, modeltrain):
        (N,d)=np.shape(X)
        predictions = np.zeros((N,1))

        if modeltrain=="Bernoulli":
           predictions += np.dot(X,self.alpha.T) + self.beta
           #print(predictions)
           for t in range(T):
              #print(np.dot(X,self.gamma[0,t]))
              predictions += np.multiply(np.reshape((-1)**(1-Y[:,t]),(-1,1)),np.reshape(np.dot(X,self.w[:,t]),(-1,1))) + self.gamma[0,t]
           #print(predictions)

        fun = lambda x:1-1/(1+np.exp(x))
        fun = np.vectorize(fun)
        probas_class_1 = fun(predictions)
        #print("HAHAHA")
        #print(probas_class_1)
        probas_class_1 = probas_class_1 > seuil
        #print(probas_class_1 > seuil)
        probas_class_1 = probas_class_1.ravel()
        convertfloat = lambda x:float(x)
        convertfloat = np.vectorize(convertfloat)
        #print(convertfloat(probas_class_1))
        return convertfloat(probas_class_1)

    def predictV2(self, Ytest, Xtrain, modeltrain):
        #Xtrain sont les données qui ont été utilisées pendant l'apprentissage (pas besoin de Ytrain ???, seul les poids appris suffisent ??)
        #modeltrain est le modèle utilisé pendant l'apprentissage
        #on prédit les vrais labels à partir des annotations de Ytest

        #On veut calculer non pas les probas des labels Z mais des labels Z sachant les annotations de Ytest pour chaque labelleur
        #Ainsi on va appliquer les mêmes méthodes "expects_labels_" mais que sur les données de Xtrain qui ont été annotées par un type d'annotation donné

        #Ou en fixant tous les Y à la Ytest comme suit ?

        N=Xtrain.shape[0]
        labels_predicted=np.zeros((N,1))

        unit=np.ones((N,1))

        for i in range(N):
            if modeltrain=="Bernoulli":
               #print(np.shape(Ytest[i,:].reshape((1,-1))))
               #print(np.shape(Ytest[i,:].reshape((1,-1)).tile((N,1))))
               Pt = self.expects_labels_Bernoulli(Xtrain, np.dot(unit,Ytest[i,:].reshape(1,-1)), self.alpha, self.beta, self.gamma, self.w)
            if modeltrain=="Gaussian":
               Pt = self.expects_labels_Gaussian(Xtrain, np.dot(unit,Ytest[i,:].reshape(1,-1)), self.alpha, self.beta, self.gamma, self.w)
            probas_class=np.mean(Pt,axis=0)
            labels_predicted[i,0] = probas_class[1] > probas_class[0]

        return labels_predicted.ravel()

    def score(self, X, Z, seuil):
        # On connaît la vérité terrain
        return np.mean(self.predict(X,seuil)==Z)

    def scoreV2(self,Ytest,Ztest,Xtrain,modeltrain):
        # On connaît la vérité terrain
        return np.mean(self.predictV2(Ytest, Xtrain, modeltrain)==Ztest)

    def scoreV3(self,X,Y,Z,seuil,modeltrain):
        # On connaît la vérité terrain
        return np.mean(self.predictV3(X,Y,seuil,modeltrain)==Z)
