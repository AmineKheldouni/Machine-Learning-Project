
from importModule import *

def traite_zero(x):
    if x==0:
        return 1
    else:
        return x

# Classifieur spécialisé en modèle annotateurs dépendants d'une consigne toujours répondre 1 (vrai) aux exemples.
class LearnCrowdOrder:
    def __init__(self, T, N, d, l=0.):
        self.alpha = np.zeros((1,d)) # Poids des dimensions
        self.beta = 0
        self.w = np.random.rand(d,T) # Poids des labelleurs
        self.gamma = np.random.rand(1,T)
        self.nu = 0.1*np.ones((1,T)) # Proba de suivre la consigne
        self.S = 0.2*np.ones((1,T)) # Propension à suivre la consigne
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

    def y_cond_z(self, X, Y, gamma, w, nu, S):

        (N,d)=np.shape(X)
        (N,T)=np.shape(Y)

        eta = 1/(1+np.exp(-np.dot(X,w)-gamma)) # Taille : N,T

        y_cond_z = np.zeros((N,T,2))
        Z = np.tile(np.zeros(N).reshape(-1,1),(1,T))
        tmpPower = np.abs(Y-Z)
        y_cond_z[:,:,0] = np.multiply((1-nu),((1-eta)**tmpPower)*(eta**(1-tmpPower)))+np.multiply(nu, S**Y * (1-S)**(1-Y))

        Z = np.tile(np.ones(N).reshape(-1,1),(1,T))
        tmpPower = np.abs(Y-Z)
        y_cond_z[:,:,1] = np.multiply((1-nu),((1-eta)**tmpPower)*(eta**(1-tmpPower)))+np.multiply(nu, S**Y * (1-S)**(1-Y))

        return y_cond_z

    def expects_labels_Bernoulli(self, X, Y, alpha, beta, gamma, w, nu, S):
        """calcule les probas des labels z pour chaque donnée -> taille (N,2)
        en multipliant sur tous les labelleurs : la proba que le vrai label soit 0 ou 1 et que le label du labelleur Yt soit celui obtenu sachant la donnée i
        donnée dans le modèle de Bernoulli par la matrice y_z_cond_x = y_cond_z_cond_x * z_cond_x = y_cond_z * z_cond_x"""

        (N,d)=np.shape(X)
        (N,T)=np.shape(Y)

        eta = 1/(1+np.exp(-np.dot(X,w)-gamma)) # Taille : N,T

        #proba cond du label Yt du labelleur t pour la donnée i sachant le vrai label 0 ou 1 (Bernoulli)

        y_cond_z = self.y_cond_z(X, Y, gamma, w, nu, S)
        mat_z_cond_x = self.z_cond_x(X, alpha, beta)
        results = np.multiply(np.prod(y_cond_z,axis=1),mat_z_cond_x)
        return results

    def likelihood(self, Pt, X, Y, model, alpha, beta, gamma, w, nu, S):
        """renvoit la log-vraisemblance totale du modèle calculée grâce à Pt matrice des probas des vrais labels Z calculés à l'E-step,
        model=Bernoulli ou Gaussian"""

        (N,d)=np.shape(X)
        (N,T)=np.shape(Y)

        eta = 1/(1+np.exp(-np.dot(X,w)-gamma)) # Taille : N,T

        #proba cond du label Yt du labelleur t pour la donnée i sachant le vrai label 0 ou 1 (Bernoulli)

        y_cond_z = self.y_cond_z(X, Y, gamma, w, nu, S)
        mat_z_cond_x = self.z_cond_x(X, alpha, beta)

        esp = np.zeros((N,T))
        esp += np.multiply(Pt[:,1].reshape((-1,1)),np.log(abs(y_cond_z[:,:,1])+pow(10,-10))+np.log(abs(mat_z_cond_x[:,1])+pow(10,-10)).reshape((-1,1)))
        esp += np.multiply(Pt[:,0].reshape((-1,1)),np.log(abs(y_cond_z[:,:,0])+pow(10,-10))+np.log(abs(mat_z_cond_x[:,0])+pow(10,-10)).reshape((-1,1)))

        return np.sum(esp) - self.lb * np.linalg.norm(w)**2

    def grad_likelihood(self, Pt, X, Y, model, alpha, beta, gamma, w, nu, S):
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

        # mat2 = np.zeros((N,T))
        # mat2 = np.multiply(np.multiply(Pt[:,1].reshape((-1,1)),np.multiply(tmp_exp_2,1-tmp[:,:,1])-tmp[:,:,1])+np.multiply(Pt[:,0].reshape((-1,1)),np.multiply(tmp_exp_2,1-tmp[:,:,0])-tmp[:,:,0]),etasigma)
        # grad_lh_gamma = np.sum(mat2,axis=0)
        #
        # mat3 = np.zeros((N,d,T))
        # mat3 = np.multiply(np.repeat(mat2[:,np.newaxis,:],d,axis=1),np.repeat(X[:,:,np.newaxis],T,axis=2))
        # grad_lh_w = np.sum(mat3,axis=0) - self.lb * w #Taille d,T
        grad_etasigma_w = np.zeros((N,T,d))
        for t in range(T):
            grad_etasigma_w[:,t,:] = (etasigma[:,t]*(1-etasigma)[:,t]).reshape((N,1)) * X # Taille : N,T,d
        grad_etasigma_gamma = etasigma*(1-etasigma)

        y_cond_zxTMP = self.y_cond_z(X, Y, gamma, w, nu, S)

        grad_lh_etasigma = Pt[:,0].reshape((N,1))*np.tile((1-nu),(N,1))*(abs(Y)*etasigma**(abs(Y)-1)*(1-etasigma)**(1-abs(Y))+(1-abs(Y)*etasigma**abs(Y)*(1-etasigma)**(-abs(Y))))\
        /y_cond_zxTMP[:,:,0] + Pt[:,1].reshape((N,1))*np.tile((1-nu),(N,1))*(abs(Y-1)*etasigma**(abs(Y)-1)*(1-etasigma)**(1-abs(Y-1))+(1-abs(Y-1)*etasigma**abs(Y-1)*(1-etasigma)**(-abs(Y-1))))\
        /y_cond_zxTMP[:,:,1]

        grad_lh_w = np.zeros((d,T))
        for i in range(d):
            grad_lh_w[i,:] = np.sum(grad_lh_etasigma*grad_etasigma_w[:,:,i].reshape((N,T)),axis=0).reshape((1,T))

        grad_lh_gamma = np.sum(grad_lh_etasigma * grad_etasigma_gamma,axis=0).reshape((1,T))

        grad_lh_nu = np.sum(Pt[:,0].reshape((N,1)) * (-etasigma**abs(Y)*(1-etasigma)**(1-abs(Y))+S**Y*(1-S)**(1-Y))/y_cond_zxTMP[:,:,0] +\
        Pt[:,1].reshape((N,1)) * (-etasigma**abs(Y-1)*(1-etasigma)**(1-abs(Y-1))+S**Y*(1-S)**(1-Y))/y_cond_zxTMP[:,:,1] ,axis=0).reshape((1,T))

        grad_lh_S = np.sum(Pt[:,0].reshape((N,1)) * (Y*S**(Y-1)*(1-S)**(1-Y)+(1-Y)*S**Y*(1-S)**(-Y))/y_cond_zxTMP[:,:,0] +\
        Pt[:,1].reshape((N,1)) * (Y*S**(Y-1)*(1-S)**(1-Y)+(1-Y)*S**Y*(1-S)**(-Y))/y_cond_zxTMP[:,:,1] ,axis=0).reshape((1,T))

        return (grad_lh_alpha, grad_lh_beta, grad_lh_gamma, grad_lh_w, grad_lh_nu, grad_lh_S)

    def fit(self, X, Y, epsGrad=10**(-2), model="Bernoulli", eps = 10**(-8), max_iter=100, draw_convergence=False):
        N = X.shape[0]
        d = X.shape[1]
        T = Y.shape[1]

        #EM Algorithm

        #Initialization
        alphaNew = np.random.rand(1,d)
        betaNew = np.random.rand()
        wNew = np.random.rand(d,T)
        gammaNew = np.random.rand(1,T)
        # nuNew = np.random.rand(1,T)
        # sNew = np.random.rand(1,T)
        nuNew = np.random.rand(1,T)
        sNew = np.random.rand(1,T)

        self.alpha = np.zeros((1,d))
        self.beta = 0

        # alphaNew = np.random.rand(1,d)*10
        # betaNew = np.random.rand()*10
        # wNew = np.random.rand(d,T)*50
        # gammaNew = np.random.rand(1,T)*50

        cpt_iter=0
        LH = []
        #if model=="Bernoulli":
        Pt = self.expects_labels_Bernoulli(X, Y, self.alpha, self.beta, self.gamma, self.w, self.nu, self.S)
        LH2 = self.likelihood(Pt, X, Y, model, self.alpha, self.beta, self.gamma, self.w, self.nu, self.S)
        LH1 = LH2-1
        # while ( np.linalg.norm(self.alpha - alphaNew)**2 + np.linalg.norm(self.beta - betaNew)**2 > eps and cpt_iter < max_iter):
        while (cpt_iter < max_iter):

            LH1 = LH2
            print("ITERATION N°",cpt_iter)

            self.alpha = alphaNew
            self.beta = betaNew
            self.gamma = gammaNew
            self.w = wNew
            self.nu = nuNew
            self.S = sNew


            # Expectation (E-step)

            #if model=="Bernoulli":
            Pt = self.expects_labels_Bernoulli(X, Y, self.alpha, self.beta, self.gamma, self.w, self.nu, self.S)
            #elif model=="Gaussian":
            #Pt = self.expects_labels_Gaussian(X, Y, self.alpha, self.beta, self.gamma, self.w)

            #print(Pt)
            # Maximization (M-step)

            Galpha, Gbeta, Ggamma, Gw, Gnu, Gs = self.grad_likelihood(Pt, X, Y, model, alphaNew, betaNew, gammaNew, wNew, nuNew, sNew)
            normGrad = np.linalg.norm(Galpha)+np.linalg.norm(Gbeta)+np.linalg.norm(Ggamma)+np.linalg.norm(Gw)+np.linalg.norm(Gnu)+np.linalg.norm(Gs)
            grad_desc_count = 0
            # print("MAXIMIZATION : ")
            while (normGrad > epsGrad and grad_desc_count < 250):
                print("counter :", grad_desc_count)
                print("Norme Grad : ", normGrad)
                step = 0.005/((grad_desc_count+1))**2
                alphaNew += step * Galpha
                betaNew += step * Gbeta
                gammaNew += step * Ggamma
                wNew += step * Gw
                nuNew += step * Gnu
                sNew += step * Gs

                Galpha, Gbeta, Ggamma, Gw, Gnu, Gs = self.grad_likelihood(Pt, X, Y, model, alphaNew, betaNew, gammaNew, wNew, nuNew, sNew)
                normGrad = np.linalg.norm(Galpha)+np.linalg.norm(Gbeta)+np.linalg.norm(Ggamma)+np.linalg.norm(Gw)+np.linalg.norm(Gnu)+np.linalg.norm(Gs)
                grad_desc_count += 1

            cpt_iter+=1

            LH2=self.likelihood(Pt, X, Y, model, alphaNew, betaNew, gammaNew, wNew, nuNew, sNew)
            LH.append(LH2)
        self.alpha = alphaNew
        self.beta = betaNew
        self.gamma = gammaNew
        self.w = wNew
        self.nu = nuNew
        self.S = sNew

        print("ALPHA : ")
        print(self.alpha)

        print("BETA : ")
        print(self.beta)

        print("GAMMA : ")
        print(self.gamma)

        print("W : ")
        print(self.w)

        print("NU : ")
        print(self.nu)
        print("S : ")
        print(self.S)

        #print("############ Test BFGS #######################")
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
