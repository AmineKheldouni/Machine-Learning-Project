# Modules importés

import numpy as np
import matplotlib.pyplot as plt
from tools import *
from scipy.optimize import minimize
from random import gauss
import math

#Bloc génération de données artificielles de dimension 2

def gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02):
    """ Generateur de donnees,
        :param centerx: centre des gaussiennes
        :param centery:
        :param sigma: des gaussiennes
        :param nbex: nombre d'exemples
        :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
        :param epsilon: bruit dans les donnees
        :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if data_type==0:
        #melange de 2 gaussiennes
        xpos=np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),int(nbex//2))
        xneg=np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),int(nbex//2))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))
    if data_type==1:
        #melange de 4 gaussiennes
        xpos=np.vstack((np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),int(nbex//4)),np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),int(nbex/4))))
        xneg=np.vstack((np.random.multivariate_normal([-centerx,centerx],np.diag([sigma,sigma]),int(nbex//4)),np.random.multivariate_normal([centerx,-centerx],np.diag([sigma,sigma]),int(nbex/4))))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex//2),-np.ones(int(nbex//2))))

    if data_type==2:
        #echiquier
        data=np.reshape(np.random.uniform(-4,4,2*nbex),(nbex,2))
        y=np.ceil(data[:,0])+np.ceil(data[:,1])
        y=2*(y % 2)-1
    # un peu de bruit
    data[:,0]+=np.random.normal(0,epsilon,nbex)
    data[:,1]+=np.random.normal(0,epsilon,nbex)
    # on mélange les données
    idx = np.random.permutation((range(y.size)))
    data=data[idx,:]
    y=y[idx]
    return data,y

def plot_data(data,labels=None):
    """
    Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    plt.figure(figsize=(8,8))
    cols,marks = ["red", "blue","green", "orange", "black", "cyan"],[".","+","*","o","x","^"]
    if labels is None:
        plt.scatter(data[:,0],data[:,1],marker="x", linewidth=3.0)
        return
    for i,l in enumerate(sorted(list(set(labels.flatten())))):
        plt.scatter(data[labels==l,0],data[labels==l,1],c=cols[i],marker=marks[i], linewidth=3.0)

def plot_frontiere(data,f,step=20):
    """ Trace un graphe de la frontiere de decision de f
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid,x,y=make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape),256)

def make_grid(data=None,xmin=-5,xmax=5,ymin=-5,ymax=5,step=20):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:,0]),  np.min(data[:,0]), np.max(data[:,1]), np.min(data[:,1])
    x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y


#Fonctions de générations de données de Bernoulli

def modifie_label_Bernoulli(label,proba):
    """modifie le vrai label en choisissant l'autre avec une probabilité 1-proba"""
    valeur_proba=np.random.uniform(0,1)
    label_res=label
    if(valeur_proba>=proba):
        label_res=1-label
    return label_res

def generation_Bernoulli(N,T,qualite_annotateur_Bernoulli,noise_truth):
    """retourne en xtrain les données de dimension 2, en ytrain les annotations, en ztrain les vrais labels
    avec pour qualite_annotateurs une liste contenant les probabilités de succès de chaque annotateur
    noise_truth est le bruit de l'attribution des vrais labels gaussiens sur les données"""
    xtrain,ztrain = gen_arti(nbex=N,data_type=0,epsilon=noise_truth) #vrai labels non bruités
    ztrain=(ztrain+1)/2
    ytrain=np.zeros((N,T)) #changement des labels
    for t in range(T):
        annote=lambda x:modifie_label_Bernoulli(x,qualite_annotateur_Bernoulli[t])
        annote=np.vectorize(annote)
        ytrain[:,t]=annote(ztrain)
    return xtrain,ytrain,ztrain

def modifie_label_Bernoulli_xdepend(data,label,proba):
    """modifie le vrai label en choisissant l'autre avec une probabilité proba[0] (dans la zone 1)
    et proba[1] (dans la zone 2)"""
    valeur_proba=np.random.uniform(0,1)
    label_res=label
    if data[0]>=0: #donnees à x négatif (groupe 1 de données)
        if(valeur_proba>=proba[0]):
           label_res=1-label
    else: #données à x positif (groupe 2 de données)
        if(valeur_proba>=proba[1]):
           label_res=1-label
    return label_res

def generation_Bernoulli_xdepend(N,T,qualite_annotateur_Bernoulli,noise_truth):
    """retourne en xtrain les données de dimension 2, en ytrain les annotations, en ztrain les vrais labels
    avec pour qualite_annotateur_Bernoulli les probabilités de succès de chaque annotateur dans chaque zone"""
    xtrain,ztrain = gen_arti(nbex=N,data_type=0,epsilon=noise_truth) #vrai labels non bruités
    ztrain=(ztrain+1)/2
    ytrain=np.zeros((N,T)) #changement des labels
    for t in range(T):
        annote=lambda idx_data,z:modifie_label_Bernoulli_xdepend(xtrain[idx_data,:],z,qualite_annotateur_Bernoulli[t])
        annote=np.vectorize(annote)
        ytrain[:,t]=annote(list(range(np.shape(xtrain)[0])),ztrain)
    return xtrain,ytrain,ztrain

def modifie_label_gaussien(label,variance_annoteur):
    """modifie le vrai label en choississant un autre label donnée par la gaussienne centré sur le vrai label
    avec la variance de l'annoteur"""
    valeur_label=gauss(label,variance_annoteur)
    return valeur_label

def generation_Gaussian(N,T,qualite_annotateur_gaussien,noise_truth):
    """retourne en xtrain les données de dimension 2, en ytrain les annotations, en ztrain les vrais labels
    avec pour qualite_annotateurs une liste contenant les variances de chaque annotateur
    noise_truth est le bruit de l'attribution des vrais labels gaussiens sur les données"""
    xtrain,ztrain = gen_arti(nbex=N,data_type=0,epsilon=noise_truth) #vrai labels non bruités
    ytrain=np.zeros((N,T)) #changement des labels
    for t in range(T):
        annote=lambda x:modifie_label_gaussien(x,qualite_annotateur_gaussien[t])
        annote=np.vectorize(annote)
        ytrain[:,t]=annote(ztrain)
    return xtrain,ytrain,ztrain


'''def get_annotations(l,datax,datay):
    """ on extrait les données qui ont une recus par les labelleurs une série d'annotations donné par l ; ex :[1,-1,1,1] pour 4 annotateurs """
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    idx = np.random.permutation(range(len(tmpy)))
    return tmpx[idx,:],tmpy[idx]'''

def traite_zero(x):
    if x==0:
        return 1
    else:
        return x

class LearnCrowd:
    def __init__(self, T, N, d):
        self.alpha = np.zeros((1,d)) # Poids des dimensions
        self.beta = 0
        self.w = np.zeros((d,T)) # Poids des labelleurs
        self.gamma = np.zeros((1,T))

    def z_cond_x(self, X, alpha, beta):
        """renvoie la matrice z_cond_x : proba que le vrai label soit 0 ou 1 sachant la donnée (Rlog) (indépendant du modèle Bernoulli/Gaussien"""
        z_cond_x = np.zeros((X.shape[0],2))
        sigm = lambda x:  1/(1+np.exp(x))
        sigm=np.vectorize(sigm)
        tmpsigm = sigm(-np.dot(X,alpha.T)-beta)
        z_cond_x[:,1] = tmpsigm.ravel()
        z_cond_x[:,0] = 1-z_cond_x[:,1]
        return z_cond_x

    def expects_labels_Bernoulli(self, X, Y, alpha, beta, gamma, w):
        """calcule les probas des labels z pour chaque donnée -> taille (N,2)
        en multipliant sur tous les labelleurs : la proba que le vrai label soit 0 ou 1 et que le label du labelleur Yt soit celui obtenu sachant la donnée i
        donnée dans le modèle de Bernoulli par la matrice y_z_cond_x = y_cond_z_cond_x * z_cond_x = y_cond_z * z_cond_x"""

        N = X.shape[0]
        T = Y.shape[1]

        eta = 1/(1+np.exp(-np.dot(X,w)-gamma)) # Taille : N,T

        #proba cond du label Yt du labelleur t pour la donnée i sachant le vrai label 0 ou 1 (Bernoulli)
        y_cond_z = np.zeros((N,T,2))

        for t in range(T):
          y_cond_z[:,t,0] = ((1-eta[:,t])**np.abs(Y[:,t]))*(eta[:,t]**(1-np.abs(Y[:,t])))
          y_cond_z[:,t,1] = ((1-eta[:,t])**np.abs(Y[:,t]-1))*(eta[:,t]**(1-np.abs(Y[:,t]-1)))
          sum = y_cond_z[:,t,0] + y_cond_z[:,t,1]
          traite = lambda x:traite_zero(x)
          traite = np.vectorize(traite)
          sum = traite(sum)
          y_cond_z[:,t,0] = np.multiply(y_cond_z[:,t,0],1/sum)
          y_cond_z[:,t,1] = np.multiply(y_cond_z[:,t,1],1/sum)
          #print("A",y_cond_z[:,t,:])

        #hyp de base que l'on pourra prendre pour simplifier neta[i,t]=rlog(i,t)=neta[t]
        #cet hyp revient à donner une proba constante de se tromper pour le labelleur t quelque soit la donnée
        #il faudrait alors rajouter un self.neta=np.zeros(1,T) au init pour le modèle de Bernoulli

        #print("B",self.z_cond_x(X, alpha, beta))
        results = np.multiply(np.prod(y_cond_z,axis=1),self.z_cond_x(X, alpha, beta))

        s = results[:,0] + results[:,1]
        traite = lambda x:traite_zero(x)
        traite = np.vectorize(traite)
        s = traite(s)
        results[:,0] = np.multiply(results[:,0],1/s)
        results[:,1] = np.multiply(results[:,1],1/s)

        return results

    def expects_labels_Gaussian(self, X, Y, alpha, beta, gamma, w):
        """calcule les probas des labels z pour chaque donnée -> taille (N,2)
        en multipliant sur tous les labelleurs : la proba que le vrai label soit 0 ou 1 et que le label du labelleur Yt soit celui obtenu sachant la donnée i
        donnée dans le modèle Gaussien par la matrice y_z_cond_x = y_cond_z_cond_x * z_cond_x"""

        #proba cond du label Yt du labelleur t pour la donnée i sachant le vrai label 0 ou 1 (Bernoulli)
        y_cond_z_cond_x = np.zeros((N,T,2))
        norm=lambda x,mu,sigma:1/(sqrt(2*np.pi)*sigma)*np.exp(-pow((x-mu),2)/pow(sigma,2))
        norm=np.vectorize(norm)
        sigma = 1/(1+np.exp(-X.dot(w)-gamma)) # Taille : N,T
        for t in range(T):
            y_cond_z_cond_x[:,t,0] = norm(Y[:,t],0,sigma[:,t])
            y_cond_z_cond_x[:,t,1] = norm(Y[:,t],1,sigma[:,t])
        return np.multiply(np.prod(y_cond_z_cond_x,axis=1),self.z_cond_x(X, alpha, beta))

    def likelihood(self, Pt, X, Y, model, alpha, beta, gamma, w):
        """renvoit la log-vraisemblance totale du modèle calculée grâce à Pt matrice des probas des vrais labels Z calculés à l'E-step,
        model=Bernoulli ou Gaussian"""

        #Pt_to_expect c'est ce qu'il y a à l'intérieur de l'espérance
        if model=="Bernoulli":
            Pt_to_expect = self.expects_labels_Bernoulli(X, Y, alpha, beta, gamma, w)
        if model=="Gaussian":
            Pt_to_expect = self.expects_labels_Gaussian(X, Y, alpha, beta, gamma, w)
        #la log-vraisemblance correspond à la somme des entropies des vrais labels sur toutes les données ??
        #on calcule l'espérance de Pt_to_expect avec les Pt calculés à l'expectation
        return np.sum(np.multiply(Pt,np.log(Pt_to_expect+pow(10,-10))))

    def grad_likelihood(self, Pt, X, Y, model, alpha, beta, gamma, w):
        """Returns the partial derivatives of likelihood according to
        alpha, beta, gamma and w
        model=Bernoulli ou Gaussian"""

        deltaPt = Pt[:,1]-Pt[:,0]
        deltaPt = deltaPt.reshape((deltaPt.shape[0],1))

        tmp_exp = np.exp(-np.dot(X,alpha.T)-beta)

        # mat = np.multiply(np.multiply(Pt[:,1].reshape((-1,1)),tmp_exp) - Pt[:,0].reshape((-1,1)),1/(1+tmp_exp))
        # grad_lh_alpha = np.sum(mat.reshape((-1,1))*X,axis=0) #Taille 1,d
        # grad_lh_beta = np.sum(mat.reshape((-1,1)),axis=0)  #Taille 1

        grad_lh_alpha = np.sum(np.multiply(deltaPt,np.multiply(tmp_exp,1/((1+tmp_exp)**2))).reshape((-1,1))*X,axis=0) #Taille 1,d
        grad_lh_beta = np.sum(np.multiply(deltaPt,np.multiply(tmp_exp,1/((1+tmp_exp)**2))))  #Taille 1

        tmp_exp = np.exp(-np.dot(X,w)-gamma) # Taille : N,T
        etasigma = 1/(1+tmp_exp)
        if (model=="Gaussian"):
            etasigma = etasigma +pow(10,-6)
        grad_etasigma_gamma = np.multiply(etasigma,1-etasigma) # Taille : N,T

        (N,d)=np.shape(X)
        (N,T)=np.shape(Y)
        grad_etasigma_w = np.zeros((N,d,T))
        for t in range(0,T):
            grad_etasigma_w[:,:,t]=np.multiply(grad_etasigma_gamma[:,t].reshape(N,1),X) #taille N,d,T

        #if (model=="Bernoulli"):
        grad_lh_etasigma = - np.multiply(deltaPt,((-1)**Y)) # Taille : N,T
        #elif (model=="Gaussian"):
        #grad_lh_etasigma = (Y**2-Pt[:,0]*(2*Y-1))/(etasigma**3) - 1/etasigma # Taille : N,T

        grad_lh_gamma = np.sum(np.multiply(grad_lh_etasigma,grad_etasigma_gamma),axis=0) #Taille 1,T
        grad_lh_w =  np.sum(np.multiply(np.repeat(grad_lh_etasigma[:,np.newaxis,:],d,axis=1),grad_etasigma_w),axis=0) #Taille d,T
        
        # "Zippage" des gradients en un grand vecteur
        Grad = np.concatenate((grad_lh_alpha.ravel(),np.array([grad_lh_beta]).ravel(),grad_lh_gamma.ravel(),grad_lh_w.ravel()),axis=0)
        return Grad.ravel()

    def fit(self, X, Y, model="Bernoulli", eps = 10**(-5)):
        N = X.shape[0]
        d = X.shape[1]
        T = Y.shape[1]

        #EM Algorithm

        #Initialization

        self.alpha = np.ones((1,d))
        self.beta = 1
        alphaNew = np.ones((1,d))
        alphaNew[:,0]=0*alphaNew[:,0];
        alphaNew[:,1]=0*alphaNew[:,1];
        betaNew = 0
        wNew = np.random.rand(d,T)
        gammaNew = np.random.rand(1,T)

        cpt_iter=0
        LH = []
        #if model=="Bernoulli":
        Pt = self.expects_labels_Bernoulli(X, Y, self.alpha, self.beta, self.gamma, self.w)
        normGrad = np.linalg.norm(-self.grad_likelihood(Pt, X, Y, model, self.alpha, self.beta, self.gamma, self.w))

        while (np.linalg.norm(self.alpha-alphaNew)**2 + (self.beta-betaNew)**2 >= eps or cpt_iter < 10):
            print("NORM",np.linalg.norm(self.alpha-alphaNew)**2 + (self.beta-betaNew)**2 )
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
            #elif model=="Gaussian":
            #Pt = self.expects_labels_Gaussian(X, Y, self.alpha, self.beta, self.gamma, self.w)

            #print(Pt)
            # Maximization (M-step)

            # "Zippage" de self.alpha, self.beta, self.gamma, self.w en un grand vecteur Teta

            def BFGSfunc(vect):
                #print("Vect",vect)
                # print("Vraisemblance : ",-self.likelihood(Pt, X, Y, model, vect[0:d].reshape((1,d)), float(vect[d:d+1]), vect[d+1:d+1+T].reshape((1,T)), vect[d+1+T:d+1+T+d*T].reshape((d,T))))
                return -self.likelihood(Pt, X, Y, model, vect[0:d].reshape((1,d)), float(vect[d:d+1]), vect[d+1:d+1+T].reshape((1,T)), vect[d+1+T:d+1+T+d*T].reshape((d,T)))

            def BFGSJac(vect):
                #print("Gradient",-self.grad_likelihood(Pt, X, Y, model, vect[0:d].reshape((1,d)), float(vect[d:d+1]), vect[d+1:d+1+T].reshape((1,T)), vect[d+1+T:d+1+T+d*T].reshape((d,T))))
                return -self.grad_likelihood(Pt, X, Y, model, vect[0:d].reshape((1,d)), float(vect[d:d+1]), vect[d+1:d+1+T].reshape((1,T)), vect[d+1+T:d+1+T+d*T].reshape((d,T)))

            Teta_init = np.concatenate((alphaNew.ravel(),np.array([betaNew]).ravel(),gammaNew.ravel(),wNew.ravel()),axis=0) #initial guess
            #rappels des tailles de alpha, beta, gamma, w : (1,d), 1, (1,T), (d,T)
            LH.append(BFGSfunc(Teta_init))
            result = minimize(BFGSfunc, Teta_init, method='BFGS', jac = BFGSJac,\
                              options={'gtol': 1e-10, 'disp': True, 'maxiter': 5000})
            print(result.message)
            # print("Optimal solution :")
            # print(result.x)

            # "Dézippage" de Teta solution en self.alpha, self.beta, self.gamma, self.w
            # To Update new vectors :

            Teta = result.x
            alphaNew = Teta[0:d].reshape((1,d))
            betaNew = float(Teta[d:d+1])
            gammaNew = Teta[d+1:d+1+T].reshape((1,T))
            wNew = Teta[d+1+T:d+1+T+d*T].reshape((d,T))

        self.alpha = alphaNew
        self.beta = betaNew
        self.gamma = gammaNew
        self.w = wNew
        print("############ Test BFGS #######################")
        Teta_f = np.concatenate((self.alpha.ravel(),np.array([self.beta]).ravel(),self.gamma.ravel(),self.w.ravel()),axis=0)
        print(Teta_init-Teta_f)
        print("")
        plt.plot(np.linspace(1,cpt_iter,cpt_iter),LH)
        plt.show()

    def predict(self, X, seuil):
        #on prédit les vrais labels à partir des données X

        tmp_exp = np.exp(-np.dot(X,self.alpha.T)-self.beta) # Taille : N
        proba_class_1 = 1/(1+tmp_exp)
        labels_predicted = proba_class_1 > seuil
        return labels_predicted.ravel()

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

    '''def get_eps(self):
        return
    def loss(self,data,y):
        return
    def loss_g(self,data,y):
        return'''

#POUT REGRESSION LOGISTIQUE SUR LA GROUND TRUTH

def optimize(fonc,dfonc,xinit,eps,max_iter):
    x_histo=[]
    f_histo=[]
    grad_histo=[]
    iter=0
    x_histo.append(xinit)
    f_histo.append(fonc(x_histo[0]))
    grad_histo.append(dfonc(x_histo[0]))
    while (iter<max_iter):
        iter+=1
        x_new=x_histo[iter-1]-eps*dfonc(x_histo[iter-1])
        x_histo.append(x_new)
        f_histo.append(fonc(x_new))
        grad_histo.append(dfonc(x_new))
    x_histo=np.array(x_histo)
    f_histo=np.array(f_histo)
    grad_histo=np.array(grad_histo)
    return (x_histo,f_histo,grad_histo)

def reglog(w,data,label):
    #Renvoit le cout en w pour la régression logistique
    ps=np.multiply(np.reshape(label,(-1,1)),np.dot(data,np.transpose(w)))
    rlog=lambda x: math.log(1/(1+math.exp(-x)))
    rlog=np.vectorize(rlog)
    return -np.mean(rlog(ps))

def grad_reglog(w,data,label):
    #Renvoit le gradient du cout en w pour la régression logistique
    (n,d)=np.shape(data)
    label=np.reshape(label,(-1,1))
    ps=np.multiply(label,np.dot(data,np.transpose(w)))
    sig=lambda x:1/(1+math.exp(-x))
    sig=np.vectorize(sig)
    tmp=np.multiply(np.multiply(label,np.exp(-ps)),sig(ps))
    return -np.mean(np.multiply(np.tile(tmp,(1,d)),data),axis=0)

def signe(x):
    if x>=0:
        return 1
    return 0

class Classifier_Binary():
    def __init__(self):
        self.w=-1 #vecteur des poids

    def score(self,data,labels):
        return (self.predict(data)==labels).mean()

class Classifier_RegLog(Classifier_Binary):
    def __init__(self):
        super().__init__()

    def predict(self,data,seuil):
        (n,d)=np.shape(data)
        col_id=np.reshape((np.ones(n)),(n,1)) # vecteur colonne de 1
        data=np.concatenate((col_id,data),axis=1)  # matrice de "design"

        predictions=np.dot(data,np.transpose(self.w))
        #sign = lambda x:signe(x)
        #sign=np.vectorize(sign)
        sigm = lambda x:1/(1+np.exp(-x))
        sigm = np.vectorize(sigm)
        predictions = sigm(predictions) > seuil
        convertfloat = lambda x:float(x)
        convertfloat = np.vectorize(convertfloat)
        predictions = convertfloat(predictions)
        return predictions.ravel()
        #return sign(predictions).flatten()

    def fit(self,data,labels,eps,nb_iter,affiche=False):
        labels=2*labels-1
        (n,d)=np.shape(data)
        col_id=np.reshape((np.ones(n)),(n,1)) # vecteur colonne de 1
        data=np.concatenate((col_id,data),axis=1)  # matrice de "design"

        cost = lambda x : reglog(x,data,labels)
        grad_cost = lambda x : grad_reglog(x,data,labels)
        winit = np.zeros((1,d+1))
        (w_histo,cost_histo,grad_cost_histo) = optimize(cost,grad_cost,winit,eps,nb_iter)
        self.w = w_histo[-1]

        if affiche:
            plt.figure()
            plt.plot(list(range(nb_iter+1)),cost_histo,color="yellow")
            plt.title("Coûts pour la classification par régression logistique \
            \n au fil des itérations de l'algorithme du gradient")
            plt.xlabel("Nombre d'itérations")
            plt.show()

    def score(self,data,labels):
        return super().score(data,labels)


class MajorityVoting:
    def __init__(self):
        pass
    def fit(self, X, Y):
        pass
    def predict(self, Y,seuil):
        (N,T)=np.shape(Y)
        predictions = np.sum(Y,axis=1)/T
        predictions = predictions > seuil
        return predictions
    def score(self, Y, Z,seuil):
        return np.mean(self.predict(Y,seuil)==Z)

def TP_FP(predictions,truth):
    tmp1 = (predictions==1)&(truth==1);
    TP = np.sum(tmp1==True)/np.sum(truth==1)
    tmp2 = (predictions==1)&(truth==0);
    FP = np.sum(tmp2==True)/np.sum(truth==0)
    return TP,FP

def trace_ROC(N,T,d,modele,qualite_annotateurs,generateur,noise_truth):
    """ trace les courbes ROC pour le modèle learning from the Crowd avec predict(X),
    compare avec un classifieur sur Z, compare avec un majority voting,
    place aussi les FP et TP de chaque labelleur // ground truth"""

    print("Rappel des paramètres")
    print("Nombre de données générées : ", N)
    print("Nombre de dimensions des données générées : ", d)
    print("Nombre d'annotateurs : ", T)
    print("Modèle : ", modele)
    print("Probabilités de succès des annotateurs : ", qualite_annotateurs)
    print("")

    print("Génération des données")

    xtrain, ytrain,ztrain = generateur(N,T,qualite_annotateurs,noise_truth)
    xtest, ytest,ztest = generateur(N,T,qualite_annotateurs,noise_truth)

    #     print("Données d'entrainement")
    #     print("Données X : ", xtrain)
    #     print("Vrai Labels : ", ztrain)
    #     print("Labels données par les annotateurs Y : ", ytrain)
    #     print("")
    #print("ytrain size :", ytrain.shape)
    #print("xtrain size :", xtrain.shape)
    #plot_data(xtrain,ztrain)
    #plt.title("Données et labels de départ (bruitées)")
    #plt.show()

    if modele=="Bernoulli":
        print(ytrain[:,0])
        plot_data(xtrain,ytrain[:,0])
        plt.title("Annotations d'un labelleur")
        plt.show()

    #print("TP et FP de chaque annotateur")
    TP_train_labelleurs=[]
    FP_train_labelleurs=[]
    for t in range(T):
        tp,fp = TP_FP(ytrain[:,t],ztrain)
        TP_train_labelleurs.append(tp)
        FP_train_labelleurs.append(fp)

    #print("TP et FP de chaque annotateur")
    TP_test_labelleurs=[]
    FP_test_labelleurs=[]
    for t in range(T):
        tp,fp = TP_FP(ytest[:,t],ztest)
        TP_test_labelleurs.append(tp)
        FP_test_labelleurs.append(fp)

    #print("Données de test")
    '''print("Données X : ", xtest)
    print("Vrai Labels : ", ztest)
    #print("Labels données par les annotateurs Y : ", ytest)
    print("")'''

    S = LearnCrowd(T,N,d)
    M = MajorityVoting()
    C = Classifier_RegLog()

    print("Apprentissage")
    S.fit(xtrain,ytrain)
    C.fit(xtrain,ztrain,0.005,1000,affiche=False)

    print("####################################################")
    print("Tests à l'aide de X")

    seuils=[0.1*k for k in range(11)]

    TP_train_crowd=[]
    FP_train_crowd=[]
    TP_test_crowd=[]
    FP_test_crowd=[]

    TP_train_majority=[]
    FP_train_majority=[]
    TP_test_majority=[]
    FP_test_majority=[]

    TP_train_class=[]
    FP_train_class=[]
    TP_test_class=[]
    FP_test_class=[]

    for s in seuils:
        tp,fp=TP_FP(S.predict(xtrain,s),ztrain)
        TP_train_crowd.append(tp)
        FP_train_crowd.append(fp)
        tp,fp=TP_FP(S.predict(xtest,s),ztest)
        TP_test_crowd.append(tp)
        FP_test_crowd.append(fp)

        tp,fp=TP_FP(M.predict(ytrain,s),ztrain)
        TP_train_majority.append(tp)
        FP_train_majority.append(fp)
        tp,fp=TP_FP(M.predict(ytest,s),ztest)
        TP_test_majority.append(tp)
        FP_test_majority.append(fp)

        tp,fp=TP_FP(C.predict(xtrain,s),ztrain)
        TP_train_class.append(tp)
        FP_train_class.append(fp)
        tp,fp=TP_FP(C.predict(xtest,s),ztest)
        TP_test_class.append(tp)
        FP_test_class.append(fp)

    plt.scatter(FP_train_labelleurs,TP_train_labelleurs)
    plt.plot(FP_train_crowd,TP_train_crowd,color="blue")
    plt.plot(FP_train_majority,TP_train_majority,color="red")
    plt.plot(FP_train_class,TP_train_class,color="yellow")
    plt.title("ROC trainset crowdlearning (bleu), majority voting (rouge), classifier truth (yellow)")
    plt.show()

    plt.scatter(FP_test_labelleurs,TP_test_labelleurs)
    plt.plot(FP_test_crowd,TP_test_crowd,color="blue")
    plt.plot(FP_test_majority,TP_test_majority,color="red")
    plt.plot(FP_test_class,TP_test_class,color="yellow")
    plt.title("ROC testset crowdlearning (bleu), majority voting (rouge), classifier truth (yellow)")
    plt.show()

'''
A=np.array([1,1,1,0])
B=np.array([1,0,0,0])
print(TP_FP(A,B))
'''

def LearnfromtheCrowd(N,T, d, modele,qualite_annotateurs, generateur,noise_truth=0):
    print("Rappel des paramètres")
    print("Nombre de données générées : ", N)
    print("Nombre de dimensions des données générées : ", d)
    print("Nombre d'annotateurs : ", T)
    print("Modèle : ", modele)
    print("Probabilités de succès des annotateurs : ", qualite_annotateurs)
    print("")

    print("Génération des données")

    xtrain, ytrain,ztrain = generateur(N,T,qualite_annotateurs,noise_truth)
    xtest, ytest,ztest = generateur(N,T,qualite_annotateurs,noise_truth)

    '''
    print("Données d'entrainement")
    print("Données X : ", xtrain)
    print("Vrai Labels : ", ztrain)
    print("Labels données par les annotateurs Y : ", ytrain)
    print("")
    #print("ytrain size :", ytrain.shape)
    #print("xtrain size :", xtrain.shape)
    plot_data(xtrain,ztrain)
    plt.title("Données et labels de départ (bruitées)")
    plt.show()
    '''
    print("Vrais labels: ", ztrain)

    # if modele=="Bernoulli":
    #     plot_data(xtrain,ytrain[:,0])
    #     plt.title("Annotations d'un labelleur")
    #     plt.show()

    #print("Données de test")
    '''print("Données X : ", xtest)
    print("Vrai Labels : ", ztest)
    #print("Labels données par les annotateurs Y : ", ytest)
    print("")'''

    S = LearnCrowd(T,N,d)

    print("Apprentissage")
    S.fit(xtrain,ytrain)

    print("alpha",S.alpha)
    print("beta",S.beta)
    print("gamma",S.gamma)
    print("w",S.w)


    print("####################################################")
    print("Test à l'aide de X")

    print("Performances sur les données d'entrainement : ")
    strain=S.score(xtrain,ztrain,0.5)
    print("Score en Train : ",strain)

    #plot_frontiere(xtest,S.predict(xtest),step=50) C'est XTEST ? Pas ZTEST ?
    # plot_data(xtrain,S.predict(xtrain,0.5))
    # plt.title("Prédictions finales sur le Train après crowdlearning")
    # plt.show()


    stest=S.score(xtest,ztest,0.5)
    print("Performances sur les données de test : ")
    print("Score en Test : ", stest)

    '''
    #plot_frontiere(xtest,S.predict(xtest),step=50)
    plot_data(xtest,S.predict(xtest,0.5))
    plt.title("Prédictions finales sur le Test après crowdlearning")
    plt.show()
    '''

    M=MajorityVoting()

    strain_majority=M.score(ytrain,ztrain,0.5)
    print("Score en Train (Majority Voting) : ", strain_majority)
    print("")

    # plot_data(xtrain,M.predict(ytrain,0.5))
    # plt.title("Prédictions finales sur le Train après MajorityVoting")
    # plt.show()

    '''
    stest_majority= M.score(ytest,ztest)
    print("Performances sur les données de test : ")
    print("Score en Test (Majority Voting) : ", stest_majority)

    plot_data(xtest,M.predict(ytest))
    plt.title("Prédictions finales sur le Test après MajorityVoting")
    plt.show()
    '''

    '''
    print("####################################################")
    print("Test à l'aide de X et Y")

    print("Performances sur les données d'entrainement : ")
    print("Score en Train : ", S.scoreV3(xtrain,ytrain,ztrain,0.5,modele))

    plot_data(xtrain,S.predictV3(xtrain,ytrain,0.5,modele))
    plt.title("Prédictions finales sur le Train après crowdlearning")
    plt.show()

    #print(S.predictV3(xtrain,ytrain,0.5,modele),ztrain)

    print("Performances sur les données de test : ")
    print("Score en Test : ", S.scoreV3(xtest,ytest,ztest,0.5,modele))

    plot_data(xtest,S.predictV3(xtest,ytest,0.5,modele))
    plt.show()


    print("####################################################")

    M=MajorityVoting()
    plt.title("Prédictions finales sur le Test")

    print("Test à l'aide de Y, Comparaison avec le Majority Voting")

    print("Performances sur les données d'entrainement : ")
    print("Score en Train : ", S.scoreV2(ytrain,ztrain,xtrain,modele))
    print("Score en Train (Majority Voting) : ", M.score(ytrain,ztrain))
    print("")

    #plot_frontiere(xtest,S.predict(xtest),step=50) C'est XTEST ? Pas ZTEST ?
    plot_data(xtrain,S.predictV2(ytrain,xtrain,modele))
    plt.title("Prédictions finales sur le Train après crowdlearning")
    plt.show()

    plot_data(xtrain,M.predict(ytrain))
    plt.title("Prédictions finales sur le Train après MajorityVoting")
    plt.show()

    print("Performances sur les données de test : ")
    print("Score en Test : ", S.scoreV2(ytest,ztest,xtrain,modele))
    print("Score en Test (Majority Voting) : ", M.score(ytest,ztest))

    #plot_frontiere(xtest,S.predict(xtest),step=50)
    plot_data(xtest,S.predictV2(ytest,xtrain,modele))
    plt.title("Prédictions finales sur le Test après  crowdlearning")
    plt.show()

    plot_data(xtest,M.predict(ytest))
    plt.title("Prédictions finales sur le Test après MajorityVoting")
    plt.show()
    '''

    return strain,strain_majority
    #stest,stest_majority

def learn_cas_unif_x():
    N = 100 #nb données
    T = 10 #nb annotateurs
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth=0.05 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
    modele= "Bernoulli"

    qualite_annotateurs_Bernoulli=[0.96]*T #Proba que l'annotateur ait raison
    LearnfromtheCrowd(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth)

    #qualite_annotateurs_Bernoulli=[0.1] #Proba que l'annotateur ait raison dans la zone 1 de données (1-la valeur dans la zone 2)
    #LearnfromtheCrowd(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli_xdepend,noise_truth)

    #trace_ROC(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth)

def learn_cas_depend_x():

    special_params=[0.1*i for i in range(6)]
    score_train_crowd=[]
    score_train_majority=[]
    #score_test_crowd=[]
    #score_test_majority=[]

    N = 100 #nb données
    T = 3 #nb annotateurs
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth=0 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
    modele= "Bernoulli"

    #qualite_annotateurs_Bernoulli=[0.9,0.6,0.6] #Proba que l'annotateur ait raison
    #LearnfromtheCrowd(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth)

    for s in special_params:
        qualite_annotateurs_Bernoulli=[[0.5+special_params,1-special_params],[1-special_params,0.5 + special_params],[0.5+special_params,0.5+special_params]] #Proba que l'annotateur ait raison dans la zone 1 de données (1-la valeur dans la zone 2)
        scores = LearnfromtheCrowd(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli_xdepend,noise_truth)
        score_train_crowd.append(score[0])
        score_train_majority.append(score[1])

    plt.plot(special_params,score_train_crowd)
    plt.plot(special_params,score_train_majority)
    plt.title("Performances entre annotateurs spécialisés et annotateurs non spécialisés")
    plt.show()
    #trace_ROC(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth)

def courbe_precision_annoteur(nombre_annoteur):
    M=MajorityVoting()
    vecteur_x=[]
    vecteur_y_crowd=[]
    vecteur_y_majority=[]
    for i in range(1,nombre_annoteur):
        vecteur_x.append(i)
        noise_truth=0.5 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
        modele= "Bernoulli"
        qualite_annotateurs_Bernoulli=np.random.uniform(0,1,(1,i))
        (score,ytest)=LearnfromtheCrowd(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth)
        vecteur_y_crowd.append(S.score)
        vecteur_y_majority.append(M.predict(ytest))
    plt.plot(vecteur_x,vecteur_y_crowd)
    plt.plot(vecteur_x,vecteur_y_crowd)
    plt.show()

def regularisation():
    lbd=[pow(10,-k) for k in range(-5,5)]
    error_train=[]
    error_test=[]

    N = 100 #nb données
    T = 10 #nb annotateurs
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth=0.5 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
    modele= "Bernoulli"

    qualite_annotateurs_Bernoulli=qualite_annoteur=[(0.6,0.6)]*T #Proba que l'annotateur ait raison

    for l in lbd:
        results=LearnfromtheCrowd(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth,lbd=l)
        error_train.append(results[0])
        error_test.append(results[2])

    plt.plot(list(range(-5,5)),error_train,color="blue")
    plt.plot(list(range(-5,5)),error_test,color="red")
    plt.xlabel("Paramètre de régularisation")
    plt.ylabel("Erreurs")
    plt.title("Erreurs d'entrainement (bleu) et de test (rouge) \n en fonction du paramètre de régularisation")
    plt.show()
'''
def nb_donnees():
    Nb=linspace(,,)
    S = LearnCrowd2(T,N,d)
    M=MajorityVoting()
    error_crowd_train=[]
    error_crowd_test=[]
    error_majority_train=[]
    error_majority_test=[]

    for N in range(Nb):
        xtrain=x[:N]
        ytrain=y[:N]
        ztrain=z[:N]

        xtest=x[N+1:]
        ytest=y[N+1:]
        ztest=z[N+1:]

        S.fit(xtrain,ytrain)

        error_crowd_train.append(S.score(xtrain,ztrain,0.5))
        error_crowd_test.append(S.score(xtest,ztest,0.5))
        error_majority_train.append(M.score(ytrain,ztrain,0.5))
        error_majority_test.append(M.score(ytest,ztest,0.5))

    plt.plot(Nb,error_crowd_train,color="blue")
    plt.plot(Nb,error_crowd_test,color="red")
    plt.plot(Nb,error_majority_train,color="yellow")
    plt.plot(Nb,error_majority_test,color="green")
    plt.xlabel("Nombre de données d'apprentissage")
    plt.ylabel("Erreurs")
    plt.title("Erreurs sur les ensembles d'apprentissage (bleu,jaune) et de test (rouge,vert)  \n formant une partition (CrowdLearning,MajorityVoting)")
'''

#courbe_precision_annoteur(100)1

learn_cas_unif_x()
