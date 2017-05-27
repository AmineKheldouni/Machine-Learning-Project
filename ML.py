# Modules importés

import numpy as np
import matplotlib.pyplot as plt
from tools import *
from scipy.optimize import minimize

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
    cols,marks = ["red", "blue","green", "orange", "black", "cyan"],[".","+","*","o","x","^"]
    if labels is None:
        plt.scatter(data[:,0],data[:,1],marker="x")
        return
    for i,l in enumerate(sorted(list(set(labels.flatten())))):
        plt.scatter(data[labels==l,0],data[labels==l,1],c=cols[i],marker=marks[i])

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
        label_res=-label
    return label_res        

def generation_Bernoulli(N,T,qualite_annotateur_Bernoulli,noise_truth):
    """retourne en xtrain les données de dimension 2, en ytrain les annotations, en ztrain les vrais labels
    avec pour qualite_annotateurs une liste contenant les probabilités de succès de chaque annotateur
    noise_truth est le bruit de l'attribution des vrais labels gaussiens sur les données"""
    xtrain,ztrain = gen_arti(nbex=N,data_type=0,epsilon=noise_truth) #vrai labels non bruités
    ytrain=np.zeros((N,T)) #changement des labels 
    for t in range(T):
        annote=lambda x:modifie_label_Bernoulli(x,qualite_annotateur_Bernoulli[t])
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


class LearnCrowd:
    def __init__(self, T, N, d):
        self.alpha = np.zeros((1,d)) # Poids des dimensions
        self.beta = 0
        self.w = np.zeros((d,T)) # Poids des labelleurs
        self.gamma = np.zeros((1,T))
    
    def z_cond_x(self, X, alpha, beta):  
        """renvoie la matrice z_cond_x : proba que le vrai label soit 0 ou 1 sachant la donnée (Rlog) (indépendant du modèle Bernoulli/Gaussien"""
        z_cond_x = np.zeros((X.shape[0],2))
        sigm = lambda x:  1/(1+np.exp(-x))
        sigm=np.vectorize(sigm)
        tmpsigm = sigm(-np.dot(X,alpha.T)-beta)
        z_cond_x[:,0] = list(tmpsigm)
        z_cond_x[:,1] = list(1-np.array(z_cond_x[:,0]))
        return z_cond_x
    
    def expects_labels_Bernoulli(self, X, Y, alpha, beta, gamma, w):
        """calcule les probas des labels z pour chaque donnée -> taille (N,2) 
        en multipliant sur tous les labelleurs : la proba que le vrai label soit 0 ou 1 et que le label du labelleur Yt soit celui obtenu sachant la donnée i 
        donnée dans le modèle de Bernoulli par la matrice y_z_cond_x = y_cond_z_cond_x * z_cond_x = y_cond_z * z_cond_x"""

        N = X.shape[0]
        T = Y.shape[1]
        
        eta = 1/(1+np.exp(-X.dot(w)-gamma)) # Taille : N,T

        #proba cond du label Yt du labelleur t pour la donnée i sachant le vrai label 0 ou 1 (Bernoulli)
        y_cond_z = np.zeros((N,T,2))

        for t in range(T):
          y_cond_z[:,t,0] = ((1-eta[:,t])**np.abs(Y[:,t]))*(eta[:,t]**(1-np.abs(Y[:,t])))
          y_cond_z[:,t,1] = ((1-eta[:,t])**np.abs(Y[:,t]-1))*(eta[:,t]**(1-np.abs(Y[:,t]-1)))

        #hyp de base que l'on pourra prendre pour simplifier neta[i,t]=rlog(i,t)=neta[t]
        #cet hyp revient à donner une proba constante de se tromper pour le labelleur t quelque soit la donnée
        #il faudrait alors rajouter un self.neta=np.zeros(1,T) au init pour le modèle de Bernoulli
        
        return np.multiply(np.prod(y_cond_z,axis=1),self.z_cond_x(X, alpha, beta))
    
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
        """renvoit la log-vraissemblance totale du modèle calculée grâce à Pt matrice des probas des vrais labels Z calculés à l'E-step,
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
        Pt matrice des probas des vrais labels Z calculés à l'E-step,
        model=Bernoulli ou Gaussian"""

        deltaPt = Pt[:,1]-Pt[:,0] 
        deltaPt = deltaPt.reshape((deltaPt.shape[0],1))
        
        tmp_exp = np.exp(-X.dot(alpha.T)-beta)
        grad_lh_alpha = np.sum(np.multiply(deltaPt,np.multiply(tmp_exp,1/((1+tmp_exp)**2))).reshape((-1,1))*X,axis=0) #Taille 1,d
        grad_lh_beta = np.sum(np.multiply(deltaPt,np.multiply(tmp_exp,1/((1+tmp_exp)**2))))  #Taille 1

        tmp_exp = np.exp(-X.dot(w)-gamma) # Taille : N,T
        etasigma = 1/(1+tmp_exp) 
        if (model=="Gaussian"):
            etasigma = etasigma +pow(10,-6)
        grad_etasigma_gamma = etasigma*(1-etasigma) # Taille : N,T

        (N,d)=np.shape(X)
        (N,T)=np.shape(Y)
        grad_etasigma_w = np.zeros((N,d,T))
        for t in range(0,T):
            grad_etasigma_w[:,:,t]=grad_etasigma_gamma[:,t].reshape(N,1)*X #taille N,d,T
        
        if (model=="Bernoulli"):
            grad_lh_etasigma = - deltaPt * ((-1)**Y) # Taille : N,T
        elif (model=="Gaussian"):
            grad_lh_etasigma = (Y**2-Pt[:,0]*(2*Y-1))/(etasigma**3) - 1/etasigma # Taille : N,T

        grad_lh_gamma = np.sum(grad_lh_etasigma*grad_etasigma_gamma,axis=0) #Taille 1,T
        grad_lh_w = np.sum(np.multiply(np.repeat(grad_lh_etasigma[:,np.newaxis,:],2,axis=1),grad_etasigma_w),axis=0) #Taille d,T

        # "Zippage" des gradients en un grand vecteur

        #print(grad_lh_alpha)
        #print(grad_lh_beta)
        #print(grad_lh_gamma)
        #print(grad_lh_w)
        
        Grad = np.concatenate((grad_lh_alpha.ravel(),np.array([grad_lh_beta]).ravel(),grad_lh_gamma.ravel(),grad_lh_w.ravel()),axis=0) 
        #print("Grad",Grad)
        
        return Grad
    
    def fit(self, X, Y, model="Bernoulli", eps = 10**(-2)):
        N = X.shape[0]
        d = X.shape[1]
        T = Y.shape[1]

        #EM Algorithm

        #Initialization

        self.alpha = np.zeros((1,d))
        self.beta = 0
        alphaNew = np.ones((1,d))
        betaNew = 1
        wNew = np.random.rand(d,T)
        gammaNew = np.random.rand(1,T)

        cpt_iter=0

        while (np.linalg.norm(self.alpha-alphaNew)**2 + (self.beta-betaNew)**2 >= eps):
            print(np.linalg.norm(self.alpha-alphaNew)**2 + (self.beta-betaNew)**2 )
            cpt_iter+=1
            print("ITERATION N°",cpt_iter)
            
            self.alpha = alphaNew
            self.beta = betaNew
            self.gamma = gammaNew
            self.w = wNew
 
            # Expectation (E-step)

            if model=="Bernoulli":
               Pt = self.expects_labels_Bernoulli(X, Y, self.alpha, self.beta, self.gamma, self.w)
            elif model=="Gaussian":
               Pt = self.expects_labels_Gaussian(X, Y, self.alpha, self.beta, self.gamma, self.w)

            # Maximization (M-step)

            # "Zippage" de self.alpha, self.beta, self.gamma, self.w en un grand vecteur Teta

            Teta_init = np.concatenate((self.alpha.ravel(),np.array([self.beta]).ravel(),self.gamma.ravel(),self.w.ravel()),axis=0) #initial guess

            #rappels des tailles de alpha, beta, gamma, w : (1,d), 1, (1,T), (d,T)
            BFGSfunc = lambda vect : -self.likelihood(Pt, X, Y, model, vect[0:d].reshape((1,d)), float(vect[d:d+1]), vect[d+1:d+1+T].reshape((1,T)), vect[d+1+T:d+1+T+d*T].reshape((d,T)))
            BFGSJac = lambda vect : -self.grad_likelihood(Pt, X, Y, model, vect[0:d].reshape((1,d)), float(vect[d:d+1]), vect[d+1:d+1+T].reshape((1,T)), vect[d+1+T:d+1+T+d*T].reshape((d,T)))
 
            #Optimisation avec BFGS

            result = minimize(BFGSfunc, Teta_init, method='BFGS', jac = BFGSJac, options={'gtol': 1e-6, 'disp': True, 'maxiter': 1000})           
            #print(result.message)
            #print("Optimal solution :")
            #print(result.x)

            # "Dézippage" de Teta solution en self.alpha, self.beta, self.gamma, self.w
            # To Update new vectors :

            Teta = result.x
            alphaNew = Teta[0:d].reshape((1,d))
            betaNew = float(Teta[d:d+1])
            gammaNew = Teta[d+1:d+1+T].reshape((1,T))
            wNew = Teta[d+1+T:d+1+T+d*T].reshape((d,T))
            
        self.alpha = alphaNew
        self.beta = betaNew
        self.w = wNew
        self.gamma = gammaNew
        
    def predict(self, X):
        #on prédit les vrais labels à partir des données X

        tmp_exp = np.exp(-X.dot(self.alpha.T)-self.beta) # Taille : N
        proba_class_1 = 1/(1+tmp_exp)
        labels_predicted = proba_class_1 > 0.5
        labels_predicted = 2*labels_predicted-1
        return labels_predicted.ravel()
          
    def predictV2(self, Ytest, Xtrain, modeltrain):
        #Xtrain sont les données qui ont été utilisées pendant l'apprentissage (pas besoin de Ytrain ???, seul les poids appris suffisent ??)
        #modeltrain est le modèle utilisé pendant l'apprentissage
        #on prédit les vrais labels à partir des annotations de Ytest

        #On veut calculer non pas les probas des labels Z mais des labels Z sachant les annotations de Ytest pour chaque labelleur
        #Ainsi on va appliquer les mêmes méthodes "expects_labels_" mais que sur les données de Xtrain qui ont été annotées par un type d'annotation donné 


        #Ou en fixant tous les Y à la Ytest comme suit ?
  
        N=Ytest.shape[0]
        labels_predicted=np.zeros((N,1))
        
        for i in range(N):
            if modeltrain=="Bernoulli":
               Pt = self.expects_labels_Bernoulli(Xtrain, Ytest[i,:].flatten().tile((Xtrain.shape[0],1)), self.alpha, self.beta, self.gamma, self.w)
            if modeltrain=="Gaussian":
               Pt = self.expects_labels_Gaussian(Xtrain, Ytest[i,:].flatten().tile((Xtrain.shape[0],1)), self.alpha, self.beta, self.gamma, self.w)
            probas_class=np.mean(Pt,axis=0)
            labels_predicted[i,1] = probas_class[:,1] > probas_class[:,0]
            
        labels_predicted = 2*labels_predicted-1
        return labels_predicted.ravel()

    def score(self, X, Z):
        # On connaît la vérité terrain
        return np.mean(self.predict(X)==Z)
    
    '''def get_eps(self):
        return
    def loss(self,data,y):
        return
    def loss_g(self,data,y):
        return'''

def LearnfromtheCrowd(N,T,modele,qualite_annotateurs,noise_truth=0):
	print("Rappel des paramètres")
	print("Nombre de données générées : ", N)
	print("Nombre de dimensions des données générées : ", d)
	print("Nombre d'annotateurs : ", T)
	print("Modèle : ", modele)
	if modele=="Bernoulli":
	   print("Probabilités de succès des annotateurs : ", qualite_annotateurs)
	print("")

	print("Génération des données")

	xtrain, ytrain,ztrain = generation_Bernoulli(N,T,qualite_annotateurs,noise_truth)
	xtest, ytest,ztest = generation_Bernoulli(N,T,qualite_annotateurs,noise_truth)

	print("Données d'entrainement")
	print("Données X : ", xtrain)
	print("Vrai Labels : ", ztrain)
	print("Labels données par les annotateurs Y : ", ytrain)
	print("")

	plt.figure()
	plot_data(xtrain,ztrain)
	plt.title("Données et labels de départ (gaussiennes bruitées)")
	plt.show()

	plt.figure()
	plot_data(xtrain,ytrain[:,0])
	plt.title("Annotations d'un labelleur")
	plt.show()

	print("Données de test")
	print("Données X : ", xtest)
	print("Vrai Labels : ", ztest)
	#print("Labels données par les annotateurs Y : ", ytest)
	print("")

	S = LearnCrowd(T,N,d)

	print("Apprentissage")
	S.fit(xtrain,ytrain)

	print("Performances sur les données d'entrainement : ")
	print("Score en Train : ", S.score(xtrain,ztrain))
	print("")

	plt.figure()
	#plot_frontiere(xtest,S.predict(xtest),step=50)
	plot_data(xtrain,S.predict(xtrain))
	plt.title("Prédictions finales sur le Train après crowdlearning")
	plt.show()

	print("Performances sur les données de test : ")
	print("Score en Test : ", S.score(xtest,ztest))

	plt.figure()
	#plot_frontiere(xtest,S.predict(xtest),step=50)
	plot_data(xtest,S.predict(xtest))
	plt.title("Prédictions finales sur le Test")
	plt.show()


N = 50 #nb données
T = 20 #nb annotateurs
d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
noise_truth=0.5 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
modele= "Bernoulli"
qualite_annotateurs_Bernoulli=[0.6]*T #Proba que l'annotateur ait raison
LearnfromtheCrowd(N,T,modele,qualite_annotateurs_Bernoulli,noise_truth)
