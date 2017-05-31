# Modules importés

import numpy as np
import matplotlib.pyplot as plt
from tools import *
from scipy.optimize import minimize
from random import gauss

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

#autre modèle

#qualite annoteur

qualite_annoteur=[(0.9,0.1),(0.9,0.1),(0.9,0.1),(0.9,0.1)]

def modifie_label(label,qualite_annoteur):
    (alpha,beta)=qualite_annoteur
    res=-1 #label initili
    if label==1:
        #on simule avec proba alpha
        valeur_proba=np.random.uniform(0,1)
        if(valeur_proba>alpha):
            res=0
        else:
            res=1
    if label==0:
        valeur_proba=np.random.uniform(0,1)
        if(valeur_proba>beta):
            res=1
        else:
            res=0
    return res

def generation_Bernouilli(N,T,qualite_annotateur_Bernoulli,noise_truth):
    """retourne en xtrain les données de dimension 2, en ytrain les annotations, en ztrain les vrais labels
    avec pour qualite_annotateurs une liste contenant les probabilités de succès de chaque annotateur
    noise_truth est le bruit de l'attribution des vrais labels gaussiens sur les données"""
    xtrain,ztrain = gen_arti(nbex=N,data_type=0,epsilon=noise_truth) #vrai labels non bruités
    ztrain=(ztrain+1)/2
    ytrain=np.zeros((N,T)) #changement des labels
    for t in range(T):
        annote=lambda x:modifie_label(x,qualite_annotateur_Bernoulli[t])
        annote=np.vectorize(annote)
        ytrain[:,t]=annote(ztrain)
    return xtrain,ytrain,ztrain

def sigmoide(x):
    return 1/(1+np.exp(-x))


def calcule_ai(alpha,yi):
    T=yi.shape[0]
    res=1
    for t in range(T):
        res=res*(alpha[0,t]**(yi[t]))*((1-alpha[0,t])**(1-yi[t]))
    return res

def calcule_bi(beta,yi):
    T=yi.shape[0]
    res=1
    for t in range(T):
        res=res*(beta[0,t]**(yi[t]))*((1-beta[0,t])**(1-yi[t]))
    return res


def gradient_modele(X,mu,w):
    N=np.shape(X)
    (d,p)=np.shape(w)
    vecteur_grad=np.zeros((d,1))
    vecteur_w_x=-np.dot(X,w)
    vecteur_addition=mu+vecteur_w_x
    Matrice_gradient=np.multiply(vecteur_addition,X)

    vecteur_res=np.sum(Matrice_gradient,axis=0)
    vecteur_res=vecteur_res.reshape((d,1))

    return vecteur_res

def hessien_modele(X,w):
    N=X.shape[0]
    (d,p)=np.shape(w)
    vecteur_w_x=np.dot(X,w)
    Matrice_Hessienne=np.zeros((d,d))
    for i in range(N):
        xi=X[i,:].reshape((d,1))
        Matrice_Hessienne+=-vecteur_w_x[i,0]*(1-vecteur_w_x[i,0])*np.dot(xi,xi.T)
    return Matrice_Hessienne

class LearnCrowd2:
    def __init__(self, T, N, d):
        self.alpha = np.zeros((1,T)) # sensitivité des annoteur
        self.beta = np.zeros((1,T)) #specificité
        self.w = np.ones((d,1)) # Poids pour le modèle (indépendant des annoteur)
        self.y_trouve=np.zeros((N,1))

    def likelihood(self,X, Y, alpha, beta, w,vrai_y):
        N = X.shape[0]
        T = Y.shape[1]
        print("shape X")
        print(X.shape)
        print("shape w")
        print(w.shape)
        print("shape de X.W")
        print(np.dot(X,w).shape)
        proba = 1/(1+np.exp(-X.dot(w)))+1.0**(-10) # vecteur des pi
        print("shape proba")
        print(proba.shape)
        proba=proba.reshape((N,1))
        Vecteur_A=np.zeros((N,1))
        vecteur_B=np.zeros((N,1))
        for i in range(N):
            Vecteur_A=calcule_ai(alpha,Y[i,:])
            Vecteur_B=calcule_bi(beta,Y[i,:])

        log_pi=np.log(proba)
        log_un_pi=np.log(1-proba)
        vrai_un_yi=(1-vrai_y)

        vecteur_a1=np.multiply(Vecteur_A,log_pi)
        vecteur_a1=np.multiply(vecteur_a1,vrai_y)

        vecteur_b1=np.multiply(Vecteur_B,log_un_pi)
        vecteur_b1=np.multiply(vecteur_B,vrai_un_yi)

        log_res=np.sum(vecteur_a1)+np.sum(vecteur_b1)

        return log_res

    def fit(self, X, Y, model="Bernoulli", eps = 10**(-2)):
        N = X.shape[0]
        d = X.shape[1]
        T = Y.shape[1]

        #EM Algorithm

        #Initialization

        alpha=np.zeros((1,T))
        alphaNew=0.5*np.ones((1,T))
        beta=np.zeros((1,T))
        betaNew=0.5*np.ones((1,T))
        #initialisation des mu avec le majority voting
        mu_inter=(np.sum(Y,axis=1)/T)

        mu=mu_inter.reshape((N,1))
        w=np.ones((d,1))

        nombre_iteration=[0]
        self.liste_em=[]
        cpt_iter=0
        self.norme_gradient=[]
        valeur_EM_before=1
        valeur_EM=1000
        while (np.linalg.norm(alpha-alphaNew)**2 + np.linalg.norm(beta-betaNew)**2 >= eps and (cpt_iter<1000)):
        #while(abs((valeur_EM-valeur_EM_before)/valeur_EM_before)>=eps and (cpt_iter<=1000)):
            print("la valeur du critère de l' EM avant l EM ")
            print(np.linalg.norm(alpha-alphaNew)**2 + np.linalg.norm(beta-betaNew)**2 )
            cpt_iter+=1
            print("ITERATION N°",cpt_iter)

            alpha = alphaNew
            beta= betaNew


            # Expectation (E-step)
            if(nombre_iteration[-1]!=0):
                    valeur_EM=self.likelihood(X,Y,alphaNew,betaNew,w,mu)
                    self.liste_em.append(valeur_EM)
                    #on change les valeurs des mu
                    proba=1/(1+np.exp(-np.dot(X,w)))+1.0**(-10)

                    for i in range(N):
                        ai=calcule_ai(alphaNew,Y[i,:])
                        bi=calcule_bi(betaNew,Y[i,:])
                        pi=proba[i,0]
                        mu[i,0]=ai*pi/(ai*pi+bi*(1-pi))



            # Maximization (M-step)

            #calcul de alpha
            alpha_inter=np.dot(Y.T,mu)/(np.sum(mu))+1.0**(-10)
            alphaNew=alpha_inter.reshape((1,T))
            #calcul de beta
            mu_inter=1-mu
            y_inter=np.ones((N,T))-Y
            beta_inter=np.dot(y_inter.T,mu_inter)/(np.sum(mu_inter))+1.0**(-10)
            betaNew=beta_inter.reshape((1,T))

            #Optimisation de w

            gad_w_avant=gradient_modele(X,mu,w)
            self.norme_gradient.append(np.log(np.linalg.norm(gad_w_avant)))
            Hess_w=hessien_modele(X,w)

            #boucle de travail
            w_avant=w
            norme_gradient=np.linalg.norm(gad_w_avant)
            matrice_inverse=np.linalg.inv(Hess_w)

            w=w-np.dot(matrice_inverse,gad_w_avant)
            norme_gradient=np.linalg.norm(gad_w_avant)
            iteration=0
            while(norme_gradient>eps and (iteration<1000)):
                gad_w=gradient_modele(X,mu,w)
                matrice_inverse=np.linalg.inv(Hess_w)
                w=w-np.dot(matrice_inverse,gad_w)
                norme_gradient=np.linalg.norm(gad_w)
                iteration+=1
            print("la valeur du critère de l' EM après l EM ")
            print(np.linalg.norm(alpha-alphaNew)**2 + np.linalg.norm(beta-betaNew)**2 )
            nombre_iteration.append(nombre_iteration[-1]+1)

        self.alpha=alphaNew
        self.beta=betaNew
        self.w=w
        print("valeur de w")
        print(w)

        plt.plot(nombre_iteration[2:],self.liste_em)
        plt.show()
        plt.legend("variation de l'Em avec les itérations")


    def predict(self, X):
        #on prédit les vrais labels à partir des données X

        tmp_exp = np.exp(-np.dot(X,self.w)) # Taille : N
        proba_class_1 = 1/(1+tmp_exp)
        labels_predicted = proba_class_1 > 0.5
        #labels_predicted = 2*labels_predicted-1
        return labels_predicted.ravel()




    def score(self, X, Z):
        # On connaît la vérité terrain
        return np.mean(self.predict(X)==Z)





def LearnfromtheCrowd2(N,T, d, modele,qualite_annotateurs, generateur,noise_truth=0):
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
    print("ytrain size :", ytrain.shape)
    print("xtrain size :", xtrain.shape)
    plot_data(xtrain,ztrain)
    plt.title("Données et labels de départ (gaussiennes bruitées)")
    plt.show()

    if modele=="Bernoulli":
        plot_data(xtrain,ytrain[:,0])
        plt.title("Annotations d'un labelleur")
        plt.show()

    print("Données de test")
    '''print("Données X : ", xtest)
    print("Vrai Labels : ", ztest)
    #print("Labels données par les annotateurs Y : ", ytest)
    print("")'''

    S = LearnCrowd2(T,N,d)

    print("Apprentissage")
    S.fit(xtrain,ytrain)

    print("Performances sur les données d'entrainement : ")
    print("Score en Train : ", S.score(xtrain,ztrain))
    print("")


    #plot_frontiere(xtest,S.predict(xtest),step=50) C'est XTEST ? Pas ZTEST ?
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


N = 100 #nb données
T = 25 #nb annotateurs
d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
noise_truth=0.1 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
modele= "Bernoulli"
qualite_annoteur=[(0.9,0.9)]*T
LearnfromtheCrowd2(N,T,d,modele,qualite_annoteur,generation_Bernouilli,noise_truth)
