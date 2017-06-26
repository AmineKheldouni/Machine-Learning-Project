# Modules importĂŠs

import numpy as np
import matplotlib.pyplot as plt
from tools import *
from scipy.optimize import minimize
from random import gauss



def sigmoide(x):
    return 1/(1+np.exp(-x))

sigmoide = np.vectorize(sigmoide)

def calcule_ai(alpha, Y):
    T = Y.shape[0]
    ai = np.prod(np.multiply(alpha**Y, (1-alpha)**(1-Y)), axis = 1).reshape((T,1))
    return ai
    

def calcule_bi(beta, Y):
    T=Y.shape[0]
    return np.prod(np.multiply(beta**(1-Y), (1-beta)**(Y)), axis = 1).reshape((T,1))
   

#calcul du gradient
def gradient_modele(X, mu, w):
    N=np.shape(X)[0]
    d=np.shape(X)[1]
    vecteur_w_x = sigmoide(np.dot(X,w))
    vecteur_w_x=vecteur_w_x.reshape((N,1))
    mu_inter=mu.reshape((-1,1))
    vecteur_addition = mu_inter-vecteur_w_x
    Matrice_gradient = np.multiply(vecteur_addition,X)
    vecteur_res=np.sum(Matrice_gradient,axis=0)
    vecteur_res=vecteur_res.reshape((d,1))
    #remplacer vecteur_res-2*w pour avoir ridge regression
    return vecteur_res

#calcul de la hessienne
def hessien_modele(X,w):
    N=X.shape[0]
    d = X.shape[1]
    vecteur_w_x = sigmoide(np.dot(X,w))
    vecteur_w_x=vecteur_w_x.reshape((-1,1))
    H = -np.multiply(np.multiply(vecteur_w_x,(1-vecteur_w_x)),X).T.dot(X)
    return H

def descente_gradient(w_init,X,mu,gradient_fonction,hessienne_fonction):
    vecteur_gradient=gradient_fonction(X,mu,w_init)
    Hessienne=hessienne_fonction(X,w_init)
    w1=w_init
    w0=w_init+10
    nombre_iteration=1
    while(np.linalg.norm(w1-w0)>1.0**(-4) and nombre_iteration<100):
        w0=w1
        coeff=0.5/np.sqrt(nombre_iteration)
        vecteur_gradient=gradient_fonction(X,mu,w1)
        Hessienne=hessienne_fonction(X,w1)
        #if(np.linalg.det(Hessienne)==0):
            #Hessienne=0.005*np.eye(X.shape[1],X.shape[1])
        w1=w1-coeff*np.linalg.inv(Hessienne).dot(vecteur_gradient)
        nombre_iteration+=1
    return w1

#classe du classifieur
class LearnCrowd2:
    def __init__(self, T, N, d, l=0):
        self.alpha = np.zeros((1,T)) # sensitivite des annotateurs
        self.beta = np.zeros((1,T)) #specificite des annotateurs�
        self.w = np.ones((d,1)) # Poids pour le modele 
        self.y_trouve=np.zeros((N,1))
        #lambda de ridge regression
        self.lb = l
        #ne sert a rien ici sauf pour faire marcher les fonctions
        self.gamma="rien"
        

    def likelihood(self, X, Y, alpha, beta, w, vrai_y):
        

        N = X.shape[0]
        T = Y.shape[1]

        p_i = 1/(1+np.exp(-X.dot(w)))
        p_i = p_i.reshape((N,1))
        a_i = calcule_ai(alpha, Y)
        b_i = calcule_bi(alpha, Y)
        pi_ai=np.multiply(p_i,a_i)
        pi_bi=np.multiply((1-p_i),b_i)

       
        vrai_un_yi = (1-vrai_y)

        
        v1=np.multiply(vrai_y,np.log(pi_ai+1.0**(-10)))
        v2=np.multiply(vrai_un_yi,np.log(pi_bi+1.0**(-10)))

    
        log_res=np.sum(v1+v2)

        return log_res

    def fit(self, X, Y,epsGrad=10**(-5), model="Bernoulli", eps = 10**(-3),max_iter=100, draw_convergence=False):
        max_iteration=100
        N = X.shape[0]
        d = X.shape[1]
        T = Y.shape[1]

        #EM Algorithm

        #Initialization
        #initialisation des mu avec le majority voting
        mu_inter=np.mean(Y,axis=1)
        mu = mu_inter.reshape((N,1))
        #Initialisation aleatoire du premier w
        w=0.1 * np.random.rand(d)
        w=w.reshape((d,1))
        
        nombre_iteration_EM=0
        liste_valeur_EM=[]
        valeur_EM_avant=-1000000
        valeur_EM=0
        while(nombre_iteration_EM<max_iteration and (valeur_EM-valeur_EM_avant)):
            #Etape  Maximization
            #calcul des alpha
            alpha_inter = np.dot(Y.T, mu) / ((np.sum(mu))+1.0**(-15) )
            alphaNew=alpha_inter.reshape((1,T))

            #calcul de beta
            mu_inter = 1-mu
            y_inter = np.ones((N,T))-Y
            beta_inter = np.dot(y_inter.T,mu_inter) / ((np.sum(mu_inter))+1.0**(-15))
            betaNew = beta_inter.reshape((1,T))

            #optimisation de w
            w=descente_gradient(w,X,mu,gradient_modele,hessien_modele)


            #Etape Expectation, mise a jour des mu
            pi = sigmoide(np.dot(X,w))
            pi=pi.reshape((-1,1))
            ai = calcule_ai(alphaNew, Y)
            bi = calcule_bi(betaNew, Y)
            numerateur=np.multiply(ai,pi)
            denominateur=np.multiply(ai,pi)+np.multiply(bi,1-pi)
            mu = np.multiply(numerateur,1/denominateur)
            nombre_iteration_EM+=1

            #valeur de l'EM a cette etape
            valeur_EM_avant=valeur_EM
            valeur_EM=self.likelihood(X, Y, alphaNew, betaNew, w, mu)
            liste_valeur_EM.append(valeur_EM)
            

        #conservation des parametres appris pour le classifieur
        self.alpha = alphaNew
        self.beta = betaNew
        self.w = w
        if draw_convergence:
            #affichage de la courbe des variations de l'EM et des estimations des annotateurs
            plt.plot([i for i in range(len(liste_valeur_EM))],liste_valeur_EM,linewidth=2.0, linestyle="-", label=('courbe de la variation de la log-vraisemblance lors de l EM'))
            plt.xlabel("nombre d'iteration de l'EM")
            plt.ylabel("Valeur de la log-vraisemblance")
            plt.title("Evolution de la log-vraisemblance dans l'EM")
            plt.show()
            #plt.legend(bbox_to_anchor=(1, 0), bbox_transform=plt.gcf().transFigure)
        print("valeur alpha (sensitivites des annotateurs)")
        print(alphaNew)
        print("valeur beta (specificites des annotateurs)")
        print(betaNew)



    def predict(self, X,seuil):
        proba_class_1=sigmoide(X.dot(self.w))
        labels_predicted = proba_class_1 > seuil
        bool2float = lambda x:float(x)
        bool2float=np.vectorize(bool2float)
        return bool2float(labels_predicted).ravel()




    def score(self, X, Z,seuil):
         #On compare a la verite terrain
        return np.mean(self.predict(X,seuil)==Z)


    def debug(self):
        print("w : \n")
        print(self.w)
        print("alpha : \n")
        print(self.alpha)
        print("beta : \n")
        print(self.beta)
