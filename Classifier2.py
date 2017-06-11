from importModule import *

def sigmoide(x):
    return 1/(1+np.exp(-x))

sigmoide = np.vectorize(sigmoide)

def calcule_ai(alpha, Y):
    T = Y.shape[0]
    res=1
    ai = np.prod(np.multiply(alpha**Y, (1-alpha)**(1-Y)), axis = 1).reshape((T,1))
    return ai
    # for t in range(T):
    #     res = res*(alpha[0,t]**(Y[t]))*((1-alpha[0,t])**(1-Y[t]))
    # return res

def calcule_bi(beta, Y):
    T=Y.shape[0]
    res=1
    return np.prod(np.multiply(beta**(1-Y), (1-beta)**(Y)), axis = 1).reshape((T,1))
    # for t in range(T):
    #     res=res*(beta[0,t]**(1-yi[t]))*((1-beta[0,t])**(yi[t]))
    # return res


def gradient_modele_w(X, mu, w, l):
    N=np.shape(X)
    d=np.shape(w)[0]
    #vecteur_grad = np.zeros((d,1))
    w=w.reshape((d,1))
    vecteur_w_x = sigmoide(np.dot(X,w))
    vecteur_addition = -mu+vecteur_w_x+1
    Matrice_gradient = np.multiply(vecteur_addition,X)
    vecteur_res=np.sum(Matrice_gradient,axis=0)
    vecteur_res=vecteur_res.reshape((d,1))
    #remplacer vecteur_res-2*w pour avoir ridge regression
    return np.ndarray.flatten(vecteur_res)



def gradient_modele_amine(X, mu, w, l):
    N=np.shape(X)
    d=np.shape(w)[0]
    #vecteur_grad = np.zeros((d,1))
    w=w.reshape((d,1))
    vecteur_w_x = sigmoide(np.dot(X,w))
    vecteur_carre=vecteur_w_x**(2)
    vecteur_addition=np.multiply(np.multiply(1/vecteur_w_x,mu)+np.multiply(1-mu,1/(1-vecteur_w_x)),np.multiply(np.exp(-np.dot(X,w)),vecteur_carre))
    Matrice_gradient = np.multiply(vecteur_addition,X)
    vecteur_res=np.sum(Matrice_gradient,axis=0)
    vecteur_res=vecteur_res.reshape((d,1))
    #remplacer vecteur_res-2*w pour avoir ridge regression
    return np.ndarray.flatten(vecteur_res)

def hessien_modele(X,w,l):
    N=X.shape[0]
    d = w.shape[0]
    vecteur_w_x = sigmoide(np.dot(X,w))
    Matrice_Hessienne=np.zeros((d,d))
    H = -np.multiply(np.multiply(vecteur_w_x,(1-vecteur_w_x)),X).T.dot(X)
    return H
    # for i in range(N):
    #     xi=X[i,:].reshape((d,1))
    #     Matrice_Hessienne -= vecteur_w_x[i,0]*(1-vecteur_w_x[i,0])*np.dot(xi,xi.T)
    # return Matrice_Hessienne

class LearnCrowd2:
    def __init__(self, T, N, d, l=0):
        self.alpha = np.zeros((1,T)) # sensitivitĂŠ des annoteur
        self.beta = np.zeros((1,T)) #specificitĂŠ
        self.w = np.ones((d,1)) # Poids pour le modĂ¨le (indĂŠpendant des annoteur)
        self.y_trouve=np.zeros((N,1))
        self.lb = l

    def likelihood(self, X, Y, alpha, beta, w, vrai_y):
        # vrai_y = mu_i

        N = X.shape[0]
        T = Y.shape[1]

        p_i = 1/(1+np.exp(-X.dot(w)))+1.0**(-10)
        p_i = p_i.reshape((N,1))
        a_i = calcule_ai(alpha, Y)
        b_i = calcule_bi(alpha, Y)
        pi_ai=np.multiply(p_i+1.0**(-10),a_i)
        pi_bi=np.multiply((1-p_i+1.0**(-10)),b_i)

        #log_pi = np.log(p_i)
        #log_un_pi = np.log(1+1.0**(-10)-p_i)
        vrai_un_yi = (1-vrai_y)

        #a1 = np.multiply(a_i, log_pi)
        #a1 = np.multiply(a1, vrai_y)
        v1=np.multiply(vrai_y,np.log(pi_ai+1.0**(-10)))
        v2=np.multiply(vrai_un_yi,np.log(pi_bi+1.0**(-10)))

        #b1 = np.multiply(b_i, log_un_pi)
        #b1 = np.multiply(b1, vrai_un_yi)

        #log_res = np.sum(a1+b1)
        log_res=np.sum(v1+v2)

        return log_res

    def fit(self, X, Y, model="Bernoulli", eps = 10**(-3), max_iter=200):
        N = X.shape[0]
        d = X.shape[1]
        T = Y.shape[1]

        #EM Algorithm

        #Initialization

        alpha=np.ones((1,T))
        alphaNew=0.5*np.ones((1,T))
        beta=np.ones((1,T))
        betaNew=0.5*np.ones((1,T))
        #initialisation des mu avec le majority voting
        mu_inter=(np.sum(Y,axis=1)/T)

        mu = mu_inter.reshape((N,1))
        w = np.ones((d,1))

        # nombre_iteration = 0
        cpt_iter=1
        # self.liste_em=[]
        # self.norme_gradient=[]
        # valeur_EM_before=1
        # valeur_EM=1000
        liste_valeur_EM=[]
        while (np.linalg.norm(alpha-alphaNew)**2 + np.linalg.norm(beta-betaNew)**2 >= eps and (cpt_iter<max_iter)):
        #while(abs((valeur_EM-valeur_EM_before)/valeur_EM_before)>=eps and (cpt_iter<=1000)):

            print("ITERATION NÂ°",cpt_iter)
            print(" ... \n")

            alpha = alphaNew
            beta= betaNew


            valeur_EM = self.likelihood(X,Y,alphaNew,betaNew,w,mu)
            liste_valeur_EM.append(valeur_EM)
            print("valeur EM")
            print(valeur_EM)
            # Expectation (E-step)
            if(cpt_iter!=0):
                    # self.liste_em.append(valeur_EM)
                    #on change les valeurs des mu
                    pi = sigmoide(np.dot(X,w)) + 1.0**(-10)
                    #
                    # for i in range(N):
                    #     ai = calcule_ai(alphaNew,Y[i,:])
                    #     bi = calcule_bi(betaNew,Y[i,:])
                    #     pi = p_i[i,0]
                    #     mu[i,0] = ai*pi/(ai*pi+bi*(1-pi))
                    ai = calcule_ai(alphaNew, Y)
                    bi = calcule_bi(betaNew, Y)

                    mu = np.multiply(ai, pi) / (ai * pi + bi * (1-pi))


            cpt_iter+=1
            # Maximization (M-step)


            #Optimisation de w
             # "Zippage" de self.w
            #autre optimisation
            w_inter=w.reshape((d,1))
            grad_w=gradient_modele_w(X,mu,w_inter,0)
            nombre_iteration=0
            while(np.linalg.norm(grad_w)>1.0**(-4) and nombre_iteration<1000):
                Hess=hessien_modele(X,w_inter,0)
                if(abs(np.linalg.det(Hess))<1.0**(-5)):
                    Hess=np.eye(d,d)
                direction=-gradient_modele_amine(X,mu,w_inter,1)
                direction=direction.reshape((d,1))
                w_inter=w_inter-0.005*np.dot(np.linalg.inv(Hess),direction)
                grad_w=direction
                nombre_iteration+=1

            def BFGSfunc(vect):
                return -self.likelihood(X, Y, alphaNew, betaNew,vect,mu)

            def BFGSJac(vect):
                d=np.shape(vect)[0]
                return -gradient_modele_amine(X, mu, vect, l=0)

            Teta_init = w
            Teta_init=np.ndarray.flatten(w)

            w=w_inter


            print("W TROUVE !")
            print(w)

            #calcul de alpha
            alpha_inter = np.dot(Y.T, mu) / ((np.sum(mu)) + 1.0**(-10))
            alphaNew=alpha_inter.reshape((1,T))
            #calcul de beta
            mu_inter = 1-mu
            y_inter = np.ones((N,T))-Y
            beta_inter = np.dot(y_inter.T,mu_inter) / ((np.sum(mu_inter))+1.0**(-10))
            betaNew = beta_inter.reshape((1,T))


        self.alpha = alphaNew
        self.beta = betaNew
        self.w = w
        print("valeur de w")
        print(w)
        #
        plt.plot([i for i in range(len(liste_valeur_EM))],liste_valeur_EM)
        plt.show()
        plt.legend("variation de l'Em avec les itĂŠrations")

    def predict(self, X, seuil):
        #on prĂŠdit les vrais labels Ă  partir des donnĂŠes X
        proba_class_1 = sigmoide(np.dot(X,self.w))
        labels_predicted = proba_class_1 > seuil
        #labels_predicted = 2*labels_predicted-1
        return labels_predicted.ravel()

    def score(self, X, Z, seuil):
        # On connaĂŽt la vĂŠritĂŠ terrain
        return np.mean(self.predict(X, seuil)==Z)


    def debug(self):
        print("w : \n")
        print(self.w)
        print("alpha : \n")
        print(self.alpha)
        print("beta : \n")
        print(self.beta)
