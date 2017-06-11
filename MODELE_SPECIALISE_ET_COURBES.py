





#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


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
