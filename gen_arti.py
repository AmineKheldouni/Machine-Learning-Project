from importModule import *

#I. BLOC GENERATION DE X ET Z
#UTILISER DATA_TYPE = 0 et DATA_TYPE =1

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
        xpos=np.vstack((np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),int(nbex//4)),np.random.multivariate_normal([-centerx,centerx],np.diag([sigma,sigma]),int(nbex/4))))
        xneg=np.vstack((np.random.multivariate_normal([centerx,-centerx],np.diag([sigma,sigma]),int(nbex//4)),np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),int(nbex/4))))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex//2),-np.ones(int(nbex//2))))
    if data_type==3:
        xpos=np.vstack((np.random.multivariate_normal([0,centerx],np.diag([sigma,sigma]),int(nbex//4)),np.random.multivariate_normal([centerx,0],np.diag([sigma,sigma]),int(nbex/4))))
        xneg=np.vstack((np.random.multivariate_normal([-centerx,0],np.diag([sigma,sigma]),int(nbex//4)),np.random.multivariate_normal([0,-centerx],np.diag([sigma,sigma]),int(nbex/4))))
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


#II. BLOC GENERATION DE Y

#ADAPTE AU MODELE 1
def modifie_label_Bernoulli(label,proba):
    """proba = (proba d'avoir juste si label 0,proba d'avoir juste si label 1) :
    modifie le vrai label 0 en choisissant l'autre avec une probabilité 1-proba[0]
    modifie le vrai label 1 en choisissant l'autre avec une probabilité 1-proba[1]"""
    valeur_proba=np.random.uniform(0,1)
    label_res=label
    if((label==0)and(valeur_proba>=proba[0])):
        label_res=1-label
    if((label==1)and(valeur_proba>=proba[1])):
        label_res=1-label
    return label_res


def generation_Bernoulli(N,T,qualite_annotateur_Bernoulli,noise_truth,data_type=0):
    """retourne en xtrain les données de dimension 2, en ytrain les annotations, en ztrain les vrais labels
    avec pour qualite_annotateurs une liste contenant les probabilités de succès de chaque annotateur TP,TN
    noise_truth est le bruit de l'attribution des vrais labels gaussiens sur les données"""
    xtrain,ztrain = gen_arti(nbex=N,data_type=data_type,epsilon=noise_truth) #vrai labels non bruités
    ztrain=(ztrain+1)/2
    ytrain=np.zeros((N,T)) #changement des labels
    for t in range(T):
        annote=lambda x:modifie_label_Bernoulli(x,qualite_annotateur_Bernoulli[t])
        annote=np.vectorize(annote)
        ytrain[:,t]=annote(ztrain)
    return xtrain,ytrain,ztrain


'''
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
'''

#ADAPTE AU MODELE 2
def modifie_label_Bernoulli_xdepend(data,label,proba):
    """proba = (proba d'avoir juste si donnee dans zone 0,proba d'avoir juste si donnee dans zone 1)
    modifie le vrai label en choisissant l'autre avec une probabilité 1-proba[0] (dans la zone 0)
    et 1-proba[1] (dans la zone 1)
    où zone 0 = donnes 2D de x positif, zone 1 = donnes 2D de x negatif"""
    valeur_proba=np.random.uniform(0,1)
    label_res=label
    if data[0]>=0: #donnees à x négatif (groupe 1 de données)
        if(valeur_proba>=proba[0]):
           label_res=1-label
    else: #données à x positif (groupe 2 de données)
        if(valeur_proba>=proba[1]):
           label_res=1-label
    return label_res

def generation_Bernoulli_xdepend(N,T,qualite_annotateur_Bernoulli,noise_truth,data_type=0):
    """retourne en xtrain les données de dimension 2, en ytrain les annotations, en ztrain les vrais labels
    avec pour qualite_annotateur_Bernoulli les probabilités de succès de chaque annotateur dans chaque zone"""
    xtrain,ztrain = gen_arti(nbex=N,data_type=data_type,epsilon=noise_truth) #vrai labels non bruités
    ztrain=(ztrain+1)/2
    ytrain=np.zeros((N,T)) #changement des labels
    for t in range(T):
        annote=lambda idx_data,z:modifie_label_Bernoulli_xdepend(xtrain[idx_data,:],z,qualite_annotateur_Bernoulli[t])
        annote=np.vectorize(annote)
        ytrain[:,t]=annote(list(range(np.shape(xtrain)[0])),ztrain)
    return xtrain,ytrain,ztrain

def modifie_label_Bernoulli_Order(label,proba,nu,S):
    valeur_proba=np.random.uniform(0,1)
    label_res=label
    l = [nu*S,nu*(1-S),(1-nu)]
    l.sort()

    if (l[0]>=valeur_proba):
        if (l[0]==nu*S):
            label_res = 1
        elif (l[0]==nu*(1-S)):
            label_res = 0
        else:
            label_res = modifie_label_Bernoulli(label,proba)
    elif (l[1]+l[0]>=valeur_proba):
        if (l[1]==nu*S):
            label_res = 1
        elif (l[1]==nu*(1-S)):
            label_res = 0
        else:
            label_res = modifie_label_Bernoulli(label,proba)
    else:
        if (l[2]==nu*S):
            label_res = 1
        elif (l[2]==nu*(1-S)):
            label_res = 0
        else:
            label_res = modifie_label_Bernoulli(label,proba)
    return label_res

def generation_Bernoulli_Order(N,T,qualite_annotateur_Bernoulli,noise_truth,nu,S,data_type=0):
    """retourne en xtrain les données de dimension 2, en ytrain les annotations, en ztrain les vrais labels
    avec pour qualite_annotateurs une liste contenant les probabilités de succès de chaque annotateur TP,TN
    noise_truth est le bruit de l'attribution des vrais labels gaussiens sur les données"""
    xtrain,ztrain = gen_arti(nbex=N,data_type=data_type,epsilon=noise_truth) #vrai labels non bruités
    ztrain=(ztrain+1)/2
    ytrain=np.zeros((N,T)) #changement des labels
    for t in range(T):
        annote=lambda x:modifie_label_Bernoulli_Order(x,qualite_annotateur_Bernoulli[t],nu[t],S[t])
        annote=np.vectorize(annote)
        ytrain[:,t]=annote(ztrain)
    return xtrain,ytrain,ztrain

#III. GENERATION DE X,Y,Z

def genere(N,T,d,modele,qualite_annotateurs,generateur,noise_truth,affiche=False, nu=None, S=None, data_type=0):
    print("Rappel des paramètres")
    print("Nombre de données générées : ", N)
    print("Nombre de dimensions des données générées : ", d)
    print("Nombre d'annotateurs : ", T)
    print("Modèle : ", modele)
    print("Probabilités de succès des annotateurs : ", qualite_annotateurs)
    print("")

    print("Génération des données")
    if nu==None and S==None:
        xtrain, ytrain,ztrain = generateur(N,T,qualite_annotateurs,noise_truth,data_type=data_type)
        xtest, ytest,ztest = generateur(N,T,qualite_annotateurs,noise_truth,data_type=data_type)
    else:
        xtrain, ytrain,ztrain = generateur(N,T,qualite_annotateurs,noise_truth,nu=nu,S=S,data_type=data_type)
        xtest, ytest,ztest = generateur(N,T,qualite_annotateurs,noise_truth,nu=nu,S=S,data_type=data_type)

    #print("Données d'entrainement")
    #print("Données X : ", xtrain)
    #print("Vrai Labels : ", ztrain)
    #print("Labels données par les annotateurs Y : ", ytrain)
    #print("")
    #print("ytrain size :", ytrain.shape)
    #print("xtrain size :", xtrain.shape)
    #print("Données de test")
    #print("Données X : ", xtest)
    #print("Vrai Labels : ", ztest)
    #print("Labels données par les annotateurs Y : ", ytest)
    #print("")

    if affiche:
        plot_data(xtrain,ztrain)
        plt.title("Données et labels de départ (bruitées)")
        plt.show()
        plot_data(xtrain,ytrain[:,0])
        plt.title("Annotations d'un labelleur")
        plt.show()
    return [xtrain,ytrain,ztrain,xtest,ytest,ztest]

#V. CREE LES CLASSIFIEURS ET APPREND A L'AIDE DU JEU D'ENTRAINEMENT
#LEARNING FROM CROWD ; MAJORITY VOTING ; CLASSIFIEUR REGLOG
#RETOURNE LES CLASSIFIEURS ENTRAINES
