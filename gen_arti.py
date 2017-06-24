from importModule import *

#I. BLOC GENERATION DES FEATURES X ET DES VRAIS LABELS Z

#gen_arti,plot_data,plot_frontiere,make_grid : fonctions reprises du TP3 de Machine Learning
#gen_arti légèrement adaptée (data_type=1,data_type=3)

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

#II. BLOC GENERATION DES LABELS DES ANNOTATEURS Y

def modifie_label_Bernoulli(label,proba):
    """renvoie le label attribué par l'annotateur de Bernouilli suivant proba :
    label est le vrai label Z (vérité terrain) correspondant à la donnée
    proba = (probabilité que l'annotateur donne le vrai label si ce dernier est 0,
    probabilité que l'annotateur donne le vrai label si ce dernier est 1)
    c'est-à-dire proba = (spécificité, sensibilité) de l'annotateur de Bernouilli"""
    valeur_proba=np.random.uniform(0,1)
    label_res=label
    if((label==0)and(valeur_proba>=proba[0])):
        label_res=1-label
    if((label==1)and(valeur_proba>=proba[1])):
        label_res=1-label
    return label_res

def generation_Bernoulli(N,T,qualite_annotateur_Bernoulli,noise_truth,data_type=0):
    """retourne trois matrices :
    xtrain les N données artificielles de dimension 2 de type data_type générées par gen_arti,
    ytrain les labels donnés par les T annotateurs de Bernouilli pour chacune des N données
    ztrain les vrais labels
    où qualite_annotateurs est une liste contenant pour chaque annotateur de Bernouilli la paire (spécificité, sensibilité)
    et où noise_truth est le bruit sur l'attribution des vrais labels"""
    xtrain,ztrain = gen_arti(nbex=N,data_type=data_type,epsilon=noise_truth) #vrai labels non bruités
    ztrain=(ztrain+1)/2
    ytrain=np.zeros((N,T)) #changement des labels
    for t in range(T):
        annote=lambda x:modifie_label_Bernoulli(x,qualite_annotateur_Bernoulli[t])
        annote=np.vectorize(annote)
        ytrain[:,t]=annote(ztrain)
    return xtrain,ytrain,ztrain

def modifie_label_Bernoulli_xy_depend(data,label,proba):
    """renvoie le label attribué par l'annotateur de Bernouilli spécialisé suivant proba :
    data correspond aux features X de la donnée de dimension 2
    label est le vrai label Z (vérité terrain) correspondant à cette donnée data
    proba = (probabilité que l'annotateur donne le vrai label si la donnée est située dans la zone |y|>|x|,
    probabilité que l'annotateur donne le vrai label si la donnée est située dans la zone |x|>|y|)"""
    valeur_proba=np.random.uniform(0,1)
    label_res=label
    if (abs(data[1])>abs(data[0])): #(groupe 1 de données)
        if(valeur_proba>=proba[0]):
           label_res=1-label
    else: #(groupe 2 de données)
        if(valeur_proba>=proba[1]):
           label_res=1-label
    return label_res

def generation_Bernoulli_xy_depend(N,T,qualite_annotateur_Bernoulli,noise_truth,data_type=0):
    """retourne trois matrices :
    xtrain les N données artificielles de dimension 2 de type data_type générées par gen_arti,
    ytrain les labels donnés par les T annotateurs de Bernouilli pour chacune des N données
    ztrain les vrais labels
    où qualite_annotateurs est une liste contenant pour chaque annotateur de Bernouilli la paire proba
    proba = (probabilité que l'annotateur donne le vrai label si la donnée est située dans la zone |y|>|x|,
    probabilité que l'annotateur donne le vrai label si la donnée est située dans la zone |x|>|y|)
    et où noise_truth est le bruit sur l'attribution des vrais labels"""
    xtrain,ztrain = gen_arti(nbex=N,data_type=data_type,epsilon=noise_truth) #vrai labels non bruités
    ztrain=(ztrain+1)/2
    ytrain=np.zeros((N,T)) #changement des labels
    for t in range(T):
        annote=lambda idx_data,z:modifie_label_Bernoulli_xy_depend(xtrain[idx_data,:],z,qualite_annotateur_Bernoulli[t])
        annote=np.vectorize(annote)
        ytrain[:,t]=annote(list(range(np.shape(xtrain)[0])),ztrain)
    return xtrain,ytrain,ztrain

def modifie_label_Bernoulli_Order(label,proba,nu,S):
    """renvoie le label attribué par l'annotateur de Bernouilli spécialisé suivant proba, nu, S:
    label est le vrai label Z (vérité terrain) correspondant à la donnée
    proba est la paire (spécificité, sensibilité) de l'annotateur s'il utilise sa connaissance
    nu est sa propension à réagir à la consigne ou à l'ordre (qui dit de voter 1) plutôt qu'à utiliser sa connaissance
    S est sa propension à aligner son label sur la consigne (vote 1) ou à s'y opposer (vote 0) s'il décide finalement d'y réagir"""
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
    """retourne trois matrices :
    xtrain les N données artificielles de dimension 2 de type data_type générées par gen_arti,
    ytrain les labels donnés par les T annotateurs de Bernouilli pour chacune des N données
    ztrain les vrais labels
    où qualite_annotateurs est une liste contenant pour chaque annotateur de Bernouilli la paire (spécificité, sensibilité) de l'annotateur s'il utilise sa connaissance
    où nu est une liste contenant pour chaque annotateur de Bernouilli sa propension à réagir à la consigne ou à l'ordre (qui dit de voter 1) plutôt qu'à utiliser sa connaissance
    où S est une liste contenant pour chaque annotateur de Bernouilli sa propension à aligner son label sur la consigne (vote 1) ou à s'y opposer (vote 0) s'il décide finalement de réagir à la consigne
    et où noise_truth est toujours le bruit sur l'attribution des vrais labels"""
    xtrain,ztrain = gen_arti(nbex=N,data_type=data_type,epsilon=noise_truth) #vrai labels non bruités
    ztrain=(ztrain+1)/2
    ytrain=np.zeros((N,T)) #changement des labels
    for t in range(T):
        annote=lambda x:modifie_label_Bernoulli_Order(x,qualite_annotateur_Bernoulli[t],nu[t],S[t])
        annote=np.vectorize(annote)
        ytrain[:,t]=annote(ztrain)
    return xtrain,ytrain,ztrain

#III. GENERATION DE X,Y,Z : FEATURES, ANNOTATIONS, VRAIS LABELS

def genere(N,T,d,modele,qualite_annotateurs,generateur,noise_truth,affiche=False, nu=None, S=None, data_type=0):
    """genère ensemble d'entrainement Xtrain,Ytrain,Ztrain et ensemble de test Xtest,Ytest,ztest
    constitués chacun de N données artificielles de dimension d, pour T annotateurs
    data_type est le type de données artificielles générées avec gen_arti
    generateur est une fonction indiquant la manière de générer les labels attribués par les T annotateurs
    qualite_annotateurs, nu et S correspondent aux caractéristiques de ces T annotateurs
    Si on choisit generateur = "generation_Bernoulli", renseigner qualite_annotateur comme précisé dans la légende de cette fonction
    Si on choisit generateur = "generation_Bernoulli_xy_depend", renseigner qualite_annotateur comme précisé dans la légende de cette fonction
    Si on choisit generateur = "generation_Bernoulli_Order", renseigner qualite_annotateur, nu et S comme précisé dans la légende de cette fonction"""
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
