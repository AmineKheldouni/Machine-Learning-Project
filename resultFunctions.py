
from gen_arti import *
from gen_true import *

from reglogClassifier import *
from majorityVoting import *
from Classifier1 import *
from Classifier2 import *
from Classifier3 import *

#I. CREATION ET APPEL DES APPRENTISSAGES DES CLASSIFIEURS

def create_class_and_learn(xtrain,ytrain,ztrain,classifier=LearnCrowd,draw_convergence=False):
    """Crée les objets classifieurs par CrowdLearning, MajorityVoting et RegLog sur Z,
    et apprend le classifieur par CrowdLearning sur {xtrain, ytrain} ainsi que le classifieur par RegLog sur {xtrain,ztrain}
    Note : classifier est le type de classifieur par CrowdLearning choisi (LearnCrowd, LearnCrowd2, LearnCrowdOrder)
    Retourne les objets classifieurs après apprentissage"""
    (N,d)=np.shape(xtrain)
    (N,T)=np.shape(ytrain)

    S = classifier(T,N,d)
    M = MajorityVoting()
    C = Classifier_RegLog()

    print("Apprentissage")
    S.fit(xtrain,ytrain,draw_convergence=draw_convergence)
    C.fit(xtrain,ztrain,0.005,1000,affiche=False)

    print("alpha",S.alpha)
    print("beta",S.beta)
    print("gamma",S.gamma)
    print("w",S.w)

    return S,M,C

#-----------------------
def giveTrueData(slicing=0.8):
    X,Z = load_XZ('true_data/dataXZ_Adult.txt')

    Y = load_Y('true_data/dataY_Adult.txt')

    XX,YY, ZZ = genereWithoutMissing(X, Y, Z)
    perm=np.random.permutation(range(XX.shape[0]))

    data_train=perm[:int(slicing*XX.shape[0])] #stations que l'on utilise pour les prédictions
    data_test=perm[int(slicing*XX.shape[0])+1:] #stations dont on cherche à prédire l'offre et sur lesquelles on va calculer la vraisemblance
    sl = int(XX.shape[0]*slicing)

    xtrain, ytrain,ztrain = XX[0:sl,:], YY[0:sl,:], ZZ[0:sl]
    xtest, ytest,ztest = XX[sl+1:,:], YY[sl+1:,:], ZZ[sl+1:]
    xtrain = np.delete(xtrain,2,axis=1)
    xtest = np.delete(xtest,2,axis=1)

    xtrain = (xtrain - np.mean(xtrain)) / np.std(xtrain)
    xtest = (xtest - np.mean(xtest)) / np.std(xtest)
    ztrain = ztrain.astype(int)
    ztest = ztest.astype(int)
    ytrain = ytrain.astype(int)
    ytest = ytest.astype(int)
    return xtrain,ytrain,ztrain,xtest,ytest,ztest

def testTrueData(classifier):
    xtrain,ytrain,ztrain,xtest,ytest,ztest = giveTrueData()
    T = ytrain.shape[1]
    N = xtrain.shape[0]
    d = xtrain.shape[1]
    S = classifier(T,N,d)
    print("Apprentissage")
    S.fit(xtrain, ytrain, max_iter=40)

    print("Performances sur les données d'entrainement : ")
    print("Score en Train : ", S.score(xtrain,ztrain,0.5))
    print("")

    print("Performances sur les données de test : ")
    print("Score en Test : ", S.score(xtest,ztest,0.5))
#----------------------

#II. APPEL DES FONCTIONS DE PREDICTIONS DES CLASSIFIEURS

def predicts(type_return,S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest,s,affiche=False):
    """Fonction de prédiction pour données artificielles 2D
    S,M,C sont les 3 classifieurs (CrowdLearning, MajorityVoting, Reglog sur Z)
    où S et C ont été préalablement appris (avant d'être passés en entrée de cette fonction) sur l'ensemble xtrain,ytrain,ztrain
    s est le seuil utilisé pour donner les prédictions et calculer les scores

    Prédit les labels sur l'ensemble d'apprentissage et sur l'ensemble de test avec les 3 classifieurs,
    donne les scores à la lumière des vrais labels, affiche les prédictions dans le plan 2D (si affiche=True)
    #if type_return=0 :
    #retourne une liste contenant les vecteurs de prédictions des 3 classifieurs sur chacun des deux ensembles (Train/Test) dans l'ordre suivant :
    #[predicts_train_LearnCrowd,predicts_train_Majority,predicts_train_RegLog,predicts_test_LearnCrowd,predicts_test_Majority,predicts_test_RegLog]
    #if type_return=1 :
    #retourne une liste contenant les scores de prédictions des 3 classifieurs sur chacun des deux ensembles (Train/Test) dans l'ordre suivant :
    #[s_train_LearnCrowd,s_train_Majority,s_train_RegLog,s_test_LearnCrowd,s_test_Majority,s_test_RegLog]"""

    s_train_LearnCrowd=S.score(xtrain,ztrain,s)
    predicts_train_LearnCrowd=S.predict(xtrain,s)
    if affiche:
        print("Performances sur les données d'entrainement : ")
        print("Score en Train : ",s_train_LearnCrowd)
        plot_data(xtrain,predicts_train_LearnCrowd)
        plt.title("Prédictions finales sur le Train après crowdlearning")
        plt.show()

    s_train_Majority=M.score(ytrain,ztrain,s)
    predicts_train_Majority=M.predict(ytrain,s)
    if affiche :
        print("Score en Train (Majority Voting) : ", s_train_Majority)
        print("")
        plot_data(xtrain,predicts_train_Majority)
        plt.title("Prédictions finales sur le Train après MajorityVoting")
        plt.show()

    s_train_RegLog=C.score(xtrain,ztrain,s)
    predicts_train_RegLog=C.predict(xtrain,s)
    if affiche:
        print("Score en Train (RegLog sur Z) : ", s_train_RegLog)
        print("")
        plot_data(xtrain,predicts_train_RegLog)
        plt.title("Prédictions finales sur le Train après RegLog sur Z")
        plt.show()

    s_test_LearnCrowd=S.score(xtest,ztest,s)
    predicts_test_LearnCrowd=S.predict(xtest,s)
    if affiche:
        print("Performances sur les données de test : ")
        print("Score en Test : ", s_test_LearnCrowd)
        plot_data(xtest,predicts_test_LearnCrowd)
        plt.title("Prédictions finales sur le Test après crowdlearning")
        plt.show()

    s_test_Majority= M.score(ytest,ztest,s)
    predicts_test_Majority=M.predict(ytest,s)
    if affiche :
        print("Score en Test (Majority Voting) : ", s_test_Majority)
        plot_data(xtest,predicts_test_Majority)
        plt.title("Prédictions finales sur le Test après MajorityVoting")
        plt.show()

    s_test_RegLog=C.score(xtest,ztest,s)
    predicts_test_RegLog=C.predict(xtest,s)
    if affiche:
        print("Score en Test (RegLog sur Z) : ", s_test_RegLog)
        print("")
        plot_data(xtest,predicts_test_RegLog)
        plt.title("Prédictions finales sur le Test après RegLog sur Z")
        plt.show()

    if type_return==0:
        return [predicts_train_LearnCrowd,predicts_train_Majority,predicts_train_RegLog,predicts_test_LearnCrowd,predicts_test_Majority,predicts_test_RegLog]
    elif type_return==1:
        return [s_train_LearnCrowd,s_train_Majority,s_train_RegLog,s_test_LearnCrowd,s_test_Majority,s_test_RegLog]

#III. POUR TRACES DE COURBES ROC

def TP_FP(predictions,truth):
    """renvoit le taux de vrai positif et le taux de faux positif pour le vecteur
    de prédiction 'predictions', et le vecteur des vrais labels 'truth' """
    tmp1 = (predictions==1)&(truth==1);
    TP = np.sum(tmp1==True)/np.sum(truth==1)
    tmp2 = (predictions==1)&(truth==0);
    FP = np.sum(tmp2==True)/np.sum(truth==0)
    return TP,FP

def trace_ROC(S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest):
    """S,M,C sont les 3 classifieurs (CrowdLearning, MajorityVoting, Reglog sur Z)
    où S et C ont été préalablement appris (avant d'être passés en entrée de cette fonction) sur l'ensemble xtrain,ytrain,ztrain.

    Pour l'ensemble d'entrainement et pour l'ensemble de test,
    trace et superpose les courbes ROC des 3 classifieurs (et ajoute les points correspondant aux FP,TP de chaque annotateur)"""

    (N,d)=np.shape(xtrain)
    (N,T)=np.shape(ytrain)

    TP_train_labelleurs=[]
    FP_train_labelleurs=[]
    for t in range(T):
        tp,fp = TP_FP(ytrain[:,t],ztrain)
        TP_train_labelleurs.append(tp)
        FP_train_labelleurs.append(fp)

    TP_test_labelleurs=[]
    FP_test_labelleurs=[]
    for t in range(T):
        tp,fp = TP_FP(ytest[:,t],ztest)
        TP_test_labelleurs.append(tp)
        FP_test_labelleurs.append(fp)

    seuils=[0.001*k for k in range(1001)]

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
        PREDICTS = predicts(0,S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest,s)
        tp,fp=TP_FP(PREDICTS[0],ztrain)
        TP_train_crowd.append(tp)
        FP_train_crowd.append(fp)
        tp,fp=TP_FP(PREDICTS[3],ztest)
        TP_test_crowd.append(tp)
        FP_test_crowd.append(fp)

        tp,fp=TP_FP(PREDICTS[1],ztrain)
        TP_train_majority.append(tp)
        FP_train_majority.append(fp)
        tp,fp=TP_FP(PREDICTS[4],ztest)
        TP_test_majority.append(tp)
        FP_test_majority.append(fp)

        tp,fp=TP_FP(PREDICTS[2],ztrain)
        TP_train_class.append(tp)
        FP_train_class.append(fp)
        tp,fp=TP_FP(PREDICTS[5],ztest)
        TP_test_class.append(tp)
        FP_test_class.append(fp)

    plt.scatter(FP_train_labelleurs,TP_train_labelleurs)
    plt.plot(FP_train_crowd,TP_train_crowd,color="blue",label="ROC trainset CrowdLearning",marker="o")
    plt.plot(FP_train_majority,TP_train_majority,color="red",label="ROC trainset MajorityVoting")
    plt.plot(FP_train_class,TP_train_class,color="yellow",label="ROC trainset  ClassifierTruth")
    plt.legend(bbox_to_anchor=(1, 1.14), loc=1, borderaxespad=0.)
    plt.show()

    plt.scatter(FP_test_labelleurs,TP_test_labelleurs)
    plt.plot(FP_test_crowd,TP_test_crowd,color="blue",label="ROC testset CrowdLearning",marker="o")
    plt.plot(FP_test_majority,TP_test_majority,color="red",label="ROC testset MajorityVoting")
    plt.plot(FP_test_class,TP_test_class,color="yellow",label="ROC testset ClassifierTruth")
    plt.legend(bbox_to_anchor=(1, 1.14), loc=1, borderaxespad=0.)
    plt.show()


def trace_ROC_Unspecialized(S,M,C,S_unspe,xtrain,ytrain,ztrain,xtest,ytest,ztest):
    """S,M,C,S_unspe sont les 4 classifieurs (CrowdLearning Modèle Spécialisé 'LearnCrowd', MajorityVoting, Reglog sur Z, CrowdLearning Modèle Non Spécialisé 'LearnCrowd2')
    où S,C et S_unspe ont été préalablement appris (avant d'être passés en entrée de cette fonction) sur l'ensemble xtrain,ytrain,ztrain.

    Pour l'ensemble d'entrainement et pour l'ensemble de test,
    trace et superpose les courbes ROC des 4 classifieurs (et ajoute les points correspondant aux FP,TP de chaque annotateur)"""

    (N,d)=np.shape(xtrain)
    (N,T)=np.shape(ytrain)

    TP_train_labelleurs=[]
    FP_train_labelleurs=[]
    for t in range(T):
        tp,fp = TP_FP(ytrain[:,t],ztrain)
        TP_train_labelleurs.append(tp)
        FP_train_labelleurs.append(fp)

    TP_test_labelleurs=[]
    FP_test_labelleurs=[]
    for t in range(T):
        tp,fp = TP_FP(ytest[:,t],ztest)
        TP_test_labelleurs.append(tp)
        FP_test_labelleurs.append(fp)

    seuils=[0.0001*k for k in range(10001)]

    TP_train_crowd=[]
    FP_train_crowd=[]
    TP_test_crowd=[]
    FP_test_crowd=[]

    TP_train_crowd_unspe=[]
    FP_train_crowd_unspe=[]
    TP_test_crowd_unspe=[]
    FP_test_crowd_unspe=[]

    TP_train_majority=[]
    FP_train_majority=[]
    TP_test_majority=[]
    FP_test_majority=[]

    TP_train_class=[]
    FP_train_class=[]
    TP_test_class=[]
    FP_test_class=[]

    for s in seuils:
        PREDICTS = predicts(0,S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest,s)
        s_train_LearnCrowd_unspe=S_unspe.predict(xtrain,s)
        s_test_LearnCrowd_unspe=S_unspe.predict(xtest,s)

        tp,fp=TP_FP(PREDICTS[0],ztrain)
        TP_train_crowd.append(tp)
        FP_train_crowd.append(fp)
        tp,fp=TP_FP(PREDICTS[3],ztest)
        TP_test_crowd.append(tp)
        FP_test_crowd.append(fp)

        tp,fp=TP_FP(PREDICTS[1],ztrain)
        TP_train_majority.append(tp)
        FP_train_majority.append(fp)
        tp,fp=TP_FP(PREDICTS[4],ztest)
        TP_test_majority.append(tp)
        FP_test_majority.append(fp)

        tp,fp=TP_FP(PREDICTS[2],ztrain)
        TP_train_class.append(tp)
        FP_train_class.append(fp)
        tp,fp=TP_FP(PREDICTS[5],ztest)
        TP_test_class.append(tp)
        FP_test_class.append(fp)

        tp,fp = TP_FP(s_train_LearnCrowd_unspe,ztrain)
        TP_train_crowd_unspe.append(tp)
        FP_train_crowd_unspe.append(fp)
        tp,fp=TP_FP(s_test_LearnCrowd_unspe,ztest)
        TP_test_crowd_unspe.append(tp)
        FP_test_crowd_unspe.append(fp)

    plt.scatter(FP_train_labelleurs,TP_train_labelleurs)
    plt.plot(FP_train_crowd,TP_train_crowd,color="blue",label="ROC trainset CrowdLearning (Specialized)",marker="o")
    plt.plot(FP_train_crowd_unspe,TP_train_crowd_unspe,color="green",label="ROC trainset CrowdLearning (Unspecialized)",marker="^")
    plt.plot(FP_train_majority,TP_train_majority,color="red",label="ROC trainset MajorityVoting")
    plt.plot(FP_train_class,TP_train_class,color="yellow",label="ROC trainset  ClassifierTruth")
    plt.legend(bbox_to_anchor=(1, 1.14), loc=1, borderaxespad=0.)
    plt.show()

    plt.scatter(FP_test_labelleurs,TP_test_labelleurs)
    plt.plot(FP_test_crowd,TP_test_crowd,color="blue",label="ROC testset CrowdLearning (Specialized)",marker="o")
    plt.plot(FP_test_crowd_unspe,TP_test_crowd_unspe,color="green",label="ROC trainset CrowdLearning (Unspecialized)",marker="^")
    plt.plot(FP_test_majority,TP_test_majority,color="red",label="ROC testset MajorityVoting")
    plt.plot(FP_test_class,TP_test_class,color="yellow",label="ROC testset ClassifierTruth")
    plt.legend(bbox_to_anchor=(1, 1.14), loc=1, borderaxespad=0.)
    plt.show()

#IV. EXPERIENCES (DECRITES DANS LE RAPPORT)

def qualite_unif_x(f=create_class_and_learn,classifier=LearnCrowd):
    """Compare les performances d'un modèle de Crowdlearning au MajorityVoting et au classifieur Reglog sur Z
    pour des annotateurs de Bernouilli non spécialisés (qualité uniforme sur tout l'espace)"""
    N = 100 #nb données
    T = 2 #nb annotateurs
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth= 0.4 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D
    modele= "Bernoulli"

    qualite_annotateurs_Bernoulli=[(0.6, 0.6)]*T #(Spécificité, sensibilité) de l'annotateur
    #qualite_annotateurs_Bernoulli=[[0.6,0.6],[0.6,0.6],[0.6,0.6],[0.7,0.7],[0.9,0.9]]
    Vect=genere(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth,data_type=1,affiche=True)

    xtrain=Vect[0]
    ytrain=Vect[1]
    ztrain=Vect[2]
    xtest=Vect[3]
    ytest=Vect[4]
    ztest=Vect[5]
    S,M,C = f(xtrain,ytrain,ztrain,classifier=classifier,draw_convergence=True)

    predicts(2,S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest,0.5,affiche=True)
    trace_ROC(S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest)

def qualite_depend_x(f=create_class_and_learn):
    """Compare les performances des deux modèles de Crowdlearning (spécialisé et non spécialisé) au MajorityVoting et au classifieur Reglog sur Z
    pour des annotateurs de Bernouilli spécialisés (qualité différente selon si la donnée vérifie |y|<|x| ou |x|<|y|)"""
    N = 100 #nb données
    T = 2 #nb annotateurs
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth= 0.4 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D
    modele= "Bernoulli"

    qualite_annotateurs_Bernoulli = [(0.6,0.9),(0.9,0.6)]  #Proba que l'annotateur ait raison dans la zone 1 de données (1-la valeur dans la zone 2)
    Vect=genere(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli_xy_depend,noise_truth,data_type=3,affiche=False)

    xtrain=Vect[0]
    ytrain=Vect[1]
    ztrain=Vect[2]
    xtest=Vect[3]
    ytest=Vect[4]
    ztest=Vect[5]

    plot_data(xtrain,ztrain)
    plt.title("Vrais labels")
    plt.show()

    plot_data(xtrain,ytrain[:,0])
    plt.title("Annotations du labelleur spécialisé |x|>|y|")
    plt.show()
    plot_data(xtrain,ytrain[:,1])
    plt.title("Annotations du labelleur spécialisé |y|>|x|")
    plt.show()

    #Création et apprentissage des classifieurs :
    #Crowdlearning modèle spécialisé (LearnCrowd), Majority Voting, RegLog sur Z
    S,M,C = f(xtrain,ytrain,ztrain,classifier=LearnCrowd,draw_convergence=True)

    #Création et apprentissage du classifieur Crowdlearning modèle non spécialisé (LearnCrowd2)
    (N,d)=np.shape(xtrain)
    (N,T)=np.shape(ytrain)
    S_unspe = LearnCrowd2(T,N,d)
    S_unspe.fit(xtrain,ytrain,draw_convergence=True)

def qualite_depend_x_evol(f=create_class_and_learn,N_MC=1):
    """Compare les performances des deux modèles de Crowdlearning (spécialisé et non spécialisé) au MajorityVoting et au classifieur Reglog sur Z
    pour 2 annotateurs de Bernouilli spécialisés et complémentaires (qualité différente selon si la donnée vérifie |y|<|x| ou |x|<|y|)
    selon la différence de spécialisation entre les 2 annotateurs dans une même zone (expérience décrite en détails dans le rapport)"""

    special_params=np.arange(0,0.5,0.05).ravel()
    score_train_crowd=[]
    score_train_crowdun=[]
    score_train_majority=[]
    score_train_reglog=[]
    score_test_crowd=[]
    score_test_crowdun=[]
    score_test_majority=[]
    score_test_reglog=[]

    N = 100 #nb données
    T = 2 #nb annotateurs
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth=0.4  #bruit sur l'attribution des vrais labels gaussiens sur les données 2D
    modele= "Bernoulli"

    for s in special_params:
        qualite_annotateurs_Bernoulli=[(0.5+s,1-s),(1-s,0.5+s)] #Proba que l'annotateur ait raison dans la zone 1 de données (1-la valeur dans la zone 2)
        strainS = 0
        strainM = 0
        strainC = 0
        stestS = 0
        stestM = 0
        stestC = 0
        strainSun = 0
        stestSun = 0

        for i in range(N_MC):
            Vect=genere(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli_xy_depend,noise_truth,data_type=3,affiche=False)

            xtrain=Vect[0]
            ytrain=Vect[1]
            ztrain=Vect[2]
            xtest=Vect[3]
            ytest=Vect[4]
            ztest=Vect[5]

            S,M,C = f(xtrain,ytrain,ztrain,classifier=LearnCrowd)

            S_unspe = LearnCrowd2(T,N,d)
            S_unspe.fit(xtrain,ytrain)
            S_unspe_train_LearnCrowd=S_unspe.score(xtrain,ztrain,0.5)
            S_unspe_test_LearnCrowd=S_unspe.score(xtest,ztest,0.5)

            SCORES=predicts(1,S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest,0.5,affiche=False)
            strainS += SCORES[0]
            strainM += SCORES[1]
            strainC += SCORES[2]
            stestS += SCORES[3]
            stestM += SCORES[4]
            stestC += SCORES[5]

            strainSun += S_unspe_train_LearnCrowd
            stestSun += S_unspe_test_LearnCrowd

        score_train_crowd.append(strainS)
        score_train_majority.append(strainM)
        score_train_reglog.append(strainC)
        score_test_crowd.append(stestS)
        score_test_majority.append(stestM)
        score_test_reglog.append(stestC)
        score_train_crowdun.append(strainSun)
        score_test_crowdun.append(stestSun)

    plt.plot(special_params,score_train_crowd,"blue",label="Score trainset CrowdLearning (spécialisé)")
    plt.plot(special_params,score_train_majority,"red",label="Score trainset MajorityVoting")
    plt.plot(special_params,score_train_reglog,"yellow",label="Score trainset ClassifierTruth")
    plt.plot(special_params,score_train_crowdun,"green",label="Score trainset CrowdLearning (non spécialisé)")
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    plt.xlabel("alpha (|0.5-2*alpha| différence de spécialisation des deux annotateurs dans une même zone)")
    plt.show()

    plt.plot(special_params,score_test_crowd,"blue",label="Score testset CrowdLearning (spécialisé)")
    plt.plot(special_params,score_test_majority,"red",label="Score testset MajorityVoting")
    plt.plot(special_params,score_test_reglog,"yellow",label="Score testset ClassifierTruth")
    plt.plot(special_params,score_test_crowdun,"green",label="Score testset CrowdLearning (non spécialisé)")
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    plt.xlabel("alpha (|0.5-2*alpha| différence de spécialisation des deux annotateurs dans une même zone)")
    plt.show()

#------------------------------------------------------------

def drawScoreQuality(s, f=create_class_and_learn, classifier=LearnCrowd, N_MC=1):
    N = 100 #nb données
    T = 10 #nb annotateurs
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth= 0.7 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
    modele= "Bernoulli"
    qualite_consideree = np.arange(0.50,1.,0.05)

    scoreS_train = []
    scoreM_train = []
    scoreC_train = []

    scoreS_test = []
    scoreM_test = []
    scoreC_test = []

    for q in qualite_consideree:
        qualite_annotateurs_Bernoulli=[(q, q)]*T #Proba que l'annotateur ait raison
        scoreStrain = 0
        scoreMtrain = 0
        scoreCtrain = 0
        scoreStest = 0
        scoreMtest = 0
        scoreCtest = 0
        for i in range(N_MC):
            Vect=genere(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth,data_type=0)
            xtrain=Vect[0]
            ytrain=Vect[1]
            ztrain=Vect[2]
            xtest=Vect[3]
            ytest=Vect[4]
            ztest=Vect[5]
            S,M,C = f(xtrain,ytrain,ztrain,classifier=classifier)
            scoreStrain += S.score(xtrain,ztrain,s)
            scoreMtrain += M.score(ytrain,ztrain,s)
            scoreCtrain += C.score(xtrain,ztrain,s)
            scoreStest += S.score(xtest,ztest,s)
            scoreMtest += M.score(ytest,ztest,s)
            scoreCtest += C.score(xtest,ztest,s)

        scoreS_train.append(scoreStrain/N_MC)
        scoreM_train.append(scoreMtrain/N_MC)
        scoreC_train.append(scoreCtrain/N_MC)
        scoreS_test.append(scoreStest/N_MC)
        scoreM_test.append(scoreMtest/N_MC)
        scoreC_test.append(scoreCtest/N_MC)

    plt.plot(qualite_consideree, np.array(scoreS_train), color='blue',label="score de train (CrowLearning)")
    plt.plot(qualite_consideree,np.array(scoreM_train),color='red',label="score de train (MajorityVoting)")
    plt.plot(qualite_consideree, np.array(scoreC_train), color='yellow',label="score de train (RegLog)")
    plt.xlabel("qualité des annotateurs")
    plt.ylabel("score en apprentissage")
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    plt.draw()
    plt.show()

    plt.plot(qualite_consideree,np.array(scoreS_test),color='blue',label="score de test (CrowLearning)")
    plt.plot(qualite_consideree, np.array(scoreM_test), color='red',label="score de test (MajorityVoting)")
    plt.plot(qualite_consideree,np.array(scoreC_test),color='yellow',label="score de test (RegLog)")
    plt.xlabel("qualité des annotateurs")
    plt.ylabel("score en test")
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    plt.draw()
    plt.show()

def drawScoreAnnotateurs(s, f=create_class_and_learn, classifier=LearnCrowd, N_MC=1):
    N = 100 #nb données
    list_T = np.arange(1,21).astype(int)
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth= 0.8 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
    modele= "Bernoulli"

    scoreS_train = []
    scoreM_train = []
    scoreC_train = []

    scoreS_test = []
    scoreM_test = []
    scoreC_test = []

    for T in list_T:
        qualite_annotateurs_Bernoulli=[(0.6, 0.6)]*T #Proba que l'annotateur ait raison
        scoreStrain = 0
        scoreMtrain = 0
        scoreCtrain = 0
        scoreStest = 0
        scoreMtest = 0
        scoreCtest = 0
        for i in range(N_MC):
            Vect=genere(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth,data_type=1)
            xtrain=Vect[0]
            ytrain=Vect[1]
            ztrain=Vect[2]
            xtest=Vect[3]
            ytest=Vect[4]
            ztest=Vect[5]
            S,M,C = f(xtrain,ytrain,ztrain,classifier=classifier)
            scoreStrain += S.score(xtrain,ztrain,s)
            scoreMtrain += M.score(ytrain,ztrain,s)
            scoreCtrain += C.score(xtrain,ztrain,s)
            scoreStest += S.score(xtest,ztest,s)
            scoreMtest += M.score(ytest,ztest,s)
            scoreCtest += C.score(xtest,ztest,s)

        scoreS_train.append(scoreStrain/N_MC)
        scoreM_train.append(scoreMtrain/N_MC)
        scoreC_train.append(scoreCtrain/N_MC)
        scoreS_test.append(scoreStest/N_MC)
        scoreM_test.append(scoreMtest/N_MC)
        scoreC_test.append(scoreCtest/N_MC)

    plt.plot(list_T, np.array(scoreS_train), color='blue',label="score de train (CrowLearning)")
    plt.plot(list_T,np.array(scoreM_train),color='red',label="score de train (MajorityVoting)")
    plt.plot(list_T, np.array(scoreC_train), color='yellow',label="score de train (RegLog)")
    plt.xlabel("nombre d'annotateurs")
    plt.ylabel("score en apprentissage")
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    plt.draw()
    plt.show()

    plt.plot(list_T,np.array(scoreS_test),color='blue',label="score de test (CrowLearning)")
    plt.plot(list_T, np.array(scoreM_test), color='red',label="score de test (MajorityVoting)")
    plt.plot(list_T,np.array(scoreC_test),color='yellow',label="score de test (RegLog)")
    plt.xlabel("nombre d'annotateurs")
    plt.ylabel("score en test")
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    plt.draw()
    plt.show()

def drawScorePropExperts(s, slicing, f=create_class_and_learn, classifier=LearnCrowd, N_MC = 1):
    N = 100 #nb données
    list_T = np.arange(1,21).astype(int)
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth= 0.8 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
    modele= "Bernoulli"
    # slicing = Nombre d'experts

    scoreS_train = []
    scoreM_train = []
    scoreC_train = []

    scoreS_test = []
    scoreM_test = []
    scoreC_test = []

    for T in list_T:
        qualite_annotateurs_Bernoulli=[(0.9, 0.9)]*slicing+ [(0.6,0.6)]*(T-slicing) #Proba que l'annotateur ait raison
        #qualite_annotateurs_Bernoulli=[[0.6,0.6],[0.6,0.6],[0.6,0.6],[0.7,0.7],[0.9,0.9]]
        scoreStrain = 0
        scoreMtrain = 0
        scoreCtrain = 0
        scoreStest = 0
        scoreMtest = 0
        scoreCtest = 0
        for i in range(N_MC):
            Vect=genere(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth,data_type=1)
            xtrain=Vect[0]
            ytrain=Vect[1]
            ztrain=Vect[2]
            xtest=Vect[3]
            ytest=Vect[4]
            ztest=Vect[5]
            S,M,C = f(xtrain,ytrain,ztrain,classifier=classifier)
            scoreStrain += S.score(xtrain,ztrain,s)
            scoreMtrain += M.score(ytrain,ztrain,s)
            scoreCtrain += C.score(xtrain,ztrain,s)
            scoreStest += S.score(xtest,ztest,s)
            scoreMtest += M.score(ytest,ztest,s)
            scoreCtest += C.score(xtest,ztest,s)

        scoreS_train.append(scoreStrain/N_MC)
        scoreM_train.append(scoreMtrain/N_MC)
        scoreC_train.append(scoreCtrain/N_MC)
        scoreS_test.append(scoreStest/N_MC)
        scoreM_test.append(scoreMtest/N_MC)
        scoreC_test.append(scoreCtest/N_MC)

    plt.plot(list_T, np.array(scoreS_train), color='blue',label="score de train (CrowLearning)")
    plt.plot(list_T,np.array(scoreM_train),color='red',label="score de train (MajorityVoting)")
    plt.plot(list_T, np.array(scoreC_train), color='yellow',label="score de train (RegLog)")
    plt.xlabel("nombre d'annotateurs")
    plt.ylabel("score en apprentissage")
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    plt.draw()
    plt.show()

    plt.plot(list_T,np.array(scoreS_test),color='blue',label="score de test (CrowLearning)")
    plt.plot(list_T, np.array(scoreM_test), color='red',label="score de test (MajorityVoting)")
    plt.plot(list_T,np.array(scoreC_test),color='yellow',label="score de test (RegLog)")
    plt.xlabel("nombre d'annotateurs")
    plt.ylabel("score en test")
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    plt.draw()
    plt.show()

def learn_cas_depend_Order(f=create_class_and_learn):
    N = 100 #nb données
    T = 9 #nb annotateurs
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth= 0.65 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
    modele= "Bernoulli"

    qualite_annotateurs_Bernoulli = [(0.6, 0.6)]*T #Proba que l'annotateur ait raison
    nu = [0]*int(T/3)+[1]*int(T/3)+[1]*int(T/3)
    S = [np.random.rand()]*int(T/3)+[1]*int(T/3)+[0]*int(T/3)
    Vect=genere(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli_Order,noise_truth,nu=nu,S=S,data_type=1,affiche=True)

    xtrain=Vect[0]
    ytrain=Vect[1]
    ztrain=Vect[2]
    xtest=Vect[3]
    ytest=Vect[4]
    ztest=Vect[5]
    S,M,C = f(xtrain,ytrain,ztrain,classifier=LearnCrowdOrder,draw_convergence=True)
    S_unspe = LearnCrowd2(T,N,d)
    S_unspe.fit(xtrain,ytrain)

    predicts(2,S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest,0.5,affiche=True)

    trace_ROC_Unspecialized(S,M,C,S_unspe,xtrain,ytrain,ztrain,xtest,ytest,ztest)

####################################
# TRACES DONNEES REELLES
####################################

def regularisation(classifier):
    lbd=[pow(10,-k) for k in range(-7,7)]
    xtrain,ytrain,ztrain,xtest,ytest,ztest = giveTrueData()
    T = ytrain.shape[1]
    N = xtrain.shape[0]
    d = xtrain.shape[1]
    error_train=[]
    error_test=[]
    M = MajorityVoting()
    for l in lbd:
        S = classifier(T,N,d,l)
        S.fit(xtrain,ytrain,max_iter=100)
        results = []
        results.append(S.score(xtrain,ztrain,0.5))
        results.append(S.score(xtest,ztest,0.5))
        results.append(S.score(xtest,ztest,0.5))
        error_train.append(results[0])
        error_test.append(results[2])

    plt.plot(list(range(-7,7)),error_train,color="blue")
    plt.xlabel("Paramètre de régularisation")
    plt.ylabel("Scores")
    plt.title("Scores d'entrainement (bleu) et de test (rouge) \n en fonction du paramètre de régularisation")
    plt.show()


############################################
############################################
############################################

def nb_donnees():
    M=MajorityVoting()
    error_crowd_train=[]
    error_crowd_test=[]
    error_majority_train=[]
    error_majority_test=[]
    Nb = np.array([0.5+i*0.05 for i in range(0,9)])
    X,Z = load_XZ('true_data/dataXZ_Adult.txt')

    Y = load_Y('true_data/dataY_Adult.txt')

    XX,YY, ZZ = genereWithoutMissing(X, Y, Z)
    perm=np.random.permutation(range(XX.shape[0]))
    #permutation des stations (pour éviter que les stations considérées soient toutes à proximité)

    for sliceTrain in Nb:
        data_train=perm[:int(sliceTrain*XX.shape[0])] #stations que l'on utilise pour les prédictions
        data_test=perm[int(sliceTrain*XX.shape[0])+1:] #stations dont on cherche à prédire l'offre et sur lesquelles on va calculer la vraisemblance

        xtrain, ytrain,ztrain = XX[data_train], YY[data_train], ZZ[data_train]
        xtest, ytest,ztest = XX[data_test], YY[data_test], ZZ[data_test]
        xtrain = np.delete(xtrain,2,axis=1)
        xtest = np.delete(xtest,2,axis=1)

        xtrain = (xtrain - np.mean(xtrain)) / np.std(xtrain)
        xtest = (xtest - np.mean(xtest)) / np.std(xtest)
        ztrain = ztrain.astype(int)
        ztest = ztest.astype(int)
        ytrain = ytrain.astype(int)
        ytest = ytest.astype(int)

        S = LearnCrowd(ytrain.shape[1], xtrain.shape[0], xtrain.shape[1])
        S.fit(xtrain,ytrain,max_iter=200)

        error_crowd_train.append(S.score(xtrain,ztrain,0.5))
        error_crowd_test.append(S.score(xtest,ztest,0.5))
        error_majority_train.append(M.score(ytrain,ztrain,0.5))
        error_majority_test.append(M.score(ytest,ztest,0.5))

    plt.plot(Nb,error_crowd_train,color="blue")
    plt.plot(Nb,error_crowd_test,color="red")
    plt.plot(Nb,error_majority_train,color="yellow")
    plt.plot(Nb,error_majority_test,color="green")
    plt.xlabel("Nombre de données d'apprentissage")
    plt.ylabel("Scores")
    plt.title("Scores sur les ensembles d'apprentissage (bleu,jaune) et de test (rouge,vert)  \n formant une partition (CrowdLearning,MajorityVoting)")
    plt.show()

def traceTrueData(classifier1=LearnCrowd, classifier2=LearnCrowd2):
    xtrain,ytrain,ztrain,xtest,ytest,ztest = giveTrueData()
    T = ytrain.shape[1]
    N = xtrain.shape[0]
    d = xtrain.shape[1]
    S = classifier1(T,N,d,0.01)
    M = MajorityVoting()
    C = Classifier_RegLog()
    S_unspe = classifier2(T,N,d)

    S.fit(xtrain, ytrain, max_iter=60,draw_convergence=True)
    C.fit(xtrain,ztrain,0.01,1000,affiche=False)
    S_unspe.fit(xtrain, ytrain,draw_convergence=True)

    # predicts(2,S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest,0.5,affiche=False)
    trace_ROC_Unspecialized(S,M,C,S_unspe,xtrain,ytrain,ztrain,xtest,ytest,ztest)
