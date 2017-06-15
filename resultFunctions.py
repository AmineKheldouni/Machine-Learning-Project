
from gen_arti import *
from gen_true import *

from reglogClassifier import *
from majorityVoting import *
from Classifier1 import *
from Classifier2 import *


def create_class_and_learn(xtrain,ytrain,ztrain,classifier=LearnCrowd,draw_convergence=False):

    (N,d)=np.shape(xtrain)
    (N,T)=np.shape(ytrain)

    S = classifier(T,N,d)
    M = MajorityVoting()
    C = Classifier_RegLog()

    print("Apprentissage")
    S.fit(xtrain,ytrain,draw_convergence=draw_convergence, epsGrad=10**(-11))
    C.fit(xtrain,ztrain,0.005,1000,affiche=False)

    print("alpha",S.alpha)
    print("beta",S.beta)
    print("gamma",S.gamma)
    print("w",S.w)

    return S,M,C


#VI. PREDIT LES VRAIS LABELS, DONNE LES SCORES, AFFICHE LES PREDICTIONS POUR LE SEUIL S

#if type_return=0
#RETOURNE LE GROS VECTEUR DES PREDICTIONS A 6 TERMES (3 CLASSIFIEURS, TRAIN OU TEST) :
#[predicts_train_LearnCrowd,predicts_train_Majority,predicts_train_RegLog,predicts_test_LearnCrowd,predicts_test_Majority,predicts_test_RegLog]

#if type_return=1
#RETOURNE LE GROS VECTEUR DES SCORES A 6 TERMES (3 CLASSIFIEURS, TRAIN OU TEST) :
#[s_train_LearnCrowd,s_train_Majority,s_train_RegLog,s_test_LearnCrowd,s_test_Majority,s_test_RegLog]

def giveTrueData():
    X,Z = load_XZ('true_data/dataXZ_Adult.txt')

    Y = load_Y('true_data/dataY_Adult.txt')

    XX,YY, ZZ = genereWithoutMissing(X, Y, Z)

    sliceTrain = int(XX.shape[0]*0.8)
    xtrain, ytrain,ztrain = XX[0:sliceTrain,:], YY[0:sliceTrain,:], ZZ[0:sliceTrain]
    xtest, ytest,ztest = XX[sliceTrain+1:,:], YY[sliceTrain+1:,:], ZZ[sliceTrain+1:]
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

def predicts(type_return,S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest,s,affiche=False):
    #print("####################################################")
    #print("Test à l'aide de X")

    s_train_LearnCrowd=S.score(xtrain,ztrain,s)
    predicts_train_LearnCrowd=S.predict(xtrain,s)
    #plot_frontiere(xtest,S.predict(xtest),step=50) C'est XTEST ? Pas ZTEST ?
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
    #plot_frontiere(xtest,S.predict(xtest),step=50)
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

#VI. TRACE DE COURBES ROC EN TRAIN ET TEST POUR LES TROIS CLASSIFIEURS
#S,M,C DOIVENT ETRE DEJA ENTRAINES SUR LE MEME TRAIN QUE L'INPUT DE LA FONCTION

def TP_FP(predictions,truth):
    tmp1 = (predictions==1)&(truth==1);
    TP = np.sum(tmp1==True)/np.sum(truth==1)
    tmp2 = (predictions==1)&(truth==0);
    FP = np.sum(tmp2==True)/np.sum(truth==0)
    return TP,FP

def trace_ROC(S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest):
    """ trace les courbes ROC pour le modèle learning from the Crowd avec predict(X),
    compare avec un classifieur sur Z, compare avec un majority voting,
    place aussi les FP et TP de chaque labelleur // ground truth"""

    (N,d)=np.shape(xtrain)
    (N,T)=np.shape(ytrain)

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
        #print(PREDICTS[0],ztrain)
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
    plt.scatter(FP_train_crowd,TP_train_crowd,color="blue",label="ROC trainset CrowdLearning ")
    plt.plot(FP_train_majority,TP_train_majority,color="red",label="ROC trainset MajorityVoting")
    plt.plot(FP_train_class,TP_train_class,color="yellow",label="ROC trainset  ClassifierTruth")
    plt.legend(bbox_to_anchor=(1, 1.14), loc=1, borderaxespad=0.)
    plt.show()

    plt.scatter(FP_test_labelleurs,TP_test_labelleurs)
    plt.scatter(FP_test_crowd,TP_test_crowd,color="blue",label="ROC testset crowdlearning")
    plt.plot(FP_test_majority,TP_test_majority,color="red",label="ROC testset majorityVoting")
    plt.plot(FP_test_class,TP_test_class,color="yellow",label="ROC testset ClassifierTruth")
    plt.legend(bbox_to_anchor=(1, 1.14), loc=1, borderaxespad=0.)
    plt.show()

#VII. EXPERIENCES

def learn_cas_unif_x(f=create_class_and_learn):
    N = 100 #nb données
    T = 5 #nb annotateurs
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth= 0.7 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
    modele= "Bernoulli"

    qualite_annotateurs_Bernoulli=[(0.6, 0.6)]*T #Proba que l'annotateur ait raison
    #qualite_annotateurs_Bernoulli=[[0.6,0.6],[0.6,0.6],[0.6,0.6],[0.7,0.7],[0.9,0.9]]
    Vect=genere(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth,data_type=1,affiche=True)

    xtrain=Vect[0]
    ytrain=Vect[1]
    ztrain=Vect[2]
    xtest=Vect[3]
    ytest=Vect[4]
    ztest=Vect[5]
    S,M,C = f(xtrain,ytrain,ztrain,draw_convergence=True)

    predicts(2,S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest,0.5,affiche=True)
    trace_ROC(S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest)

def specialisedAnnotators(f=create_class_and_learn):
    N = 100 #nb données
    T = 2 #nb annotateurs
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth= 0.5 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
    modele= "Bernoulli"
    qualite_annotateurs_Bernoulli = [[0.4,0.6],[0.9,0.1]]

    Vect=genere(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth,affiche=True,data_type=1)

    xtrain=Vect[0]
    ytrain=Vect[1]
    ztrain=Vect[2]
    xtest=Vect[3]
    ytest=Vect[4]
    ztest=Vect[5]
    S,M,C = f(xtrain,ytrain,ztrain)
    SCORES=predicts(1,S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest,0.5,affiche=True)

    trace_ROC(S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest)

def compareNoneSpecialised(f=create_class_and_learn):
    N = 100 #nb données
    T = 2 #nb annotateurs
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth= 0.65 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
    modele= "Bernoulli"

    qualite_annotateurs_Bernoulli = [(0.9,0.1),(0.4,0.6)]
    #qualite_annotateurs_Bernoulli=[[0.6,0.6],[0.6,0.6],[0.6,0.6],[0.7,0.7],[0.9,0.9]]
    Vect=genere(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth,data_type=1,affiche=True)

    xtrain=Vect[0]
    ytrain=Vect[1]
    ztrain=Vect[2]
    xtest=Vect[3]
    ytest=Vect[4]
    ztest=Vect[5]

    S,M,C = f(xtrain,ytrain,ztrain,draw_convergence=True)
    (N,d)=np.shape(xtrain)
    (N,T)=np.shape(ytrain)

    S_now = LearnCrowd(T,N,d)

    print("Apprentissage")
    S_now.fitNoW(xtrain,ytrain,draw_convergence=False)

    sNoW_train_LearnCrowd=S_now.score(xtrain,ztrain,0.5)

    print("Performances sur les données d'entrainement (non supervisé) : ")
    print("Score en Train : ",sNoW_train_LearnCrowd)
    sNoW_test_LearnCrowd=S_now.score(xtest,ztest,0.5)

    print("Performances sur les données de test (non supervisé): ")
    print("Score en Test : ", sNoW_test_LearnCrowd)

    predicts(2,S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest,0.5,affiche=True)

    trace_ROC(S,M,S_now,xtrain,ytrain,ztrain,xtest,ytest,ztest)

def learn_cas_depend_x(f=create_class_and_learn,N_MC=1):

    special_params=np.arange(0.05,0.45,0.01).ravel()
    score_train_crowd=[]
    score_train_majority=[]
    score_train_reglog=[]
    score_test_crowd=[]
    score_test_majority=[]
    score_test_reglog=[]

    N = 100 #nb données
    T = 2 #nb annotateurs
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth=0.5 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
    modele= "Bernoulli"

    for s in special_params:
        qualite_annotateurs_Bernoulli=[[0.5+s,0.5-s],[1-s,s]] #Proba que l'annotateur ait raison dans la zone 1 de données (1-la valeur dans la zone 2)
        strainS = 0
        strainM = 0
        strainC = 0
        stestS = 0
        stestM = 0
        stestC = 0

        for i in range(N_MC):
            Vect=genere(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth,affiche=False,data_type=1)

            xtrain=Vect[0]
            ytrain=Vect[1]
            ztrain=Vect[2]
            xtest=Vect[3]
            ytest=Vect[4]
            ztest=Vect[5]

            S,M,C = f(xtrain,ytrain,ztrain)

            SCORES=predicts(1,S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest,0.5)
            strainS += SCORES[0]
            strainM += SCORES[1]
            strainC += SCORES[2]
            stestS += SCORES[3]
            stestM += SCORES[4]
            stestC += SCORES[5]

        score_train_crowd.append(strainS)
        score_train_majority.append(strainM)
        score_train_reglog.append(strainC)
        score_test_crowd.append(stestS)
        score_test_majority.append(stestM)
        score_test_reglog.append(stestC)

    plt.plot(special_params,score_train_crowd,"blue",label="Score CrowdLearning")
    plt.plot(special_params,score_train_majority,"red",label="Score MajorityVoting")
    plt.plot(special_params,score_train_reglog,"yellow",label="Score ClassifierTruth")
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    plt.show()

    plt.plot(special_params,score_test_crowd,"blue")
    plt.plot(special_params,score_test_majority,"red")
    plt.plot(special_params,score_test_reglog,"yellow")
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    plt.show()


def LearnfromtheCrowd2(N,T, d, modele,qualite_annotateurs, generateur, classifier=LearnCrowd,noise_truth=0):
    print("Rappel des paramĂ¨tres")
    print("Nombre de donnĂŠes gĂŠnĂŠrĂŠes : ", N)
    print("Nombre de dimensions des donnĂŠes gĂŠnĂŠrĂŠes : ", d)
    print("Nombre d'annotateurs : ", T)
    print("Modèle : ", modele)
    # print("ProbabilitĂŠs de succĂ¨s des annotateurs : ", qualite_annotateurs)
    print("")

    print("Génération des donnĂŠes")

    xtrain, ytrain,ztrain = generateur(N,T,qualite_annotateurs,noise_truth)
    xtest, ytest,ztest = generateur(N,T,qualite_annotateurs,noise_truth)

    plot_data(xtrain,ztrain)
    plt.title("Données et labels de départ (gaussiennes bruitées)")
    plt.show()
    if modele=="Bernoulli":
        plot_data(xtrain,ytrain[:,0])
        plt.title("Annotations d'un labelleur")
        plt.show()
    S = LearnCrowd2(T, N, d)

    print("Apprentissage ... \n")
    S.fit(xtrain,ytrain)
    print("Apprentissage terminé ! \n")

    print("Performances sur les donnĂŠes d'entrainement : ")
    print("Score en Train : ", S.score(xtrain,ztrain))
    print("")
    plot_data(xtrain,S.predict(xtrain))
    plt.title("Prédictions finales sur le Train après crowdlearning")
    plt.show()
    print("Performances sur les données de test : ")
    print("Score en Test : ", S.score(xtest,ztest))
    plot_data(xtest,S.predict(xtest))
    plt.title("Prédictions finales sur le Test")
    plt.show()

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
####################################
# TRACES DONNEES REELLES
####################################

def regularisation(classifier):
    lbd=[pow(10,-k) for k in range(-5,5)]
    error_train=[]
    error_test=[]
    M = MajorityVoting()
    strain_majority=M.score(ytrain,ztrain,0.5)
    stest_majority= M.score(ytest,ztest,0.5)
    for l in lbd:
        S = classifier(T,N,d,l)
        S.fit(xtrain,ytrain,max_iter=20)
        results = []
        results.append(S.score(xtrain,ztrain,0.5))
        results.append(M.score(xtrain,ztrain,0.5))
        results.append(S.score(xtest,ztest,0.5))
        results.append(S.score(xtest,ztest,0.5))
        error_train.append(results[0])
        error_test.append(results[2])

    plt.plot(list(range(-5,5)),error_train,color="blue")
    plt.plot(list(range(-5,5)),error_test,color="red")
    plt.xlabel("Paramètre de régularisation")
    plt.ylabel("Erreurs")
    plt.title("Erreurs d'entrainement (bleu) et de test (rouge) \n en fonction du paramètre de régularisation")
    plt.show()


############################################
############################################
############################################

def nb_donnees(classifier):
    global N

    Nb=np.array([XX.shape[0]*0.5,XX.shape[0]*0.6,XX.shape[0]*0.7,XX.shape[0]*0.8])
    S = classifier(T,N,d)
    M=MajorityVoting()
    error_crowd_train=[]
    error_crowd_test=[]
    error_majority_train=[]
    error_majority_test=[]

    for sliceTrain in Nb.astype(int):

        xtrain, ytrain,ztrain = XX[0:sliceTrain,:], YY[0:sliceTrain,:], ZZ[0:sliceTrain]
        xtest, ytest,ztest = XX[sliceTrain+1:,:], YY[sliceTrain+1:,:], ZZ[sliceTrain+1:]
        xtrain = np.delete(xtrain,2,axis=1)
        xtest = np.delete(xtest,2,axis=1)

        xtrain = (xtrain - np.mean(xtrain)) / np.std(xtrain)
        xtest = (xtest - np.mean(xtest)) / np.std(xtest)
        ztrain = ztrain.astype(int)
        ztest = ztest.astype(int)
        ytrain = ytrain.astype(int)
        ytest = ytest.astype(int)

        S = LearnCrowd2(ytrain.shape[1], xtrain.shape[0], xtrain.shape[1])
        S.fit(xtrain,ytrain,max_iter=20)

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
    plt.show()

def traceTrueData(classifier):
    xtrain,ytrain,ztrain,xtest,ytest,ztest = giveTrueData()
    T = ytrain.shape[1]
    N = xtrain.shape[0]
    d = xtrain.shape[1]
    S = classifier(T,N,d)
    M = MajorityVoting()
    C = Classifier_RegLog()

    S.fit(xtrain, ytrain, max_iter=100,draw_convergence=True)
    C.fit(xtrain,ztrain,0.005,1000,affiche=False)

    predicts(2,S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest,0.5,affiche=False)
    trace_ROC(S,M,C,xtrain,ytrain,ztrain,xtest,ytest,ztest)
