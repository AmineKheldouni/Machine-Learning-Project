#Dans resultsFunctions.py
from mainExec import *

def compareModele23(f=create_class_and_learn):
    N = 100 #nb données
    T = 5 #nb annotateurs
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth= 0.6 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
    modele= "Bernoulli"

    qualite_annotateurs_Bernoulli = [(0.65,0.65)]*T
    # S = [0,1,1]
    # nu = [1,0,1]
    S=[0.1,0,0.8,0.3,0.9]
    nu=[0.8,0,0.1,0.9,0.8]
    # NU :
    # [[ 0.3617427   0.75865228  0.65538373 -0.097082    0.28526687]]
    # S :
    # [[ 2.04500706  1.01161838  1.44296252  1.48409623  1.05572158]]

    # qualite_annotateurs_Bernoulli=[[0.6,0.6],[0.6,0.6],[0.6,0.6],[0.7,0.7],[0.9,0.9]]
    Vect=genere(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli_Order,noise_truth,nu=nu, S=S,data_type=0,affiche=False)

    xtrain=Vect[0]
    ytrain=Vect[1]
    ztrain=Vect[2]
    xtest=Vect[3]
    ytest=Vect[4]
    ztest=Vect[5]

    plot_data(xtrain,ztrain)
    plt.title("Vrais labels")
    plt.show()

    # plot_data(xtrain,ytrain[:,0])
    # plt.title("Annotations d'un labelleur indépendant")
    # plt.show()
    # plot_data(xtrain,ytrain[:,9])
    # plt.title("Annotations d'un labelleur très sensible à la consigne (0.9 et la suivant)")
    # plt.show()
    # plot_data(xtrain,ytrain[:,5])
    # plt.title("Annotations d'un labelleur assez sensible à la consigne (0.3 et s'y opposant)")
    # plt.show()
    S_order = LearnCrowdOrder(T,N,d)
    S_order.fit(xtrain,ytrain,draw_convergence=True)
    print(S_order.score(xtrain,ztrain,0.5))
    print(S_order.score(xtest,ztest,0.5))

    #JE VEUX JUSTE CONVERGENCE EM MODELE DEPENDANT ET VALEURS FINALES DES PARAMETRES
    #PAS DE ROC NI DE PREDICTIONS

compareModele23()
