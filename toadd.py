#Dans resultsFunctions.py
def compareModele23(f=create_class_and_learn):
    N = 100 #nb données
    T = 10 #nb annotateurs
    d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
    noise_truth= 0.4 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
    modele= "Bernoulli"

    qualite_annotateurs_Bernoulli = [(0.6,0.6)]*10
    S=[0,0,0,0,0,0.3,0.3,0.5,0.9,0.9]
    nu=[0,0,0,0,0,0,1,0,0,1]

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

    plot_data(xtrain,ytrain[:,0])
    plt.title("Annotations d'un labelleur indépendant")
    plt.show()
    plot_data(xtrain,ytrain[:,9])
    plt.title("Annotations d'un labelleur très sensible à la consigne (0.9 et la suivant)")
    plt.show()
    plot_data(xtrain,ytrain[:,5])
    plt.title("Annotations d'un labelleur assez sensible à la consigne (0.3 et s'y opposant)")
    plt.show()

    #JE VEUX JUSTE CONVERGENCE EM MODELE DEPENDANT ET VALEURS FINALES DES PARAMETRES
    #PAS DE ROC NI DE PREDICTIONS
