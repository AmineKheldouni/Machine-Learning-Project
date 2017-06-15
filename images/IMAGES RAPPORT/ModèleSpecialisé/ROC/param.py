N = 100 #nb données
T = 5 #nb annotateurs
d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
noise_truth= 0.7 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
modele= "Bernoulli"
qualite_annotateurs_Bernoulli = [(0.6,0.6)]*T

alphaNew = np.random.rand(1,d)
betaNew = np.random.rand()
wNew = np.random.rand(d,T)
gammaNew = np.random.rand(1,T)
