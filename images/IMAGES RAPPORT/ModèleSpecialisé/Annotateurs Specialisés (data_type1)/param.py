N = 100 #nb données
T = 2 #nb annotateurs
d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
noise_truth= 0.5 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
modele= "Bernoulli"
qualite_annotateurs_Bernoulli = [[0.4,0.6],[0.9,0.1]]


alphaNew = np.ones((1,d))*(0.4)
betaNew = 0.1
wNew = np.random.rand(d,T)
gammaNew = np.random.rand(1,T)
