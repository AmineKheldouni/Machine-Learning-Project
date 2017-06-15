N = 100 #nb données
T = 2 #nb annotateurs
d = 2 #nb dimension des données : pas modifiable (gen_arti ne génère que des données de dimension 2)
noise_truth= 0.4 #bruit sur l'attribution des vrais labels gaussiens sur les données 2D (on pourrait aussi jouer sur ecart-type gaussienne avec sigma)
modele= "Bernoulli"

qualite_annotateurs_Bernoulli = [(0.9,0.1),(0.4,0.6)]
#qualite_annotateurs_Bernoulli=[[0.6,0.6],[0.6,0.6],[0.6,0.6],[0.7,0.7],[0.9,0.9]]
Vect=genere(N,T,d,modele,qualite_annotateurs_Bernoulli,generation_Bernoulli,noise_truth,data_type=1,affiche=True)
