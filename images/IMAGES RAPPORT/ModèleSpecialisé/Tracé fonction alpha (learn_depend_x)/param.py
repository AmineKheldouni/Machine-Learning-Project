
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
    qualite_annotateurs_Bernoulli=[[0.5+s,0.5-s],[1-s,s]]
    ...
    
N_MC = 10
data_type = 1
