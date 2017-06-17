# Modules importés



from resultFunctions import *


# N = 10
# T = 5
# d = 2
# noise_truth=0.05
# modele= "Bernoulli"
# qualite_annoteur = [(0.6,0.6)]*T

# Test donnĂŠes artificielles :

#LearnfromtheCrowd2(N,T,d,modele,qualite_annoteur,generation_Bernouilli,noise_truth)

# learn_cas_unif_x(create_class_and_learn)
#learn_cas_unif_x(create_class_and_learn2)

# Courbe (a) sharelatex :
#
# drawScoreQuality(0.5,N_MC=1)
# drawScoreAnnotateurs(0.5,N_MC=5)
# drawScorePropExperts(0.5,2,N_MC=5)

# specialisedAnnotators()


#exp1
compareNoneSpecialized()

#exp2
# learn_cas_depend_x(N_MC=1)

#exp3
# traceTrueData()

#exp4
# learn_cas_depend_Order()

"""
Résumé des constantes à prendre :
-> Le programme marche avec une initialisation alpha < 0, beta < 0 (Pas trop petits non plus ...)
-> noise = 0.8 pour un ensemble non séparable à peine, N = 100
-> Nombre max ditérations : 300 c'est bien
-> Scores train & test bons pour les données réelles mais pas de ROC pour le modèle
-> Monte Carlo pour les 3 courbes de comparaison. Ajout de la classification V2 dans les comparaison ? Ce serait intéressant ?
-> Le programme tourne longtemps pour un petit nombre Monte-Carlo, diminution de max_iter pour augmenter N_MC ? => Compromis.


Belles courbes :
-> alphaNew initial : 0.3 * ones ..
-> betaNew initial : 0.2
-> actu avec += step*grad ..
-> noise = 0.6
-> N = 100
-> T = 5

et pour l'experience :
tu prends trois groupes d'annotateurs
l'un avec un nu_t=0
l'un avec un nu_t=1, C_t=1
l'un avec un nu_t=1, C_t=0
"""
