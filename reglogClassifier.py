from importModule import *

#Classifieur par régression logistique à apprendre directement sur les vrais labels Z

def optimize(fonc,dfonc,xinit,eps,max_iter):
    x_histo=[]
    f_histo=[]
    grad_histo=[]
    it=0
    x_histo.append(xinit)
    f_histo.append(fonc(x_histo[0]))
    grad_histo.append(dfonc(x_histo[0]))
    while (it<max_iter):
        it+=1
        x_new=x_histo[it-1]-eps/np.sqrt(1+it) * dfonc(x_histo[it-1])
        x_histo.append(x_new)
        f_histo.append(fonc(x_new))
        grad_histo.append(dfonc(x_new))
    x_histo=np.array(x_histo)
    f_histo=np.array(f_histo)
    grad_histo=np.array(grad_histo)
    return (x_histo,f_histo,grad_histo)

def signe(x):
    if x>=0:
        return 1
    return 0

def reglog(w,data,label):
    #Renvoit le cout en w pour la régression logistique
    ps=np.multiply(np.reshape(label,(-1,1)),np.dot(data,np.transpose(w)))
    rlog=lambda x: math.log(1/(1+math.exp(-x)))
    rlog=np.vectorize(rlog)
    return -np.mean(rlog(ps))

def grad_reglog(w,data,label):
    #Renvoit le gradient du cout en w pour la régression logistique
    (n,d)=np.shape(data)
    label=np.reshape(label,(-1,1))
    ps=np.multiply(label,np.dot(data,np.transpose(w)))
    sig=lambda x:1/(1+math.exp(-x))
    sig=np.vectorize(sig)
    tmp=np.multiply(np.multiply(label,np.exp(-ps)),sig(ps))
    return -np.mean(np.multiply(np.tile(tmp,(1,d)),data),axis=0)


class Classifier_Binary():
    def __init__(self):
        self.w=-1 #vecteur des poids

class Classifier_RegLog(Classifier_Binary):
    def __init__(self):
        super().__init__()

    def predict(self,data,seuil):
        (n,d)=np.shape(data)
        col_id=np.reshape((np.ones(n)),(n,1)) # vecteur colonne de 1
        data=np.concatenate((col_id,data),axis=1)  # matrice de "design"

        predictions=np.dot(data,np.transpose(self.w))
        #sign = lambda x:signe(x)
        #sign=np.vectorize(sign)
        sigm = lambda x:1/(1+np.exp(-x))
        sigm = np.vectorize(sigm)
        predictions = sigm(predictions) > seuil
        convertfloat = lambda x:float(x)
        convertfloat = np.vectorize(convertfloat)
        predictions = convertfloat(predictions)
        return predictions.ravel()
        #return sign(predictions).flatten()

    def fit(self,data,labels,eps,nb_iter,affiche=False):
        labels=2*labels-1
        (n,d)=np.shape(data)
        col_id=np.reshape((np.ones(n)),(n,1)) # vecteur colonne de 1
        data=np.concatenate((col_id,data),axis=1)  # matrice de "design"

        cost = lambda x : reglog(x,data,labels)
        grad_cost = lambda x : grad_reglog(x,data,labels)
        winit = np.zeros((1,d+1))
        (w_histo,cost_histo,grad_cost_histo) = optimize(cost,grad_cost,winit,eps,nb_iter)
        self.w = w_histo[-1]

        if affiche:
            plt.figure()
            plt.plot(list(range(nb_iter+1)),cost_histo,color="yellow")
            plt.title("Coûts pour la classification par régression logistique \
            \n au fil des itérations de l'algorithme du gradient")
            plt.xlabel("Nombre d'itérations")
            plt.show()

    def score(self,data,labels,s):
        return (self.predict(data,s)==labels).mean()
