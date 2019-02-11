# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt


# Exemple d'utilisation de vstack (pour plus de détails, chercher la documentation sur le web)

vecteur_1 = np.array([0, 1.5, 4.2])
vecteur_2 = np.array([1.1, 3.8, 20.01])

vecteur_3 = vecteur_1 + vecteur_2

np.vstack( (vecteur_1, vecteur_2, vecteur_3) )



class LabeledSet:  
    """ Classe pour représenter un ensemble d'exemples (base d'apprentissage)
        Variables d'instance :
            - input_dimension (int) : dimension de la description d'un exemple (x)
            - nb_examples (int) : nombre d'exemples dans l'ensemble
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de LabeledSet
            Argument: 
                - intput_dimension (int) : dimension de x
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.nb_examples = 0
    
    def addExample(self,vector,label):
        """ Ajout d'un exemple dans l'ensemble
            Argument: 
                - vector ()
                - label (int) : classe de l'exemple (+1 ou -1)
            
        """
        if (self.nb_examples == 0):
            self.x = np.array([vector])
            self.y = np.array([label])
        else:
            self.x = np.vstack((self.x, vector))
            self.y = np.vstack((self.y, label))
        
        self.nb_examples = self.nb_examples + 1
    
    def getInputDimension(self):
        """ Renvoie la dimension de l'espace d'entrée
        """
        return self.input_dimension
    
    def size(self):
        """ Renvoie le nombre d'exemples dans l'ensemble
        """
        return self.nb_examples
    
    def getX(self, i):
        """ Renvoie la description du i-eme exemple (x_i)
        """
        return self.x[i]
    
    #
    def getY(self, i):
        """ Renvoie la classe de du i-eme exemple (y_i)
        """
        return(self.y[i])

# Exemple d'utilisation de LabeledSet

une_base = LabeledSet(2)        # définition d'une base pour contenir des exemples en 2D
une_base.addExample([0, 1],1)   # ajout de l'exemple (0, 1) de classe +1
une_base.addExample([2, 3],1)   # ajout de l'exemple (2, 3) de classe +1
une_base.addExample([1, 2],-1)  # ajout de l'exemple (1, 2) de classe -1
une_base.addExample([2, 2],-1)  # ajout de l'exemple (2, 2) de classe -1



def affiche_base(label) :
    taille = label.size()
    
    for i in range(taille) :
        print("Exemple "+str(i)+": ")
        print("description :"+str(label.getX(i)))
        print("label : "+str(label.getY(i)))



#affiche_base(une_base)

def plot2DSet(set):
    """ LabeledSet -> NoneType
        Hypothèse: set est de dimension 2
        affiche une représentation graphique du LabeledSet
        remarque: l'ordre des labels dans set peut être quelconque
    """
    S_pos = set.x[np.where(set.y == 1),:][0]      # tous les exemples de label +1
    S_neg = set.x[np.where(set.y == -1),:][0]     # tous les exemples de label -1
    plt.scatter(S_pos[:,0],S_pos[:,1],marker='o') # 'o' pour la classe +1
    plt.scatter(S_neg[:,0],S_neg[:,1],marker='x') # 'x' pour la classe -1
    plt.show()

# Par exemple :
#plot2DSet(une_base)

def createGaussianDataset(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    x, y = np.random.multivariate_normal(positive_center, positive_sigma, nb_points).T
    a, b = np.random.multivariate_normal(negative_center, negative_sigma, nb_points).T
    label = LabeledSet(2)
    for i in range(len(a)):
        label.addExample(np.array([x[i], y[i]]), 1)
        label.addExample(np.array([a[i], b[i]]),-1)
    return label

the_set = createGaussianDataset(np.array([1,1]),np.array([[1,0],[0,1]]),np.array([-1,-1]),np.array([[1,0],[0,1]]),10)

#print("Taille de la base jouet générée :", the_set.size(), "exemples")

# Affichage :
#plot2DSet(the_set)

class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        
        raise NotImplementedError("Please Implement this method")
    
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        s = 0
        for i in range(dataset.size()) :
            if (self.predict(dataset.getX(i)) * dataset.getY(i) > 0) :
                s += 1
        return (s *1.0 / dataset.size()) *100



class ClassifierRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    def __init__(self, input_dimension):
        self.w = np.random.rand(1,input_dimension)
    
    def predict(self, x):
        return np.vdot(x,self.w) 
        
    def train(self, labeledSet):
        print("Pas d'apprentissage pour ce classifieur")
    #TODO: définir le constructeur, et les méthodes predict et train

# Création d'un classifieur linéaire aléatoire de dimension 2:

un_classifieur = ClassifierRandom(2)

the_set = createGaussianDataset(np.array([1,1]),np.array([[1,0],[0,1]]),np.array([-1,-1]),np.array([[1,0],[0,1]]),100)
#print(un_classifieur.accuracy(the_set))

def plot_frontiere(set,classifier,step=10):
    """ LabeledSet * Classifier * int -> NoneType
        Remarque: le 3e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=set.x.max(0)
    mmin=set.x.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    plt.contourf(x1grid,x2grid,res,colors=["red","cyan"],levels=[-1000,0,1000])


plot_frontiere(the_set, un_classifieur)
#plot2DSet(the_set)



classifieur_random=ClassifierRandom(2)

plot_frontiere(the_set,classifieur_random)
#plot2DSet(the_set)

class ClassifierKNN(Classifier):
    #TODO
    def __init__(self, input_dimension, k):
        self.w = np.random.rand(1,input_dimension)
        self.k = k
        
    def predict(self, x):
        l = []
        for i in range(self.labeledSet.size()) :
            l.append(np.linalg.norm(x - self.labeledSet.getX(i)))
        t = np.argsort(np.array(l))
        tr = t.tolist()
        #print(t)
        #print(tr)
        plus = []
        moins = []
        for i in range(self.k):
            #print(tr[i])
            #print(l[tr[i]])
            if(self.labeledSet.getY(tr[i]) > 0):
                plus.append(1)
            else:
                moins.append(-1)
        if(len(plus) > len(moins)):
            return 1
        return -1
        
        
    def train(self, labeledSet):
        self.labeledSet = labeledSet

# Exemple d'utilisation :
knn = ClassifierKNN(2,1)
knn.train(the_set)

plot_frontiere(the_set,knn,20)
#plot2DSet(the_set)

def protocole(n) :
    the_set = createGaussianDataset(np.array([1,1]),np.array([[1,0],[0,1]]),np.array([-1,-1]),np.array([[1,0],[0,1]]),100)
    the_set_train = createGaussianDataset(np.array([1,1]),np.array([[1,0],[0,1]]),np.array([-1,-1]),np.array([[1,0],[0,1]]),100)
    
    knn = ClassifierKNN(2, n)
    knn.train(the_set_train)
    un_classifieur = ClassifierRandom(2)
    print(un_classifieur.accuracy(the_set))
    print(knn.accuracy(the_set))
    
#protocole(3)

def performance(nombre):
    
    the_set = createGaussianDataset(np.array([1,1]),np.array([[1,0],[0,1]]),np.array([-1,-1]),np.array([[1,0],[0,1]]),100)
    the_set_train = createGaussianDataset(np.array([1,1]),np.array([[1,0],[0,1]]),np.array([-1,-1]),np.array([[1,0],[0,1]]),100)
    for i in range(nombre) :
        knn = ClassifierKNN(2, n)
        knn.train(the_set_train)
