# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd

# ---------------------------
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

# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    #TODO: A Compléter
    
     def __init__(self, input_dimension):
        self.w = np.random.rand(1,input_dimension)
    
    def predict(self, x):
        return np.vdot(x,self.w) 
        
    def train(self, labeledSet):
        print("Pas d'apprentissage pour ce classifieur")
    
# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
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
# ---------------------------
