# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions utiles pour les TDTME de 3i026

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importation de LabeledSet
from . import LabeledSet as ls

def plot2DSet(set):
    """ LabeledSet -> NoneType
        Hypothèse: set est de dimension 2
        affiche une représentation graphique du LabeledSet
        remarque: l'ordre des labels dans set peut être quelconque
    """
    S_pos = set.x[np.where(set.y == 1),:][0]      # tous les exemples de label +1
    S_neg = set.x[np.where(set.y == -1),:][0]     # tous les exemples de label -1
    plt.scatter(S_pos[:,0],S_pos[:,1],marker='o') # 'o' pour la classe +1
    plt.scatter(S_neg[:,0],S_neg[:,1],marker='x') # 'x' pour la classe -1

def plot_frontiere(set,classifier,step=50):
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
    
# ------------------------ 

def createGaussianDataset(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ 
        rend un LabeledSet 2D généré aléatoirement.
        Arguments:
        - positive_center (vecteur taille 2): centre de la gaussienne des points positifs
        - positive_sigma (matrice 2*2): variance de la gaussienne des points positifs
        - negative_center (vecteur taille 2): centre de la gaussienne des points négative
        - negative_sigma (matrice 2*2): variance de la gaussienne des points négative
    - nb_points (int):  nombre de points de chaque classe à générer
"""
#TODO: A Compléter
    x, y = np.random.multivariate_normal(positive_center, positive_sigma, nb_points).T
    a, b = np.random.multivariate_normal(negative_center, negative_sigma, nb_points).T
    label = ls.LabeledSet(2)
    for i in range(len(a)):
        label.addExample(np.array([x[i], y[i]]), 1)
        label.addExample(np.array([a[i], b[i]]),-1)
    return label

    raise NotImplementedError("Please Implement this method")
    
# Exemple d'utilisation de utils

the_set = createGaussianDataset(np.array([1,1]),np.array([[1,0],[0,1]]),np.array([-1,-1]),np.array([[1,0],[0,1]]),100)



def super_entrainement(n, label, perceptron, show=False) :
    x = []
    y = []
    for i in range(n) :
        train, test = split(label)
        perceptron.train(train)
        train = perceptron.accuracy(test)
        if(show):
            print(str(i) + " entrainement")
            print("Accuracy "+str(train)+"%\n")
        y.append(train)
        x.append(i)
    plt.plot(x,y)
    plt.xlabel('N')
    plt.ylabel('Accuracy')
    plt.title('performances accuracy')
    plt.legend()
    plt.show()
    plot_frontiere(test,perceptron)
    plot2DSet(test)

    
    # Fonction pour afficher le LabeledSet
def affiche_base(LS):
    """ LabeledSet
        affiche le contenu de LS
    """
    for i in range(0,LS.size()):
        print("Exemple "+str(i))
        print("\tdescription : ",LS.getX(i))
        print("\tlabel : ",LS.getY(i))
    return

def split(label,pourcentage=60) :

    size = label.size()

    label_train = ls.LabeledSet(label.getInputDimension())
    label_test = ls.LabeledSet(label.getInputDimension())
    
    indice = np.arange(label.size())
    temoin = np.random.permutation(indice)
    
    for i in range (temoin.size) :
        if (i < pourcentage*len(temoin)/100):
            label_train.addExample(label.getX(temoin[i]), label.getY(temoin[i]))
        else : 
            label_test.addExample(label.getX(temoin[i]), label.getY(temoin[i]))

    return (label_train, label_test)
