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
from . import Classifiers as cl

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
#_Entrainement

def entrainement(n, label, perceptron, pourcentage=60, show=False) :

    mean = 0
    meanList = []
    for i in range(n) :
        train, test = split(label, pourcentage)
        perceptron.train(train) 
        acc = perceptron.accuracy(test)
        mean += acc
        if(show):
            print(str(i) + " entrainement")
            print("Accuracy "+str(acc)+"%\n")

        meanList.append(acc)
    mean = mean /n
    vari = np.var(meanList)
    print("Mean accuracy",str(mean))
    print("Variance accuracy", str(vari))

    return (mean, vari)





# Super_entrainement

def super_entrainement(n, label, perceptron, pourcentage=60, show=False) :
    x = []
    y = []
    mean = 0
    meanList = []
    for i in range(n) :
        train, test = split(label, pourcentage)
        perceptron.train(train) 
        acc = perceptron.accuracy(test)
        mean += acc
        if(show):
            print(str(i) + " entrainement")
            print("Accuracy "+str(acc)+"%\n")
        y.append(acc)
        x.append(i)
        meanList.append(acc)
    mean = mean /n
    vari = np.var(meanList)
    print("Mean accuracy",str(mean))
    print("Variance accuracy", str(vari))
    plt.plot(x,y)
    plt.xlabel('N')
    plt.ylabel('Accuracy')
    plt.title('performances accuracy')
    plt.legend()
    plt.show()
    plot_frontiere(test,perceptron)
    plot2DSet(test)
    return (mean, vari)


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



def bestClassifier(en, method, caracteristics,learningRate, criterion):
    #Ordre : 0 = KNN, 1: Random, 2: PerceptronKernel, 3:Gradient Stochastique, 4:Stochastique Kernel
    nb = 0
    while(nb < 4 ):
        if nb ==  0 :
            classifier = "KNN"
        elif nb == 1:
            classifier = "Classifier Random"
        elif nb == 2 :
            classifier = "Classifier PercepetronKernel"
        elif nb == 3 :
            classifier = "Classifier Gradient Stochastique"
        elif nb == 4 :
            classifier = "Classifier Stochastique Kernel"
        print("init",classifier,"\n\n")
        df = en.toDataFrame(method,criterion)
        lis = np.arange(len(df))
        for c1 in range(len(caracteristics)):
            for c2 in range(c1+1, len(caracteristics)):
                une_base = ls.LabeledSet(2)
                ca1 = caracteristics[c1]
                ca2 = caracteristics[c2]

                indice = np.random.permutation(lis)
                indice = indice[:1000]
                for i in indice:
                    une_base.addExample([df.iloc[i][ca1], df.iloc[i][c2]]
                                , df.iloc[i]['target'])
                if nb ==  0 :
                    cla = cl.ClassifierKNN(une_base.getInputDimension(),3)
                elif nb == 1 :
                    k= cl.KernelPoly()
                    cla= cl.ClassifierPerceptronKernel(6,learningRate,k)
                elif nb == 2 :
                    cla = cl.ClassifierGradientStochastique(une_base.getInputDimension(), learningRate)
                elif nb == 3 :
                    k = cl.KernelPoly()
                    cla = cl.ClassifierGradientStochastiqueKernel(6, learningRate, k)
                if ( (c1 == 0) and (c2 == 1) and(nb == 0) ):
                    maxi_mean, mini_vari = entrainement(25, une_base,cla, 40)
                    minica1 = ca1
                    minica2 = ca2
                    criterion = caracteristics[c1]
                    clamini = cla

                mean, vari = entrainement(25, une_base,cla, 40)
                if ( (vari < mini_vari) and (mean> maxi_mean) ):
                    mini_vari = vari
                    maxi_mean = mean
                    minica1 = ca1
                    minica2 = ca2
                    clamini = cla
                    classifiermini = classifier
                if( (c1==len(caracteristics) -1) and (c2 == len(caracteristics)-1) ):
                    print("\n",classifier,"done")
        print("\n\n",classifier,"done\n\n")
        nb+=1
    print("\n\nClassifier chosen",classifiermini,"Chosen criterion",criterion, "\nParams :", minica1, "and", minica2, "\nMean :",maxi_mean,
          "\nVariance", mini_vari)
    df = en.toDataFrame(method,criterion)
    une_base = ls.LabeledSet(2)
    for i in range(1000):
        une_base.addExample([df.iloc[i][minica1], df.iloc[i][minica2]]
                    , df.iloc[i]['target'])
    mean, vari = super_entrainement(25, une_base,clamini, 40)


    
def bestRegressor(eng, method, caracteristics, learningRate, criterion):
    nb = 0
    while(nb < 2):
        if nb == 0:
            classifier = "Classifier Gradient Batch"
        else :
            classifier = "Classifier Gradient Batch Kernel"
        print("init", classifier, "\n\n")
        
        df = eng.toDataFrame(method, criterion)
        lis = np.arange(len(df))
        for c1 in range(len(caracteristics)):
            for c2 in range(c1+1, len(caracteristics)):
                une_base = ls.LabeledSet(2)
                ca1 = caracteristics[c1]
                ca2 = caracteristics[c2]
                
                indice = np.random.permutation(lis)
                indice = indice[:1000]
                
                
                for i in indice :
                    une_base.addExample([df.iloc[i][ca1], df.iloc[i][ca2]], df.iloc[i]['target'])
                
                
                if nb == 0 :
                    cla = cl.ClassifierGradientBatch(une_base.getInputDimension(), learningRate)
                    print("ok cgb")
                else :
                    k = cl.KernelPoly()
                    cla = cl.ClassifierGradientBatchKernel(6, learningRate, k)
                    
                if( (c1 == 0) and (c2== 1) and (nb == 0)):
                    maxi_mean , mini_var = entrainement(100, une_base, cla, 50)
                    minica1 = ca1
                    minica2 = ca2
                    clamini = cla
                    classifiermini = classifier
                    
                mean, var = entrainement(100, une_base, cla, 50)
                
                if( (mini_var > var) and (maxi_mean < mean) ):
                    mini_var = var
                    maxi_mean = mean
                    minica1 = ca1
                    minica2 = ca2
                    clamini = cla
                    classifiermini = classifier
                    
        
        print('\n\n',classifier, "done\n\n")
        nb += 1
    print("\n\n Classifier chosen", classifiermini, "\nParameters :", minica1,",", minica2, "\nMean ", maxi_mean,
         "\nVariance :", mini_var)
    une_base = ls.LabeledSet(2)
    indice = np.random.permutation(lis)
    indice = indice[:1000]
    for i in indice:
        une_base.addExample([df.iloc[i][minica1], df.iloc[i][minica2]], df.iloc[i]['target'])
    mean , var = super_entrainement(100, une_base, clamini, 40)



