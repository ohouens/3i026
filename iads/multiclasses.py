"""
Package: iads
Fichier: multiclasses.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""
#Importations des fichiers nécessaires :
from . import LabeledSet as ls
from . import Classifiers as cl
from . import LabeledSet as ls


# Import de packages externes
import numpy as np
import pandas as pd


class ClassifierMultiClasses():
    """ Classe pour représenter un classifieur multiclasses qui ramène en problème binaire.
    Utilise le classifier ClassifierPerceptronKernel.
    """
    def __init__(self, listDfs, listBases, listClas):
        self.bases = listBases
        self.classifiers = listClas
        self.listDataFrames = listDfs
        
    def predict(claId, x):
        k = cl.KernelPoly()
        y = k.transform(x)
        return np.dot(self.classifiers[claId].w, y) > 0

    def predScores(movieId, criterion1, criterion2):
        """ Rend l'id du meilleur classifier dans la liste de classifiers de la classe"""
        coord = [self.listDataFrames[0].iloc[movieId][criterion1],self.listDataFrames[0].iloc[movieId][criterion2]]
        best = -1
        bestId = -1
        for i in range(len(self.listClassifiers)):
            p = self.predict(i, coord)
            if p > best :
                best = p
                bestId = i
        return bestId


    def add(self, cla):
        """ Permet d'ajouter un classifier déjà entrainé """
        self.classifiers.append(cla)
# ---------------------------