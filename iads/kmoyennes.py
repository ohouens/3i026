# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: kmoyennes.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions pour les k-moyennes

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt

import math
import random

# ---------------------------
# Dans ce qui suit, remplacer la ligne "raise.." par les instructions Python
# demandées.
# ---------------------------

# Normalisation des données :

# ************************* Recopier ici la fonction normalisation()
def normalisation(DF):
    """ DataFrame -> DataFrame
        rend le dataframe obtenu par normalisation des données selon 
             la méthode vue en cours 8.
    """
    return (DF-DF.min())/(DF.max()-DF.min())

# -------
# Fonctions distances

# ************************* Recopier ici la fonction dist_vect()
def dist_vect(s1, s2):
    """ Series**2 -> float
        rend la valeur de la distance euclidienne entre les 2 vecteurs
    """
    s = 0
    dimension = list(s1.to_frame().T)
    for d in dimension:
        s += (s2[d]-s1[d])**2
    return math.sqrt(s)

# -------
# Calculs de centroïdes :
# ************************* Recopier ici la fonction centroide()
def centroide(df):
    """ DataFrame -> DataFrame
        Hypothèse: len(M) > 0
        rend le centroïde des exemples contenus dans M
    """
    if(len(df) == 1):
        return df
    return df.mean().to_frame().T

# -------
# Inertie des clusters :
# ************************* Recopier ici la fonction inertie_cluster()
def inertie_cluster(df):
    """ DataFrame -> float
        DF: DataFrame qui représente un cluster
        L'inertie est la somme (au carré) des distances des points au centroide.
    """
    jk = 0
    uk = centroide(df)
    for i in range(len(df)):
        jk += dist_vect(df.iloc[i], uk)**2
    return jk


# -------
# Algorithmes des K-means :
# ************************* Recopier ici la fonction initialisation()
def initialisation(k, df):
    """ int * DataFrame -> DataFrame
        K : entier >1 et <=n (le nombre d'exemples de DF)
        DF: DataFrame contenant n exemples
    """
    indice = [k for k in range(len(df))]
    random.shuffle(indice)
    indice = indice[:k]
    result = pd.DataFrame(df.iloc[indice])
    return result


# -------
# ************************* Recopier ici la fonction plus_proche()
def plus_proche(exemple, df):
    """ Series * DataFrame -> int
        Exe : Series contenant un exemple
        Centres : DataFrame contenant les K centres
    """
    mini = dist_vect(exemple, df.iloc[0])
    indice = 0
    for i in range(len(df)):
        if dist_vect(exemple, df.iloc[i]) < mini:
            mini = dist_vect(exemple, df.iloc[i])
            indice = i
    return indice

# -------
# ************************* Recopier ici la fonction affecte_cluster()
def affecte_cluster(df, ens):
    """ DataFrame * DataFrame -> dict[int,list[int]]
        Base: DataFrame contenant la base d'apprentissage
        Centres : DataFrame contenant des centroides
    """
    d = dict()
    for i in range(len(ens)) :
        d[i] = []
    for i in range(len(df)) :
        d[plus_proche(df.iloc[i], ens)].append(i)
    return d

# -------
# ************************* Recopier ici la fonction nouveaux_centroides()
def nouveaux_centroides(df, mat):
    """ DataFrame * dict[int,list[int]] -> DataFrame
        Base : DataFrame contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    ke = list(mat.keys())
    ensemble = pd.DataFrame(centroide(df.iloc[mat[ke[0]]]))
    ke.pop(0)
    for k in ke :
        ensemble = pd.concat([ensemble, centroide(df.iloc[mat[k]])], ignore_index=True)
    return ensemble

# -------
# ************************* Recopier ici la fonction inertie_globale()
def inertie_globale(df, mat):
    """ DataFrame * dict[int,list[int]] -> float
        Base : DataFrame pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    s=0
    ke = list(mat.keys())
    for k in ke :
        s+= inertie_cluster(df.iloc[mat[k]])
    return s
# -------
# ************************* Recopier ici la fonction kmoyennes()
def kmoyennes(k, df, epsilon, iter_max):
    """ int * DataFrame * float * int -> tuple(DataFrame, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : DataFrame pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    cen_test = initialisation(k,df)
    mat = affecte_cluster(df, cen_test)
    j = inertie_globale(df, mat)
    new_cen = nouveaux_centroides(df,mat)
    
    mat2= affecte_cluster(df, new_cen)
    new_cent2 = nouveaux_centroides(df, mat2)
    j2 = inertie_globale(df, mat2)
    
    i = 2
    while((abs(j2 - j)) and (i<= iter_max)) :
          j = j2
          cen_test = new_cen
          mat = mat2
          mat2 = affecte_cluster(df, cen_test)
          new_cent2 = nouveaux_centroides(df, mat2)
          j2 = inertie_globale(df, mat2)
          i += 1
    return (new_cent2, mat2)
# -------
# Affichage :
# ************************* Recopier ici la fonction affiche_resultat()
def affiche_resultat(df,les_centres,mat):
    """ DataFrame **2 * dict[int,list[int]] -> None
    """    
    # Remarque: pour les couleurs d'affichage des points, quelques exemples:
    # couleurs =['darkviolet', 'darkgreen', 'orange', 'deeppink', 'slateblue', 'orangered','y', 'g', 'b']
    # voir aussi (google): noms des couleurs dans matplolib
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.scatter(les_centres['X'],les_centres['Y'],color='r',marker='x')
    ke = list(mat.keys())
    for i in range(len(ke)) :
        plt.scatter(df.iloc[mat[ke[i]]]['X'],df.iloc[mat[ke[i]]]['Y'],color=colors[i%len(colors)])
    plt.show()
# -------
