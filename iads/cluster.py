import scipy.cluster.hierarchy
import numpy as np
import math
import matplotlib.pyplot as plt

def normalisation(df):
    return (df-df.min())/(df.max()-df.min())

def dist_euclidienne_vect(v1,v2):
    s = 0
    for i in range(len(v1)) :
        s +=(v1[i] - v2[i])**2
    return  math.sqrt(s)

def dist_manhattan_vect(v1,v2):
    s = 0
    for i in range(len(v1)):
        s += abs(v1[i] - v2[i])
    return s

def dist_vect(chaine, v1, v2) :
    if chaine == "euclidienne":
        return dist_euclidienne_vect(v1,v2)
    return dist_manhattan_vect(v1,v2)

def dist_groupes(chaine, g1, g2):
    c1 = centroide(g1)
    c2= centroide(g2)
    if chaine == "euclidienne" :
        return dist_euclidienne_vect(c1,c2)
    return dist_manhattan_vect(c1,c2)

def centroide(matrice) :
    if (len(matrice.shape) == 1) :
        return matrice
    return np.mean(matrice,axis=0)

def initialise(matrice):
    d = {}
    #print(matrice)
    for i in range(len(matrice)) :
        d[i] = matrice.iloc[i]
    return d

def fusionne(chaine, partition) :
    ke = list(partition.keys())
    mini = dist_groupes(chaine, partition[ke[0]], partition[ke[1]])
    cle1 = ke[0]
    cle2 = ke[1]
    for i in ke :
        for j in ke :
            if i != j :
                d = dist_groupes(chaine,partition[i], partition[j])
                if mini >= d :
                    mini = d
                    cle1 = i
                    cle2 = j
    partition[i+1] = np.vstack([partition[cle1],partition[cle2]])
    del partition[cle1]
    del partition[cle2]


    print("Fusion de " + str(cle1) + " et " + str(cle2) + " pour une distance de " + str(mini))
    return (partition, cle1, cle2, mini)

def clustering_hierarchique(data, chaine) :
    matrice = normalisation(data)
    # initialisation
    courant = initialise(matrice)       # clustering courant, au départ:s données data_2D normalisées
    M_Fusion = []                        # initialisation
    while len(courant) >=2:              # tant qu'il y a 2 groupes à fusionner
        new,k1,k2,dist_min = fusionne(chaine,courant)
        if(len(M_Fusion)==0):
            M_Fusion = [k1,k2,dist_min,2]
        else:
            M_Fusion = np.vstack( [M_Fusion,[k1,k2,dist_min,2] ])
        courant = new

    return M_Fusion

def dendogramme(ch):
    # Paramètre de la fenêtre d'affichage:
    plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
    plt.title('Dendrogramme', fontsize=25)
    plt.xlabel('Exemple', fontsize=25)
    plt.ylabel('Distance', fontsize=25)

    # Construction du dendrogramme à partir de la matrice M_Fusion:
    scipy.cluster.hierarchy.dendrogram(
        ch,
        leaf_font_size=18.,  # taille des caractères de l'axe des X
    )

    # Affichage du résultat obtenu:
    plt.show()
