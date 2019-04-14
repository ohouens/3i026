#---- Importation des fichiers 

from . import LabeledSet as ls
from . import Classifiers as cl

#----- Importations de librairies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

#-----



def shannon(ensemble):
    k = len(ensemble)
    if k == 1 :
        return 0
    somme = 0
    for i in range(k):
        if ensemble[i] > 0 :
            somme += ensemble[i]*math.log(ensemble[i], k)
    return - somme

def classe_majoritaire(label) :
    cpt_p = 0
    cpt_m = 0
    
    for i in range(label.size()) :
        if(label.getY(i) > 0) :
            cpt_p += 1
        else :
            cpt_m += 1
            
    if cpt_m > cpt_p :
        return -1
    return +1


def entropie(label) :
    plus = 0
    moins = 0
    for i in range(label.size()) :
        if(label.getY(i) > 0) :
            plus += 1 / label.size()
        else :
            moins += 1 / label.size()
            
    return shannon([plus, moins])


def discretise(LSet, col):
    """ LabelledSet * int -> tuple[float, float]
        Hypothèse: LSet.size() >= 2
        col est le numéro de colonne sur X à discrétiser
        rend la valeur de coupure qui minimise l'entropie ainsi que son entropie.
    """
    # initialisation:
    min_entropie = 1.1  # on met à une valeur max car on veut minimiser
    min_seuil = 0.0     
    # trie des valeurs:
    ind= np.argsort(LSet.x,axis=0)
    
    # calcul des distributions des classes pour E1 et E2:
    inf_plus  = 0               # nombre de +1 dans E1
    inf_moins = 0               # nombre de -1 dans E1
    sup_plus  = 0               # nombre de +1 dans E2
    sup_moins = 0               # nombre de -1 dans E2       
    # remarque: au départ on considère que E1 est vide et donc E2 correspond à E. 
    # Ainsi inf_plus et inf_moins valent 0. Il reste à calculer sup_plus et sup_moins 
    # dans E.
    for j in range(0,LSet.size()):
        if (LSet.getY(j) == -1):
            sup_moins += 1
        else:
            sup_plus += 1
    nb_total = (sup_plus + sup_moins) # nombre d'exemples total dans E
    
    # parcours pour trouver le meilleur seuil:
    for i in range(len(LSet.x)-1):
        v_ind_i = ind[i]   # vecteur d'indices
        courant = LSet.getX(v_ind_i[col])[col]
        lookahead = LSet.getX(ind[i+1][col])[col]
        val_seuil = (courant + lookahead) / 2.0;
        # M-A-J de la distrib. des classes:
        # pour réduire les traitements: on retire un exemple de E2 et on le place
        # dans E1, c'est ainsi que l'on déplace donc le seuil de coupure.
        if LSet.getY(ind[i][col])[0] == -1:
            inf_moins += 1
            sup_moins -= 1
        else:
            inf_plus += 1
            sup_plus -= 1
        # calcul de la distribution des classes de chaque côté du seuil:
        nb_inf = (inf_moins + inf_plus)*1.0     # rem: on en fait un float pour éviter
        nb_sup = (sup_moins + sup_plus)*1.0     # que ce soit une division entière.
        # calcul de l'entropie de la coupure
        val_entropie_inf = shannon([inf_moins / nb_inf, inf_plus  / nb_inf])
        val_entropie_sup = shannon([sup_moins / nb_sup, sup_plus  / nb_sup])
        val_entropie = (nb_inf / nb_total) * val_entropie_inf \
                       + (nb_sup / nb_total) * val_entropie_sup
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (min_entropie > val_entropie):
            min_entropie = val_entropie
            min_seuil = val_seuil
    return (min_seuil, min_entropie)



def divise(LSet, att, seuil):
    result1 = ls.LabeledSet(LSet.input_dimension)
    result2 = ls.LabeledSet(LSet.input_dimension)
    for i in range(LSet.size()):
        if(LSet.getX(i)[att] > seuil):
            result1.addExample(LSet.getX(i), LSet.getY(i))
        else:
            result2.addExample(LSet.getX(i), LSet.getY(i))
    return (result2, result1)


import graphviz as gv
# Eventuellement, il peut être nécessaire d'installer graphviz sur votre compte:
# pip install --user --install-option="--prefix=" -U graphviz

    
class ArbreBinaire:
    def __init__(self):
        self.attribut = None   # numéro de l'attribut
        self.seuil = None
        self.inferieur = None # ArbreBinaire Gauche (valeurs <= au seuil)
        self.superieur = None # ArbreBinaire Gauche (valeurs > au seuil)
        self.classe = None # Classe si c'est une feuille: -1 ou +1
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille """
        return self.seuil == None
    
    def ajoute_fils(self,ABinf,ABsup,att,seuil):
        """ ABinf, ABsup: 2 arbres binaires
            att: numéro d'attribut
            seuil: valeur de seuil
        """
        self.attribut = att
        self.seuil = seuil
        self.inferieur = ABinf
        self.superieur = ABsup
    
    def ajoute_feuille(self,classe):
        """ classe: -1 ou + 1
        """
        self.classe = classe
        
    def classifie(self,exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple: +1 ou -1
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.inferieur.classifie(exemple)
        return self.superieur.classifie(exemple)
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir
            l'afficher
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.attribut))
            self.inferieur.to_graph(g,prefixe+"g")
            self.superieur.to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))
        
        return g

# ---------------------------

"""    
def construit_AD(LSet, epsilon):
    result = ArbreBinaire()
    if(entropie(LSet) <= epsilon) :
        result.ajoute_feuille(classe_majoritaire(LSet))
    else :
        mini_seuil, mini_ent = discretise(LSet, 0)
        indice = 0
        for i in range(1, LSet.input_dimension):
            seuil, e = discretise(LSet, i)
            if( e < mini_ent) :
                mini_ent = e
                indice = i
                mini_seuil = seuil
        inf, sup = divise(LSet, indice, mini_seuil)
        result.ajoute_fils(construit_AD(inf, epsilon), construit_AD(sup, epsilon), indice, mini_seuil)  
    return result  
"""

class ArbreDecision(cl.Classifier):
    # Constructeur
    def __init__(self,epsilon):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.racine = None
    
    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend 0 (classe -1) ou 1 (classe 1)
        classe = self.racine.classifie(x)
        if (classe == 1):
            return(1)
        else:
            return(-1)
    
    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        # construction de l'arbre de décision 
        self.set=set
        self.racine = construit_AD(set,self.epsilon)

    # Permet d'afficher l'arbre
    def plot(self):
        gtree = gv.Digraph(format='png')
        return self.racine.to_graph(gtree)


#------


def construit_AD(LSet, epsilon):
    result = ArbreBinaire()
    if(entropie(LSet) < epsilon) :
        result.ajoute_feuille(classe_majoritaire(LSet))
    else :
        mini_seuil, mini_ent = discretise(LSet, 0)
        indice = 0
        for i in range(1, LSet.input_dimension):
            seuil, e = discretise(LSet, i)
            if( e < mini_ent) :
                mini_ent = e
                indice = i
                mini_seuil = seuil
        inf, sup = divise(LSet, indice, mini_seuil)
        gain = entropie(LSet) - mini_ent
        if (gain >= epsilon) :
            result.ajoute_fils(construit_AD(inf, epsilon), construit_AD(sup, epsilon), indice, mini_seuil)
        else :
            result.ajoute_feuille(classe_majoritaire(LSet))
    return result


def tirage(vx, m, r) :
    if (r == True) :
        t = []
        for i in range(m) :
            t.append(random.choice(vx))
        return t
    else :
        return random.sample(vx,m)
    
def echantillonLS(x, m, r):
    vx = [x for x in range(x.size())]
    t = tirage(vx, m, r)
    label = ls.LabeledSet(x.input_dimension)
    for i in t:
        label.addExample(x.getX(i), x.getY(i))
    return label

class ClassifierBaggingTree(cl.Classifier):

  
    def __init__(self, B, pourcentage, b, seuil):
        self.B = B
        self.pourcentage = pourcentage 
        self.b = b
        self.seuil = seuil
        self.ensemble = []
        
    def predict(self, x):
        plus = 0
        moins = 0
        for i in range(self.B) :
            if (self.ensemble[i].predict(x) > 0):
                plus += 1
            else :
                moins += 1
        if plus > moins :
            return 1
        return -1

    def train(self, labeledSet, epsilon=0) :
        for i in range(self.B):
            a = int(labeledSet.size()*self.pourcentage)
            e  = echantillonLS(labeledSet,a, self.b)
            
            decision = ArbreDecision(epsilon)
            decision.train(e)
            self.ensemble.append(decision)
#-------


class ClassifierBaggingTreeOOB(ClassifierBaggingTree):
    
    def __init__(self, B, pourcentage, b, seuil):
        super().__init__(B, pourcentage, b, seuil)
        self.taux = []
        self.X = []
        self.T = []
    
    def train(self, labeledSet, epsilon=0) :
        for i in range(self.B):
            a = int(labeledSet.size()*self.pourcentage)
            e  = echantillonLS(labeledSet,a, self.b)
            self.T.append(labeledSet)
            self.X.append(e)
            decision = ArbreDecision(epsilon)
            decision.train(e)
            self.ensemble.append(decision)
    
    def calcul(self):
        for i in range(1,self.B+1):
            foret = ClassifierBaggingTreeOOB(self.B, self.pourcentage, self.b, self.seuil)
            foret.train(self.X[i-1])
            self.taux.append(foret.accuracy(self.T[i-1]))
        return np.sum(self.taux)/self.B
    
    
#----- Random Forest


def construit_AD_aleatoire(LSet,epsilon,nbatt):
    result = ArbreBinaire()
    if(entropie(LSet) < epsilon) :
        result.ajoute_feuille(classe_majoritaire(LSet))
    else :
        res = []
        for j in range(nbatt):
            mini_seuil, mini_ent = discretise(LSet, 0)
            indice = 0
            for i in range(1, LSet.input_dimension):
                seuil, e = discretise(LSet, i)
                if( (e < mini_ent) and (i not in res)) :
                    mini_ent = e
                    indice = i
                    mini_seuil = seuil
            res.append(indice)
            inf, sup = divise(LSet, indice, mini_seuil)
            gain = entropie(LSet) - mini_ent
            if (gain >= epsilon) :
                result.ajoute_fils(construit_AD(inf, epsilon), construit_AD(sup, epsilon), indice, mini_seuil)
            else :
                result.ajoute_feuille(classe_majoritaire(LSet))
    return result 

class ArbreDecisionAleatoire (ArbreDecision):
        def __init__(self,epsilon, nbatt):
            # valeur seuil d'entropie pour arrêter la construction
            self.epsilon= epsilon
            self.racine = None
            self.nbatt = nbatt
        
        def train(self,set):
            # construction de l'arbre de décision 
            self.set=set
            self.racine = construit_AD_aleatoire(set,self.epsilon,self.nbatt)
            
            
            
class ClassifierRandomForest(ClassifierBaggingTreeOOB):
    
    def __init__(self, B, pourcentage, b, seuil):
        super().__init__(B, pourcentage, b, seuil)
        self.taux = []
        self.X = []
        self.T = []
        
    def predict(self, x):
        plus = 0
        moins = 0
        for i in range(self.B) :
            if (self.ensemble[i].predict(x) > 0):
                plus += 1
            else :
                moins += 1
        if plus > moins :
            return 1
        return -1

    def train(self, labeledSet, epsilon=0) :
        for i in range(self.B):
            a = int(labeledSet.size()*self.pourcentage)
            e  = echantillonLS(labeledSet,a, self.b)
            self.T.append(labeledSet)
            self.X.append(e)
            nbatt = np.random.randint(1, labeledSet.input_dimension+1)
            decision = ArbreDecisionAleatoire(epsilon, nbatt)
            decision.train(e)
            self.ensemble.append(decision)
            

