import math
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt

# Chargement des fichiers de donnees :
prices_pd = pd.read_csv("../TME-01-etudiants/data-01/Weed_Price.csv", parse_dates=[-1])
demography_pd = pd.read_csv("../TME-01-etudiants/data-01/Demographics_State.csv")
population_pd = pd.read_csv("../TME-01-etudiants/data-01/Population_State.csv")
les_etats = np.unique(prices_pd["State"].values)


def calcul():
	return -(1/3.0)*math.log(1/3.0, 2)-(2/3.0)*math.log(2/3.0, 2)

def moyenne(list):
	somme = 0
	for i in list:
		somme = somme + i
	return somme/len(list)*1.0

#print(calcul())
#print(type(prices_pd))
#print(type(demography_pd))
#print(type(population_pd))
#print(prices_pd.head())
#print(demography_pd.head())
#print(population_pd.tail())
print("la moyenne (qualite moyenne) est {} dollars".format(prices_pd["MedQ"].mean()))
print("la moyenne (qualite moyenne) est {} dollars".format(moyenne(prices_pd["MedQ"])))