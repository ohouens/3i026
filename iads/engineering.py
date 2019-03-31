import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalisation(df):
    return (df-df.min())/(df.max()-df.min())

class Engineering():
    def __init__(self, name):
        self.name = name+"Engineering"
        print(self.name, "init in process")

    def reverse(self):
        raise NotImplementedError("Please Implement this method")

    def toDataFrame(self):
        raise NotImplementedError("Please Implement this method")


class ActorsEngineering(Engineering):
    def __init__(self, base):
        super().__init__("Actors")
        self.actors = {}
        self.actorsReversed = {}
        self.casting = {}
        self.playedMovies = {}
        self.playedMoviesReversed = {}
        self.averageIncomes = {}
        self.favoriteGenres = {}
        #on initialise actors, actorsReversed et playedMovies
        for lista in base:
            for a in lista:
                # ajoute le nombre de film dans lequel l'acteur à joué
                if a['name'] in self.playedMovies.keys():
                    self.playedMovies[a['name']] += 1
                else:
                    self.playedMovies[a['name']] = 1
                # affecte une valeur à une clé si la clé n'est pas utilisée
                res = self.actors.setdefault(a['name'], len(self.actors))
                if res == len(self.actors)-1:
                    self.actorsReversed[len(self.actors)-1] = a['name']
        print(self.name, "init successful")

    def reverse(self):
        # on index les acteurs par nombre de films dans lequels ils ont joué
        for k, v in self.playedMovies.items():
            if v in self.playedMoviesReversed.keys():
                self.playedMoviesReversed[v].append(k)
            else:
                self.playedMoviesReversed[v] = [k]

    def toDataFrame(self, column=[]):
        pass


class MoviesEngineering():
    def __init__(self, base):
        super().__init__("Movies")
        self.genres = {}
        for mtg in base['genres']:
            g = mtg.split('|')
            for a in g:
                self.genres.add(a)
        print(self.name,  "init successful")

    def reverse(self):
        pass

    def toDataFrame(self, column=[]):
        pass
