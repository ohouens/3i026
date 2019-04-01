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
        self.averageRating = {}
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


class GenresEngineering(Engineering):
    def __init__(self, base, complement):
        super().__init__("Genres")
        self.genres = {}
        self.nbFilms = {}
        self.averageRating = {}
        self.ratingCount = {}
        for i in range(len(base)):
            genre = base.iloc[i]['genres'].split('|')
            film = self.linkFilm(i, base.iloc[i]['movieId'], complement)
            for g in genre:
                #on initialise les genres possibles
                if g in self.genres.keys():
                    self.genres[g].append(base.iloc[i]['movieId'])
                else:
                    self.genres[g] = [base.iloc[i]['movieId']]
                #on compte les tailles
                if g in self.nbFilms.keys():
                    self.nbFilms[g] += 1
                else:
                    self.nbFilms[g] = 1
                #on ajoute les nombres de votes
                if g in self.ratingCount.keys():
                    self.ratingCount[g] += film['vote_count']
                else:
                    self.ratingCount[g] = film['vote_count']
                #on ajoute les notes moyennes
                if g in self.averageRating.keys():
                    self.averageRating[g] += film['vote_average']*film['vote_count']
                else:
                    self.averageRating[g] = film['vote_average']*film['vote_count']
        #on effectue la moyenne des note moyennes(cela n'a pas de sens mais on fait avec ce que l'on a)
        for k in self.genres.keys():
            self.averageRating[k] /= self.ratingCount[k]
        print(self.name,  "init successful")

    def linkFilm(self, i, movieId, complement):
        links, films = complement
        inter = int(links.loc[links["movieId"] == movieId]["tmdbId"])
        if(films[i]['id'] == inter):
            return films[i]
        else:
            #print("miss")
            for film in films:
                if film['id'] == inter:
                    return film
            print("FAIL")
            exit(0)

    def toDataFrame(self, column=[]):
        df = {}
        df["name"] = []
        df["quantite"] = []
        df["note"] = []
        df["engagement"] = []
        for k in self.genres.keys():
            df["name"].append(k)
            df["quantite"].append(self.nbFilms[k])
            df["note"].append(self.averageRating[k])
            df["engagement"].append(self.ratingCount[k])
        return pd.DataFrame.from_dict(df)
