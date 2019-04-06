import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalisation(df):
    return (df-df.min())/(df.max()-df.min())

def toTarget(list, method):
    final = []
    if(method=="median"):
        temoin = np.median([list])
    elif(method=="mean"):
        temoin = np.mean([list])
    else:
        temoin = 0
    for inter in list:
        target = -1
        if inter > temoin:
            target = 1
        final.append(target)
    return final

class Engineering():
    def __init__(self, name):
        self.name = name+"Engineering"
        self.df = {}
        self.index = []
        self.target = []
        print(self.name, "init in process")

    def toDataFrame(self, method="median"):
        df = normalisation(pd.DataFrame(self.df, index=self.index))
        df["target"] = toTarget(self.target, method)
        return df


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


class GenresEngineering(Engineering):
    def __init__(self, base, complement):
        super().__init__("Genres")
        genres = {}
        nbFilms = {}
        averageRating = {}
        ratingCount = {}
        indice = base.index.values.tolist()
        for i in range(len(base)):
            genre = base.iloc[i]['genres'].split('|')
            film = self.linkFilm(i, base.iloc[i]['movieId'], complement)
            for g in genre:
                #on initialise les genres possibles
                if g in genres.keys():
                    genres[g].append(base.iloc[i]['movieId'])
                else:
                    genres[g] = [base.iloc[i]['movieId']]
                #on compte les tailles
                if g in nbFilms.keys():
                    nbFilms[g] += 1
                else:
                    nbFilms[g] = 1
                #on ajoute les nombres de votes
                if g in ratingCount.keys():
                    ratingCount[g] += film['vote_count']
                else:
                    ratingCount[g] = film['vote_count']
                #on ajoute les notes moyennes
                if g in averageRating.keys():
                    averageRating[g] += film['vote_average']*film['vote_count']
                else:
                    averageRating[g] = film['vote_average']*film['vote_count']
        #on effectue la moyenne des note moyennes(cela n'a pas de sens mais on fait avec ce que l'on a)
        for k in genres.keys():
            averageRating[k] /= ratingCount[k]
        #on rempli le dictionnaire
        self.df["quantite"] = []
        self.df["engagement"] = []
        for k in genres.keys():
            self.target.append(averageRating[k])
            self.index.append(k)
            self.df["quantite"].append(nbFilms[k])
            self.df["engagement"].append(ratingCount[k])
        print(self.name,  "init successful")

    def linkFilm(self, i, movieId, complement):
        links, films = complement
        inter = int(links.loc[links["movieId"] == movieId]["tmdbId"])
        """
        for i in indice:
            if films[i]['id'] == inter:
                indice.remove(i)
                return films[i]
        """
        if(films[i]['id'] == inter):
            return films[i]
        else:
            #print("miss")
            for film in films:
                if film['id'] == inter:
                    return film
            print("FAIL")
            exit(0)
        print("FAIL", movieId, len(indice))
        exit(0)


class MoviesEngineering(Engineering):
    def __init__(self, base, complement):
        super().__init__("Movies")
        self.movies = {}
        self.mainActors = {}
        self.languages = {}
        self.popularity = {}
        print(name,  "init successful")
