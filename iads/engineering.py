import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

genres = ["Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy",
          "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]

genres_tmdb = ["Action", "Adventure" , "Animation", "Comedy", "Crime", "Documentary", "Drama",  "Family",
               "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie",
               "Thriller", "War", "Western"]

genres_tmdb_dict = {28 : "Action",12 : "Adventure", 16 : "Animation", 35 : "Comedy", 80: "Crime",
                    99: "Documentary",18 : "Drama",10751 : "Family",14 : "Fantasy", 36 : "History",
                    27 : "Horror", 10402 : "Music",9648 : "Mystery", 10749 : "Romance", 878 : "Science Fiction",
                    10770 : "TV Movie", 53 : "Thriller", 10752 : "War",  37 : "Western"}

genres_tmdb_dict_inv = {v: k for k, v in genres_tmdb_dict.items()}

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
    print(method, temoin)
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


class UtilsEngineering(Engineering):
    def __init__(self, base, complement):
        super().__init__("Utils")
        self.actors = {}
        self.actorsReversed = {}
        self.actorsPlayedMovies = {}
        #introduire complements
        acteurs = complement
        #operation de base 1
        for lista in acteurs:
            for a in lista:
                # affecte une valeur à une clé si la clé n'est pas utilisée
                res = self.actors.setdefault(a['name'], len(self.actors))
                if res == len(self.actors)-1:
                    self.actorsReversed[len(self.actors)-1] = a['name']
        #operation de base 2
        for i in self.actors.keys():
            self.actorsPlayedMovies[i] = {}
            for k in genres_tmdb_dict.keys() :
                self.actorsPlayedMovies[i][genres_tmdb_dict[k]] = 0
            self.actorsPlayedMovies[i]["Total"] = 0
        for i_film in range(len(base)) :
            desc_film = base[i_film]
            for act in range(len(acteurs[i_film])) :
                actorName = acteurs[i_film][act]["name"]
                self.actorsPlayedMovies[actorName]["Total"] += 1
                for id in desc_film["genre_ids"] :
                    genre = genres_tmdb_dict[id]
                    self.actorsPlayedMovies[actorName][genre] += 1
        print(self.name, "init successful")

    def toDataFrame(self, method):
        pass

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
        self.df["vote_count"] = []
        self.df["moy_main_actors"] = []
        self.df["original_language"] = []
        self.df["popularity"] = []
        #on introduit les complements
        plays, moy_act_films = complement
        #on effectue les operations de Base
        for i in range(len(base)):
            title = base[i]["original_title"]
            self.df["vote_count"].append(base[i]["vote_count"])
            acteurs = plays[title]
            acteurs = acteurs[0:5]
            genres_id = films[i]["genre_ids"]
            n = 0
            total = 0
            genres = []
            for g in genres_id:
                genres.append(genres_tmdb_dict[g])
            for g in genres:
                for act in acteurs:
                    n += moy_act_films[act][g]
                    total += 1
            if total == 0:
                self.df["moy_main_actors"].append(0)
            else:
                self.df["moy_main_actors"].append(n/total)
            self.df["original_language"].append(base[i]["original_language"])
            if "popularity" not in base[i].keys():
                self.df["popularity"].append(0)
            else:
                self.df["popularity"].append(base[i]["popularity"])
            self.target.append(base[i]["vote_average"])
            self.index.append(base[i]["title"])
        print(self.name,  "init successful")
