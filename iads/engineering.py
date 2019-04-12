import math
import copy
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
        print(self.name, "init in process")

    def toDataFrame(self, method="median", axis=''):
        cp = copy.deepcopy(self.df)
        if axis == '':
            temoin = list(cp)[-1]
        else:
            temoin = axis
        target = cp[temoin]
        del cp[temoin]
        stack = {}
        for k,v in self.df.items():
            if not (isinstance(v[0], int) or isinstance(v[0], float)):
                stack[k] = v
                del cp[k]
        result = normalisation(pd.DataFrame(cp, index=self.index))
        for k, v in stack.items():
            result[k] = v
        result["target"] = toTarget(target, method)
        return result


class UtilsEngineering(Engineering):
    def __init__(self, base, complement):
        super().__init__("Utils")
        self.actors = {}
        self.actorsReversed = {}
        self.actorsPlayedMovies = {}
        self.actorsMeanMovies = {}
        self.plays = {}
        self.languages = {}
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
        #operation de base 3
        for i_film in range(len(base)):
            name = base[i_film]["original_title"]
            self.plays[name] = []
            for a in acteurs[i_film] :
                self.plays[name].append(a["name"])
        #operation de base 4
        ke_act = self.actors.keys()
        ke_gen = genres_tmdb_dict.keys()
        for i in ke_act :
            self.actorsMeanMovies[i] = dict()
            for k in ke_gen :
                self.actorsMeanMovies[i][genres_tmdb_dict[k]] = 0
            self.actorsMeanMovies[i]["Total"] = 0
        for i_film in range(len(base)) :
            desc_film = base[i_film]
            vote = desc_film["vote_average"]
            for act in range(len(acteurs[i_film])) :
                actor_name = acteurs[i_film][act]["name"]
                self.actorsMeanMovies[actor_name]["Total"] += vote
                for id in desc_film["genre_ids"] :
                    genre = genres_tmdb_dict[id]
                    self.actorsMeanMovies[actor_name][genre] += vote
        ke_gen = list(genres_tmdb_dict_inv.keys())
        ke_gen.append("Total")
        for act in ke_act :
            for k in ke_gen :
                if(self.actorsMeanMovies[act][k] > 0):
                    self.actorsMeanMovies[act][k] = (self.actorsMeanMovies[act][k] / self.actorsPlayedMovies[act][k])
        #operation de base 5
        language = []
        cpt = 0
        for fi in base :
            la = fi["original_language"]
            if la not in language :
                language.append(la)
                self.languages[la] = cpt
                cpt +=1
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
        self.df["note"] = []
        for k in genres.keys():
            self.index.append(k)
            self.df["quantite"].append(nbFilms[k])
            self.df["engagement"].append(ratingCount[k])
            self.df["note"].append(averageRating[k])
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
        self.df["mean_main_actors"] = []
        self.df["original_language"] = []
        self.df["popularity"] = []
        self.df["note"] = []
        #on introduit les complements
        plays, actorsMeanMovies, languages = complement
        #on effectue les operations de Base
        for i in range(len(base)):
            title = base[i]["original_title"]
            self.df["vote_count"].append(base[i]["vote_count"])
            acteurs = plays[title]
            acteurs = acteurs[0:5]
            genres_id = base[i]["genre_ids"]
            n = 0
            total = 0
            genres = []
            for g in genres_id:
                genres.append(genres_tmdb_dict[g])
            for g in genres:
                for act in acteurs:
                    n += actorsMeanMovies[act][g]
                    total += 1
            if total == 0:
                self.df["mean_main_actors"].append(0)
            else:
                self.df["mean_main_actors"].append(n/total)
            la = base[i]["original_language"]
            nbr = languages[la] / len(languages)
            self.df["original_language"].append(nbr)
            if "popularity" not in base[i].keys():
                self.df["popularity"].append(0)
            else:
                self.df["popularity"].append(base[i]["popularity"])
            self.df["note"].append(base[i]["vote_average"])
            self.index.append(base[i]["title"])
        print(self.name,  "init successful")
