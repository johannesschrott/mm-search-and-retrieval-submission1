import pandas as pd
import statistics
import numpy as np
from evaluation.Genres import Genres
from collections import defaultdict


def checkDataTypeOfGenreColumn(preLoadedGenres: pd.DataFrame):
    # preLoadedGenres[preLoadedGenres.apply(lambda s: i in s for i in genresOfQuerySong).any(axis=1)]
    print('preLoadedGenres: ')
    print(preLoadedGenres)
    split_genres_df = pd.DataFrame(columns=['songId'])
    split_genres_df['genres'] = split_genres_df.apply(lambda row: preLoadedGenres['genre'].strip(" ").split(', '),axis=1)
    print('split genres per song:', split_genres_df)
    print('genres of song 2', preLoadedGenres.iloc[2])
    # ==
    # print('genres of song 1', preLoadedGenres.iloc[2][0])
    # dataTypeGenre = preLoadedGenres.dtypes # == object (not a list but rather a string)
    # print('datatype of genre column: ', dataTypeGenre)

def getGenresPerSongAsArray(preLoadedGenres: pd.DataFrame):
    # preLoadedGenres[preLoadedGenres.apply(lambda s: i in s for i in genresOfQuerySong).any(axis=1)]
    # print('preLoadedGenres: ', preLoadedGenres)
    # print('tracks count: ', len(preLoadedGenres))
    split_genres_df = pd.DataFrame(columns=['songId'])
    # axis 0 = index, axis 1 = columns (see documentation of apply)
    #split_genres_df['genres'] = split_genres_df.apply(lambda row: preLoadedGenres['genre']).any(axis=1)
    #print('split genres per song:', split_genres_df)
    split_genres_df = preLoadedGenres.apply(lambda row: row['genre'].strip(" []").split(', '), axis = 1)
    #print('split: ', split_genres_df)
    #print('2nd genre of second song: ', split_genres_df.iloc[1][0])
    #print(split_genres_df.shape)
    most_f_genres = list()
    for i in split_genres_df:
        for x in i:
            most_f_genres.append(x)
    genres_dict = defaultdict(int)
    genres_dict = dict.fromkeys(most_f_genres)
    for i in genres_dict.keys():
        genres_dict.update({i: 0})
        #print('item: ', i)
    for i in most_f_genres:
        genres_dict[i] += 1
        #print('item: ', i)

    sorted_genres_count = sorted(genres_dict.items(), key=lambda x:x[1], reverse=True)

    #print(sorted_genres_count)

    avgGenresCount = len(most_f_genres) / len(preLoadedGenres)
    avgTracksSharing1Genre = len(preLoadedGenres) / len(most_f_genres) * len(preLoadedGenres)

    meanTracksPerGenre = statistics.mean(genres_dict.values())
    medianTracksPerGenre = statistics.median(genres_dict.values())

    return sorted_genres_count, avgGenresCount, avgTracksSharing1Genre, meanTracksPerGenre, medianTracksPerGenre


# dataframe of all genres with only one row - add +1 in column of certain genre, if found in the next track's genre list; add genre as new column if not yet there with default counter 1
# OR more performant?:
# dataframe where index is the genre name, and data the counter of appearance. add +1 to list entry if found and index/id existent, otherwise add new list entry with counter 1
def getStats(genresArray: list, howMany: int):
    GenresPerSong, avgGenresCount, avgTracksSharing1Genre, meanTracksPerGenre, medianTracksPerGenre = getGenresPerSongAsArray(genresArray)
    return GenresPerSong[:howMany], avgGenresCount, avgTracksSharing1Genre, meanTracksPerGenre, medianTracksPerGenre


