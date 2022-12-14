# from rank_eval import Qrels, Run, evaluate
import pandas as pd
import statistics
import sklearn as sk
import numpy as np

import database
from evaluation.Genres import Genres

class Evaluation:

    # for task 1 relevant songs are those sharing at least 1 genre with the query song's genres
    # returns a data frame with id/index = song id and data row being the genre-list of the compared song (all the genres, not only the equal ones)
    # the querySong's id is excluded from the returned data frame!

    # TODO: this is called about once per evaluation metric per query song. -> potential for optimization
    def getGenreSharingSongsFromId(self, querySongId: str, preLoadedGenres: pd.DataFrame):
        genresOfQuerySong = preLoadedGenres['genre'][querySongId]


        genresOfQuerySong = genresOfQuerySong.replace('[', '')
        genresOfQuerySong = genresOfQuerySong.replace(']', '')
        genresOfQuerySong = genresOfQuerySong.replace('\'', '')
        genresOfQuerySong = genresOfQuerySong.strip(" ").split(', ')


        songsSharingGenre = preLoadedGenres[preLoadedGenres.apply(lambda s: i in s for i in genresOfQuerySong).any(axis=1)]
        songsSharingGenre.drop(index=querySongId)

        songsSharingGenre.index.drop_duplicates(keep='first')
        songsSharingGenreIds = songsSharingGenre.index


        #print('songs that share genres with query song (', querySongId, '): ', songsSharingGenre)
        #print('song ids that share genres with query song (', querySongId, '): ', songsSharingGenreIds)
        return songsSharingGenreIds


    def calcPrecisionForQuery(self, querySongId: str, retrievedIds: list[str], preLoadedGenres: pd.DataFrame):
        songsSharingGenreIds = self.getGenreSharingSongsFromId(querySongId, preLoadedGenres)

        #print('\ncalcPrecisionForQuery QUERY ID: ', querySongId, ' RETRIEVED SONG IDS count: ', len(retrievedIds))
        return self.calcPrecision(retrievedIds, songsSharingGenreIds)

    # k is a size/length. indices start at 0, i.e. runs through entries with indices [0..k-1] or through subset [0:k] (start at entry 0 with length k)
    def calcPrecisionAtKForQuery(self, querySongId: str, retrievedIds: list[str], preLoadedGenres: pd.DataFrame, k: int):
        songsSharingGenreIds = self.getGenreSharingSongsFromId(querySongId, preLoadedGenres)

        #print('\ncalcPrecisionAtKForQuery QUERY ID: ', querySongId, ' RETRIEVED SONG IDS: ', retrievedIds, 'k: ', k)
        return self.calcPrecisionAtK(retrievedIds, songsSharingGenreIds, k)

    # k must be > 0 (is a length/size/counter), maximum index of retrievedIds list entry is k-1
    def calcPrecisionAtK(self, retrievedIds: list[str], relevantIds: list[str], k: int): # querySongId must be excluded from both lists
        relevantIdx = pd.Index(relevantIds)
        if k > len(retrievedIds):
            #print(len(retrievedIds), 'results retrieved, which is less than k=', k, '. Thus precision@k is switched to precision on the whole set.')
            k = len(retrievedIds)
        #print('calcPrecisionAtK for k = ', k)
        retrievedIdx = pd.Index(retrievedIds[0:k])  # [start:len]; lecture slides: indices [1..k] fits python list [0:k]
        if k > 0:
            precisionAtK = len(relevantIdx.intersection(retrievedIdx)) / k
        else:
            precisionAtK = 0
        return precisionAtK

    def calcPrecision(self, retrievedIds: list[str], relevantIds: list[str]):
        return self.calcPrecisionAtK(retrievedIds, relevantIds, len(retrievedIds))

    # returns all Precision@i, for all entries/documents d up to position of d ([first pos (in python 0, in slides 1] .. |retrieved|] - see lecture slides)
    # TODO: somebody else check calculation / formula
    def calcPrecisionsAtI(self, retrievedIds: list[str], relevantIds: list[str]):  # querySongId must be excluded from both lists
        precisionsAtI = list()
        #print('retrievedIdx: ', pd.Index(retrievedIds))
        #print('relevantIdx: ', pd.Index(relevantIds))
        for i in range(0, len(retrievedIds)):  # for documents at [0 .. (|retrievedIds| - 1)] - lecture slides: up to document at i (python: at i-1, as 0-based index)
            precisionAtI = self.calcPrecisionAtK(retrievedIds, relevantIds, (i + 1))
            #print('P@k for k = ', (i + 1), ': ', precisionAtI)

            precisionsAtI.append(tuple([i, precisionAtI])) # tuple([entry.id, precsisionAtI]) or append(precsisionAtI) would also be possible but MAP acccesses it like this
        return precisionsAtI

    # in lecture slides: "relevant(i) = 1 iff the i-th retrieved document is relevant, 0 otherwise" === binary rank
    # returns: relevant(i) for all i
    def calcRelevances(self, retrievedIds: list[str], relevantIds: list[str]):  # querySongId must be excluded from both lists of ids!
        return self.calcBinaryRank(retrievedIds, relevantIds);

    # measure, that is performed on one query
    def calcAveragePrecision(self, querySongId: str, retrievedIds: list[str], preLoadedGenres: pd.DataFrame):
        songsSharingGenreIds = self.getGenreSharingSongsFromId(querySongId, preLoadedGenres)
        precisionsAtI = self.calcPrecisionsAtI(retrievedIds, songsSharingGenreIds)
        relevances = self.calcRelevances(retrievedIds, songsSharingGenreIds)
        averagePrecision = self.calcAveragePrecisionByFormula(retrievedIds, songsSharingGenreIds, relevances, precisionsAtI)
        return averagePrecision

    def calcAveragePrecisionByFormula(self, retrievedIds: list[str], relevantIds: list[str], relevances: list[tuple], precisionsAtI: list[tuple]):
        averagePrecision = 0
        i = 0
        for retrievedIdEntry in enumerate(retrievedIds):
            # P@i and relevant(i) are needed, i.e. precision @ position of i-th item and relevance of i-th item (with certain id). in relevances also on index i
            # id is of the i-th item (iteration over item on position 0 to |ret|-1) -> fits formula of lecture slides
            id = retrievedIdEntry[0]    # ids in retrievedIds list are indices of the elements (0-based), not actual ids (equivalent to i)
            #print('id: ', id, 'i: ', i)
            # indices of relevances list are the same as ids in retrievedIdEntry (use their tuple field 0, rather than their index to be save)
            # TODO: somebody else check calculation / formula (precsisionsAtI[id] vs. precsisionsAtI[i] )
            #print('retrievedIdEntry[0]: ', retrievedIdEntry, ' retrieved relevance of the item with the same id: ', relevances[id], ' relevances[i]: ', relevances[i], ' precision@i[id]: ', precisionsAtI[id], ' precision@i[i]: ', precisionsAtI[i])
            averagePrecision += (relevances[id][1] * precisionsAtI[i][1]) # if relevance is not given, it is 0, thus the total term is 0 and not added to/affecting the sum
            i += 1
        averagePrecision = averagePrecision / len(relevantIds)
        #print('AP: ', averagePrecision)
        return averagePrecision

    # for binary ranks ("yes"/"not" in relevant list according to genre): 1 / (position of first rank 1)
    # https://en.wikipedia.org/wiki/Mean_reciprocal_rank - "If none of the proposed results are correct, reciprocal rank is 0."
    # lecture slides: find the minimum position/index (== highest rank) of a relevant entry/document
    # implementation: run through the (unsorted) list of retrieved elements, and return the first index, if it is relevant (relevance is 1); RR = 1 / found lowest index == 1 / highest rank.
    # returns RR, or 0 for empty list of relevantIds, or if no retrieved element is relevant
    def calcReciprocalRank(self, listOfIdBinaryRankTuples: list[tuple], relevantIds: list[str]):
        nOfPossibleCorrectResults = len(relevantIds)
        if len(relevantIds) == 0:
            #print("no relevant ids - reciprocal rank is 0")
            return 0

        posOfFirstRank1 = -1
        for index, entry in enumerate(listOfIdBinaryRankTuples):
    #        print('check entry: ', entry, ' at index: ', index)
            if entry[1] == 1:
                posOfFirstRank1 = index
                break
        #print(posOfFirstRank1, ' was 0-based entry index: ', listOfIdBinaryRankTuples[posOfFirstRank1], ' +1 for 1-based position count: ', posOfFirstRank1+1)

        if posOfFirstRank1 == -1: # case normally covered with first check of this function, just used for incorrect dummy data
            #print("no relevant ids - reciprocal rank is 0")
            return 0

        posOfFirstRank1 += 1
        #print('RR is: 1 / ', posOfFirstRank1, ' = ', (1 / posOfFirstRank1))
        return 1 / posOfFirstRank1

    # MRR and precision can be falsified, if there is no ranking and retrievedIds are restricted to TOP 100, but relevant are not.
    # possible solution: sort retrieved (unranked) results by id and restrict afterwards to 100 (TOP 100), 50, etc. and do the same for relevant (according to genres).
    # NOTE: equal length is not necessary but cutting results due to TOP x when exceeding x would be a problem
    # for understanding of MRR: https://softwaredoug.com/blog/2021/04/21/compute-mrr-using-pandas.html
    # for combination with binary ranking: https://stats.stackexchange.com/questions/127041/mean-average-precision-vs-mean-reciprocal-rank
    def calcBinaryRank(self, xTopRetrievedIds: list[str], relevantIds: list[str]):  # querySongId must be excluded from both lists of ids!
        # if a song from the retrieved ids list, is also in the list of relevant ids (determines the "relevancy grade 1" == relevance is given) the binary rank is 1, otherwise 0
        # songs only in relevant ids list have no rank, as not part of the result
        binaryRankOfRetrieved = list()
        for entry in xTopRetrievedIds:
            if entry in relevantIds:
    #           print('list element: ', entry)
                binaryRankOfRetrieved.append(tuple([entry, 1]))
            else:
                binaryRankOfRetrieved.append(tuple([entry, 0]))
        return binaryRankOfRetrieved

    # returns Reciprocal Rank (can be 0, if no retrieved entries were relevant), or returns 0 if no relevant entries exist or if no entries were retrieved
    # TODO this is called twice per query song (1x for rrAt10, 1x for rrAt100) -> maybe optimization possible
    def calcRRForQuery(self, querySongId: str, xTopRetrievedIds: list[str], preLoadedGenres: pd.DataFrame):
        #print('\ncalcRRForQuery QUERY ID: ', querySongId, ' RETRIEVED SONG IDS count: ', len(xTopRetrievedIds))
        songsSharingGenreIds = self.getGenreSharingSongsFromId(querySongId, preLoadedGenres)

        retrieved = xTopRetrievedIds  # always a maximum count of 100 for TOP 100, 50 for TOP 50 etc.
        relevant = songsSharingGenreIds  # could be more than 100 for TOP 100, 50 for TOP 50, as currently not ranked/equally sorted, also restricting them to TOP x can falsify the precision
        # print('retrieved: ', retrieved)
        # print('relevant: ', relevant)

        if len(xTopRetrievedIds) == 0 or len(songsSharingGenreIds) == 0:    # no relevant entries exist or nothing retrieved
            #print('RR is 0 as no entries were retrieved or no relevant songs do exist')
            return 0

        relevantIdx = pd.Index(songsSharingGenreIds)
        retrievedIdx = pd.Index(xTopRetrievedIds)

        listOfIdBinaryRankTuples = self.calcBinaryRank(xTopRetrievedIds, songsSharingGenreIds)
        return self.calcReciprocalRank(listOfIdBinaryRankTuples, songsSharingGenreIds)

    # NOTE: RRs being 0, as no retrieved entry was relevant or no relevant entry was retrieved, are not manually excluded, they negatively effect MRR
    def calcMRR(self, reciprocalRanksByQuery: list[float]): # one reciprocal rank per query
        return statistics.mean(reciprocalRanksByQuery)

    def calcMAP(self, averagePrecisionsByQuery: list[float]): # one average precision per query
        return statistics.mean(averagePrecisionsByQuery)

    def calcNDCG(self, querySongId: str, retrievedSongs: pd.DataFrame, preLoadedGenres: pd.DataFrame, ids: list[str]):


        relevantSongsSharingGenreIds = self.getGenreSharingSongsFromId(querySongId, preLoadedGenres)

        retrieved = list(map(lambda x: x if isinstance(x, float) else x[0][0], retrievedSongs["similarity"].values))
        relevant = list(map(lambda id: int(id in relevantSongsSharingGenreIds), retrievedSongs["id"].values))
        # all songs sharing at least one common genre are seen with the same level of relevance (1)

        return sk.metrics.ndcg_score([np.asarray(relevant)], [np.asarray(retrieved)])