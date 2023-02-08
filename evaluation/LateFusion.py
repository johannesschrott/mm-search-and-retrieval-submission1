import pandas as pd
import statistics
import sklearn as sk
import numpy as np
from numpy import NaN

# creates scores by voter (score based on modified borda count)
# data frame rows are merged by column 'id' of the input data frames (will be present as index and column in the
# returned data set, but only need to be there as column in the input)
# creates a column 'voterCount' showing how many of the voters ranked/scored the row
# creates a column 'sumScores' summing up the NaN scores of all voters, for later calculating the meanScore.
# creates a column 'meanScore' for the mean score over all voters, ignoring NaN cells,
# i.e. if 2 out of 5 voters did not vote: mean of 2
# - if all votes are NaN (no voter ranked the song), instead of a real mean: column is filled with NaN,
# - if only 1 voter ranked a song, the mean is the score of exactly that one voter (not an actual mean)
# sorts the resulting dataframe by mean score (descending, i.e. highest score on position 0, NaN in the end) and voterCount
# (i.e. a high mean by 3 voters is better than a high mean by 2 or only 1 voter)
# writes 2 csv files, one with all rows not sorted, one with all rows sorted by mean score
#
# parameters:
# - allIdsDf: all song ids that appear in any of the voters
#   (no voter shall contain an id missing there but not all voters need to contain all ids)
# - maxScore: 100 if only the Top100 shall be scored, every rank worse than 100 (index 0-99) shall be seen as not
#   ranked an get a score of 0 (represented by NaN)
#
# possible use cases:
# - set maxScore to the maximum len of all voters' dataframes from outside and let the caller decide
#   for the number x of "Top<x>"
# - if x / maxScore is too high, not all voters might have a dataframe with enough entries,
#   then the highest available scores are assigned and lowest scores for those voters are missing
# - see description of createScoreColumn()
def modifiedBordaCount(allIdsDf: pd.DataFrame, maxScore: int, similaritiesByVoters: [pd.DataFrame], scoreColumnNames: [str]):
    scores = pd.DataFrame()
    print(allIdsDf.index)
    scores['id'] = allIdsDf.index
    scores.set_index('id', drop=False,
                     inplace=True)  # drop=False keeps the new index as column - needed for join on column
    nameIndex = 0
    for similaritesDF in similaritiesByVoters:
        resultScores = createScoreColumn(scoreColumnNames[nameIndex], similaritesDF, maxScore)
        resultScores.drop('id', axis=1,
                           inplace=True)  # still existent as index and after the join as one column 'id' from the scores dataframe

        scores = scores.join(resultScores, how='outer')

        nameIndex = nameIndex + 1

    # to_string() does not truncate results in any direction (column, rows) -> better for debugging
    # print('\nfirst 500 after join and before mean: \n', scores.head(500).to_string())

    # row wise average of the score columns, ignoring NaN cells, mean would fail if only NaNs per row as division by 0 -> where clause needed
    # scores['meanScore'] = np.nanmean(scores[scoreColumnNames], axis=1)

    # no voter -> Nan/Nan == same as div by zero thus Nan/1 = Nan
    scores['voterCount'] = scores[scoreColumnNames].count(axis=1)  # implementation does not count NaN values
    scores['sumScores'] = scores[scoreColumnNames].sum(axis=1)  # may be NaN if all scores in a row are NaN
    scores['meanScore'] = scores.apply(lambda row: row['sumScores'] / max(row['voterCount'], 1) if row['voterCount'] >= 1 else NaN, axis=1)

    scores.to_csv('data/temp_lateFusion_notSorted.csv', sep=';', header=True)

    scores = scores.sort_values(by=['meanScore', 'voterCount'], ascending=False)
    # print('after join: scores (sorted by mean): \n', scores.to_string())
    # print('after join: scores (sorted by mean): \n', scores)

    scores.to_csv('data/temp_lateFusion_sortedByMean.csv', sep=';', header=True)
    return scores

# modified borda score: unranked get score 0 (here: NaN), lowest score for a ranked entry = 1
# if top 100 shall be scored but more than 100 are in the data frame:
# - highest ranked (rank 1, index 0) gets best score = 100
# - entries at index 100 and higher (Top 101, Top 102, ...) get score 0, represented as NaN to identify 'non-ranked'
#
# special case: if top 100 shall be scored but less than 100 (e.g. 54) are in the data frame:
# - highest ranked (rank 1, index 0) gets best score = 100
# - entry at rank 54 (index 53) gets score 53, no lower scores existent, no 0-scores / NaN
def createScoreColumn(voterName: str, similaritiesFromVoter: pd.DataFrame, highestRank: int):
    scores = pd.DataFrame()
    scores = similaritiesFromVoter.copy(deep=True)

    scores[voterName] = "0"

    # print('scores: \n', scores)
    # print('indices: ', scores.index)
    scores.set_index('id', drop=False, inplace=True)  # drop=False keeps the new index as column
    # print('indices: ', scores.index)

    index = 0
    # top 1 (rank 1, index 0), gets highest score which is maxRank - index, i.e. 100. rank 100 (index 99) gets score 1, not ranked shall get rank 0
    for rowIndex, row in scores.iterrows():
        id = row['id']
        # print('id is: ', id)
        #print('row: \n', row)
        #  print('in scores: ', scores[id])
        # scores._set_value(id, 'voter1', (maxRank - index))
        score = (highestRank - index) if (highestRank - index) > 0 else 0
        # if highestRank < len(similaritiesFromVoter) some rows will be seen as unranked, i.e. assigned a score of 0, but never a score below 0!
        scores.at[id, voterName] = score
        #print('new score: ', scores.at[id, voterName])
        index = index + 1

    # print('scores afterwards: \n', scores)
    scores.drop(columns=['similarity'], inplace=True)     # otherwise they would need to be suffixed (as overlapping) at join
    return scores
