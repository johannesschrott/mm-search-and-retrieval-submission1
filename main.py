import argparse

from evaluation.Genres import Genres
from evaluation.Evaluation import Evaluation
from similarity import *

def getIdForString(string: str):
    """Helping function that returns the id of a specific song that is specified by a string."""
    ids = pd.read_csv(constants.IDS_PATH, sep="\t", index_col=3, header=0)
    ids["key"] = ids["artist"]+" "+ids["song"]
    ids.set_index("key", inplace=True)
    return ids.loc[string]['id']

def getIds():
    return pd.read_csv(constants.IDS_PATH, sep="\t", index_col=3, header=0)["id"].values

parser = argparse.ArgumentParser(prog=constants.PROGRAM_NAME, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('features', choices=['tf-idf', 'word2vec', 'bert'], type=str, help="Please choose one set of features")
parser.add_argument('similarityFunction', choices=['cosine', 'inner-product', 'jaccard'], type=str, help="Please choose one similarity function")
parser.add_argument('artistSong',type=str, help="Please enter \"Artist SongTitle\" for a single song query or \"all\" to compute similarities for all songs.")

args = parser.parse_args()


features = None
similarityFunction = None

match args.features:
    case "tf-idf":
        features = FeaturesType.TF_IDF
    case "word2vec":
        features = FeaturesType.WORD2VEC
    case "bert":
        features = FeaturesType.BERT

match args.similarityFunction:
    case "cosine":
        similarityFunction = SimilarityFunctionType.COSINE_SIMILARITY
    case "inner-product":
        similarityFunction = SimilarityFunctionType.INNER_PRODUCT
    case "jaccard":
        similarityFunction = SimilarityFunctionType.JACCARD

if args.artistSong != "all":

    id = getIdForString(args.artistSong)
    similarSongs = doQueryWithId(id, 100, features, similarityFunction)



    evaluation = Evaluation()

    genres = Genres()
    allIds = getIds()

    ndcgAt10 = evaluation.calcNDCG(id, similarSongs["id"].head(10), genres.genres, allIds)
    ndcgAt100 = evaluation.calcNDCG(id, similarSongs["id"].head(100), genres.genres, allIds)

    if similarSongs.empty:
        precisionAt10 = 0
        precisionAt100 = 0

    else:
        precisionAt10 = evaluation.calcPrecisionForQuery(id, similarSongs["id"].head(10), genres.genres)
        precisionAt100 = evaluation.calcPrecisionForQuery(id, similarSongs["id"].head(100), genres.genres)


    print("ID: " + str(id))
    print("\tPrecision@10: " + str(precisionAt10))
    print("\tPrecision@100: " + str(precisionAt100))
    print("\tNDCG@10: " + str(ndcgAt10))
    print("\tNDCG@100: " + str(ndcgAt100))

else:
    allIds = getIds()

    ids = allIds[0:15]
    nrOfPorcessed = len(ids)

    precisionSumAt10 = 0
    precisionSumAt100 = 0

    ndcgSumAt10 = 0
    ndcgSumAt100 = 0

    genres = Genres()

    rrsAt10 = []
    rrsAt100 = []


    for id in ids:
        similarSongs = doQueryWithId(id, 100, features, similarityFunction)



        evaluation = Evaluation()

        ndcgAt10 = evaluation.calcNDCG(id, similarSongs.head(10), genres.genres, allIds)
        ndcgAt100 = evaluation.calcNDCG(id, similarSongs.head(100), genres.genres, allIds)

        if similarSongs.empty:
            precisionAt10 = 0
            precisionAt100 = 0

        else:
            precisionAt10 = evaluation.calcPrecisionForQuery(id, similarSongs["id"].head(10), genres.genres)
            precisionAt100 = evaluation.calcPrecisionForQuery(id, similarSongs["id"].head(100), genres.genres)



        precisionSumAt10 = precisionSumAt10 + precisionAt10
        precisionSumAt100 = precisionSumAt100 + precisionAt100

        ndcgSumAt10 = ndcgSumAt10 + ndcgAt10
        ndcgSumAt100 = ndcgSumAt100 + ndcgAt100

        rrsAt10.append(evaluation.calcRRForQuery(id, similarSongs["id"].head(10), genres.genres))
        rrsAt100.append(evaluation.calcRRForQuery(id, similarSongs["id"].head(10), genres.genres))

        print("ID: "+str(id))
        print("\tPrecision@10: "+str(precisionAt10))
        print("\tPrecision@100: "+str(precisionAt100))
        print("\tNDCG@10: "+str(ndcgAt10))
        print("\tNDCG@100: "+str(ndcgAt100))

    evaluation = Evaluation()

    print("==================")
    print("Total for the "+ str(nrOfPorcessed) +" processed songs:")
    print("==================")
    print("\tPrecision@10: " + str(precisionSumAt10/nrOfPorcessed))
    print("\tPrecision@100: " + str(precisionSumAt100/nrOfPorcessed))
    print("\tNDCG@10: " + str(ndcgSumAt10/nrOfPorcessed))
    print("\tNDCG@100: " + str(ndcgSumAt100/nrOfPorcessed))
    print("\tMRR@10: " + str(evaluation.calcMRR(rrsAt10)))
    print("\tMRR@100: " + str(evaluation.calcMRR(rrsAt100)))


