import argparse

import constants
from evaluation.Genres import Genres
from evaluation.Evaluation import Evaluation
from similarity import *
from tqdm import tqdm
import logging
import multiprocessing
import time
import pandas as pd
import numpy as np


def getIdForString(string: str):
    """Helping function that returns the id of a specific song that is specified by a string."""
    ids = pd.read_csv(constants.IDS_PATH, sep="\t", index_col=3, header=0)
    ids["key"] = ids["artist"]+";"+ids["song"]
    ids.set_index("key", inplace=True)
    return ids.loc[string]['id']

def getIds():
    return pd.read_csv(constants.IDS_PATH, sep="\t", index_col=3, header=0)["id"].values

# if genres and allIds are initialized in main, they must be given as function parameters, otherwise not initialized correctly.
# if global variables, the re-initialization might be done per child process -> better solution to provide them as function parameters.
# using global variables similarityFunction and features works as expected - primitive as values / strings.
def eval_songs(id: id, genres, allIds, precisionSumAt10Values, precisionSumAt100Values, ndcgSumAt10Values, ndcgSumAt100Values, rrAt10List, rrAt100List, features):
    similarSongs = doQueryWithId(id, 100, features, similarityFunction)

    evaluation = Evaluation()
    ndcgAt10 = evaluation.calcNDCG(id, similarSongs.head(10), genres.genres, allIds)
    ndcgAt100 = evaluation.calcNDCG(id, similarSongs.head(100), genres.genres, allIds)
    ndcgSumAt10Values.append(ndcgAt10) # in order to avoid the need for locks, add the list values to a sum in main program
    ndcgSumAt100Values.append(ndcgAt100)

    if similarSongs.empty:
        logging.info('similar songs empty')
        precisionAt10 = 0
        precisionAt100 = 0

    else:
        precisionAt10 = evaluation.calcPrecisionForQuery(id, similarSongs["id"].head(10), genres.genres)
        precisionAt100 = evaluation.calcPrecisionForQuery(id, similarSongs["id"].head(100), genres.genres)

    precisionSumAt10Values.append(precisionAt10)  # in order to avoid the need for locks, add the list values to a sum in main program
    precisionSumAt100Values.append(precisionAt100)

    rrAt10List.append(evaluation.calcRRForQuery(id, similarSongs["id"].head(10), genres.genres))
    rrAt100List.append(evaluation.calcRRForQuery(id, similarSongs["id"].head(100), genres.genres))



parser = argparse.ArgumentParser(prog=constants.PROGRAM_NAME, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('features', choices=['tf-idf', 'word2vec', 'bert', 'blf_correlation', 'blf_deltaspectral', 'blf_logfluc', 'blf_spectralcontrast', 'blf_vardeltaspectral', 'essentia', 'incp', 'mfcc_bow', 'mfcc_stats', 'resnet', 'vgg19'], type=str, help="Please choose one set of features")
parser.add_argument('similarityFunction', choices=['cosine', 'inner-product', 'jaccard', 'baseline'], type=str, help="Please choose one similarity function")
parser.add_argument('artistSong',type=str, help="Please enter \"Artist;SongTitle\" for a single song query or \"all\" to compute similarities for all songs.")

args = parser.parse_args()
logging.getLogger().setLevel(logging.INFO)

features = None
similarityFunction = None

# complex objects as globals would work, but it seems each child process has its own global variables ->
# initialization is done more often. -> use it as function param instead and init in main
# allIds = getIds()
# genres = Genres()

match args.features:
    case "tf-idf":
        features = FeaturesType.LYRICS_TF_IDF
    case "word2vec":
        features = FeaturesType.LYRICS_WORD2VEC
    case "bert":
        features = FeaturesType.LYRICS_BERT
    case "vgg19":
        features = FeaturesType.VGG19
    case "blf_correlation":
        features = FeaturesType.BLF_CORRELATION
    case "blf_deltaspectral":
        features = FeaturesType.BLF_DELTASPECTRAL
    case "blf_spectral":
        features = FeaturesType.BLF_SPECTRAL
    case "blf_logfluc":
        features = FeaturesType.BLF_LOGFLUC
    case "blf_spectralcontrast":
        features = FeaturesType.BLF_SPECTRALCONTRAST
    case "blf_vardeltaspectral":
        features = FeaturesType.BLF_VARDELTASPECTRAL
    case "essentia":
        features = FeaturesType.ESSENTIA
    case "incp":
        features = FeaturesType.INCP
    case "mfcc_bow":
        features = FeaturesType.MFCC_BOW
    case "mfcc_stats":
        features = FeaturesType.MFCC_STATS
    case "resnet":
        features = FeaturesType.RESNET

match args.similarityFunction:
    case "cosine":
        similarityFunction = SimilarityFunctionType.COSINE_SIMILARITY
    case "inner-product":
        similarityFunction = SimilarityFunctionType.INNER_PRODUCT
    case "jaccard":
        similarityFunction = SimilarityFunctionType.JACCARD
    case "baseline":
        similarityFunction = SimilarityFunctionType.BASELINE

# please read: https://superfastpython.com/multiprocessing-pool-vs-process/ - Comparison of Pool vs Process - especially: Differences Between Pool and Process -> no need for Pool; also read: How to Choose Pool or Process and following.
# "Don’t use the multiprocessing.Process class when you need to execute and manage multiple tasks concurrently." -> use Pool
# https://towardsdatascience.com/python-concurrency-multiprocessing-327c02544a5a -> Shared variables across processes, Queues and Pipes -> using Process is sufficient -> no multiprocessing.Manager() needed BUT: processes need to be put into for-loop (no map function)
# https://zetcode.com/python/multiprocessing/  - simple_queue2.py combined with queue_order.py
# https://docs.python.org/3/library/multiprocessing.html: Shared memory, Server process -> how to use list as shared object "[managers] they can be made to support arbitrary object types. Also, a single manager can be shared by processes ..."
# locks on shared values: https://stackoverflow.com/questions/60646431/how-to-use-multiprocessing-manager-value-to-store-a-sum
# analysis of approach without pool but manager and processes showed that there is indeed no concurrency or no correct concurrency - final approach: manager for sharing variables + pool -> link see below
if __name__ == '__main__':
    #startTime = time.process_time() # process_time is actual execution time per process and thus not correct when there are child processes
    startTime = time.time()
    evaluation = Evaluation()
    allIds = getIds()
    genres = Genres()

    precisionSumAt10 = 0
    precisionSumAt100 = 0

    ndcgSumAt10 = 0
    ndcgSumAt100 = 0

    rrAt10 = []
    rrAt100 = []

    if args.artistSong != "all" and args.artistSong != "batch":
        print("main called for single query id")

        id = getIdForString(args.artistSong)
        similarSongs = doQueryWithId(id, 100, features, similarityFunction)

        ndcgAt10 = evaluation.calcNDCG(id, similarSongs.head(10), genres.genres, allIds)
        ndcgAt100 = evaluation.calcNDCG(id, similarSongs.head(100), genres.genres, allIds)

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

    elif args.artistSong == "batch":

        print("main called for batch computation with 1000 batches for all feature sets\n")

        featureList = [FeaturesType.BLF_CORRELATION,
                       FeaturesType.LYRICS_TF_IDF,
                       FeaturesType.LYRICS_WORD2VEC,
                       FeaturesType.LYRICS_BERT,
                       FeaturesType.VGG19,
                       FeaturesType.BLF_DELTASPECTRAL,
                       FeaturesType.BLF_SPECTRALCONTRAST,
                       FeaturesType.BLF_VARDELTASPECTRAL,
                       FeaturesType.BLF_SPECTRAL,
                       FeaturesType.ESSENTIA,
                       FeaturesType.INCP,
                       FeaturesType.MFCC_BOW,
                       FeaturesType.MFCC_STATS,
                       FeaturesType.RESNET,
                       FeaturesType.VGG19,
                       FeaturesType.BLF_LOGFLUC]

        if constants.NEWSIMILARITIES:
            # this part is for the first time running the program to calculate similarities for all datasets
            # and similarity functions
            simList = [SimilarityFunctionType.BASELINE,
                       SimilarityFunctionType.COSINE_SIMILARITY,
                       SimilarityFunctionType.INNER_PRODUCT,
                       SimilarityFunctionType.JACCARD
                       ]

            for simfunc in simList:
                print("Similarity Function: ", simfunc)
                for feat in featureList:
                    print("    Feature: ", feat)
                    doQueryWithId(allIds[1], 100, feat, simfunc)

        # end of the database check run similarity computation

        # ids = allIds[0:10]  # TODO: run on all ids or provide additional parameter

        ###########################
        # generation of 1000 random subsamples
        idsets = list()
        for y in range(0, 5):

            randIdNr = np.random.randint(len(allIds), size=10)
            ids = list()
            for x in randIdNr:
                ids.append(allIds[x])
            idx = np.array(ids)
            idsets.append(idx)
        ############################
        nrOfProcessed = len(ids)

        # check if results are already computed - if not, this triggers their computation
        _ = doQueryWithId(ids[0], 100, features, similarityFunction)

        # https://superfastpython.com/multiprocessing-pool-share-with-workers/ - "Example of Sharing the Multiprocessing Pool With Workers"
        # https://superfastpython.com/multiprocessing-pool-apply_async/#How_to_Use_apply_async
        # we need to use a pool for proper concurrency (1 process per id is too much) and a manager for sharing variables
        for idrun in range(0, len(idsets)):
            for features in featureList:
                print("\n")
                print("Computing precisions".center(80, "="))
                print("INFO:")
                print("Random ID-SET:      #", idrun + 1, "/", len(idsets))
                print("Dataset:             ", features)
                print("Similarity function: ", args.similarityFunction.upper())
                print("Computation progress".center(80, "-"))
                with multiprocessing.Manager() as manager:
                    rrAt10List = manager.list()
                    rrAt100List = manager.list()
                    precisionSumAt10Values = manager.list()
                    precisionSumAt100Values = manager.list()
                    ndcgSumAt10Values = manager.list()
                    ndcgSumAt100Values = manager.list()
                    with manager.Pool() as pool:
                        # https: // docs.python.org / 3 / library / multiprocessing.html - Using a pool of workers
                        print("computing...")
                        multipleResults = [
                            pool.apply_async(eval_songs, args=(queryId, genres, allIds, precisionSumAt10Values,
                                                               precisionSumAt100Values, ndcgSumAt10Values,
                                                               ndcgSumAt100Values, rrAt10List, rrAt100List, features))
                            for queryId in tqdm(idsets[idrun])]
                        # instead of map function, "for" is needed

                        # waiting for all to finish: https://superfastpython.com/multiprocessing-pool-wait-for-all-tasks/
                        print("Catching results of computation...")
                        [result.wait() for result in tqdm(multipleResults)]

                    # print('rrAt10List: ', rrAt10List)
                    # print('precisionSumAt10Values: ', precisionSumAt10Values)

                    precisionSumAt10 = sum(precisionSumAt10Values)
                    precisionSumAt100 = sum(precisionSumAt100Values)

                    # print('precisionSumAt10Values summed up: ', precisionSumAt10)

                    ndcgSumAt10 = sum(ndcgSumAt10Values)
                    ndcgSumAt100 = sum(ndcgSumAt100Values)

                    # print('ndcgSumAt10Values: ', ndcgSumAt10Values)
                    # print('ndcgSumAt10Values summed up: ', ndcgSumAt10)

                    print("RESULT".center(80, "="))
                    print("Total for the " + str(nrOfProcessed) + " processed songs:")
                    print("".center(80, "-"))
                    print("\tPrecision@10: " + str(precisionSumAt10 / nrOfProcessed))
                    print("\tPrecision@100: " + str(precisionSumAt100 / nrOfProcessed))
                    print("\tNDCG@10: " + str(ndcgSumAt10 / nrOfProcessed))
                    print("\tNDCG@100: " + str(ndcgSumAt100 / nrOfProcessed))
                    print("\tMRR@10: " + str(evaluation.calcMRR(rrAt10List)))  # ATTENTION: the managed shared variable is not available outside of the scope (between "with")
                    print("\tMRR@100: " + str(evaluation.calcMRR(rrAt100List)))

        # endTime = time.process_time()
        endTime = time.time()
        executionTime = endTime - startTime
        # logging.info("Execution took: {0} seconds".format(executionTime))
        hours = executionTime / 3600
        min = (hours - int(hours)) * 60
        seconds = (min - int(min)) * 60
        print("".center(80, "-"))
        print("Computation time: ", int(hours), " hours ", int(min), " minutes ", int(seconds), " seconds")
        print("".center(80, "-"))

    else:
        print("main called for all query ids")

        featureList = [FeaturesType.BLF_CORRELATION,
                       FeaturesType.LYRICS_TF_IDF,
                       FeaturesType.LYRICS_WORD2VEC,
                       FeaturesType.LYRICS_BERT,
                       FeaturesType.VGG19,
                       FeaturesType.BLF_DELTASPECTRAL,
                       FeaturesType.BLF_SPECTRALCONTRAST,
                       FeaturesType.BLF_VARDELTASPECTRAL,
                       FeaturesType.ESSENTIA,
                       FeaturesType.INCP,
                       FeaturesType.MFCC_BOW,
                       FeaturesType.MFCC_STATS,
                       FeaturesType.RESNET,
                       FeaturesType.VGG19,
                       FeaturesType.BLF_SPECTRAL,
                       FeaturesType.BLF_LOGFLUC]

        if constants.NEWSIMILARITIES:
            # this part is for the first time running the program to calculate similarities for all datasets
            # and similarity functions
            simList = [SimilarityFunctionType.COSINE_SIMILARITY, SimilarityFunctionType.INNER_PRODUCT, SimilarityFunctionType.JACCARD]

            for simfunc in simList:
                print("Similarity Function: ", simfunc)
                for feat in featureList:
                    print("    Feature: ", feat)
                    doQueryWithId(allIds[1], 100, feat, simfunc)

        # end of the database check run similarity computation

        ids = allIds[0:1000]  # TODO: run on all ids or provide additional parameter
        nrOfProcessed = len(ids)


        # check if results are already computed - if not, this triggers their computation
        _ = doQueryWithId(ids[0], 100, features, similarityFunction)


        # https://superfastpython.com/multiprocessing-pool-share-with-workers/ - "Example of Sharing the Multiprocessing Pool With Workers"
        # https://superfastpython.com/multiprocessing-pool-apply_async/#How_to_Use_apply_async
        # we need to use a pool for proper concurrency (1 process per id is too much) and a manager for sharing variables
        print("\n")
        print("Computing precisions".center(80, "="))
        print("INFO:")
        print("Dataset:             ", features)
        print("Similarity function: ", args.similarityFunction.upper())
        print("Computation progress".center(80, "-"))
        with multiprocessing.Manager() as manager:
            rrAt10List = manager.list()
            rrAt100List = manager.list()
            precisionSumAt10Values = manager.list()
            precisionSumAt100Values = manager.list()
            ndcgSumAt10Values = manager.list()
            ndcgSumAt100Values = manager.list()
            with manager.Pool() as pool:
                # https: // docs.python.org / 3 / library / multiprocessing.html - Using a pool of workers
                print("computing...")
                multipleResults = [pool.apply_async(eval_songs, args=(queryId, genres, allIds, precisionSumAt10Values,
                                                                        precisionSumAt100Values, ndcgSumAt10Values,
                                                                        ndcgSumAt100Values, rrAt10List, rrAt100List, features)) for queryId in tqdm(ids)]
                # instead of map function, "for" is needed


                # waiting for all to finish: https://superfastpython.com/multiprocessing-pool-wait-for-all-tasks/
                print("Catching results of computation...")
                [result.wait() for result in tqdm(multipleResults)]

            #print('rrAt10List: ', rrAt10List)
            #print('precisionSumAt10Values: ', precisionSumAt10Values)

            precisionSumAt10 = sum(precisionSumAt10Values)
            precisionSumAt100 = sum(precisionSumAt100Values)

            #print('precisionSumAt10Values summed up: ', precisionSumAt10)

            ndcgSumAt10 = sum(ndcgSumAt10Values)
            ndcgSumAt100 = sum(ndcgSumAt100Values)

            #print('ndcgSumAt10Values: ', ndcgSumAt10Values)
            #print('ndcgSumAt10Values summed up: ', ndcgSumAt10)

            print("RESULT".center(80, "="))
            print("Total for the " + str(nrOfProcessed) + " processed songs:")
            print("".center(80, "-"))
            print("\tPrecision@10: " + str(precisionSumAt10/nrOfProcessed))
            print("\tPrecision@100: " + str(precisionSumAt100/nrOfProcessed))
            print("\tNDCG@10: " + str(ndcgSumAt10/nrOfProcessed))
            print("\tNDCG@100: " + str(ndcgSumAt100/nrOfProcessed))
            print("\tMRR@10: " + str(evaluation.calcMRR(rrAt10List))) # ATTENTION: the managed shared variable is not available outside of the scope (between "with")
            print("\tMRR@100: " + str(evaluation.calcMRR(rrAt100List)))



    #endTime = time.process_time()
    endTime = time.time()
    executionTime = endTime - startTime
    #logging.info("Execution took: {0} seconds".format(executionTime))
    hours = executionTime/3600
    min = (hours - int(hours) ) * 60
    seconds = (min-int(min)) * 60
    print("".center(80, "-"))
    print("Computation time: ", int(hours), " hours ", int(min), " minutes ", int(seconds), " seconds")
    print("".center(80, "-"))



