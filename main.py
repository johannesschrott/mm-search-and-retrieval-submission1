import argparse
import datetime
import operator
import pathlib
import statistics

import constants
from evaluation.Genres import Genres
from evaluation.Evaluation import Evaluation
from similarity import *
from tqdm import tqdm
import logging
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Manager
import time
import pandas as pd
import numpy as np
import evaluation.LateFusion as lateFusion

def getIdForString(string: str):
    """Helping function that returns the id of a specific song that is specified by a string."""
    ids = pd.read_csv(constants.IDS_PATH, sep="\t", index_col=3, header=0)
    ids["key"] = ids["artist"]+";"+ids["song"]
    print(ids['key'])
    ids.set_index("key", inplace=True)
    return ids.loc[string]['id']

def getIds():
    return pd.read_csv(constants.IDS_PATH, sep="\t", index_col=3, header=0)["id"].values

# if splitGenres and allIds are initialized in main, they must be given as function parameters, otherwise not initialized correctly.
# if global variables, the re-initialization might be done per child process -> better solution to provide them as function parameters.
# using global variables similarityFunction and features works as expected - primitive as values / strings.
def eval_songs(id: id, splitGenres, allIds, precisionSumAt10Values, precisionSumAt100Values, ndcgSumAt10Values, ndcgSumAt100Values, rrAt10List, rrAt100List, percentDeltaMean10List, percentDeltaMean100List, spotifyList, features, similarityFunction):
    similarSongs = doQueryWithId(id, 100, features, similarityFunction)

    evaluation = Evaluation()
    ndcgAt10 = evaluation.calcNDCG(id, similarSongs.head(10), splitGenres, allIds)
    ndcgAt100 = evaluation.calcNDCG(id, similarSongs.head(100), splitGenres, allIds)
    ndcgSumAt10Values.append(ndcgAt10) # in order to avoid the need for locks, add the list values to a sum in main program
    ndcgSumAt100Values.append(ndcgAt100)

    if similarSongs.empty:
        logging.info('similar songs empty')
        precisionAt10 = 0
        precisionAt100 = 0

    else:
        precisionAt10 = evaluation.calcPrecisionForQuery(id, similarSongs["id"].head(10), splitGenres)
        precisionAt100 = evaluation.calcPrecisionForQuery(id, similarSongs["id"].head(100), splitGenres)

    precisionSumAt10Values.append(precisionAt10)  # in order to avoid the need for locks, add the list values to a sum in main program
    precisionSumAt100Values.append(precisionAt100)

    rrAt10List.append(evaluation.calcRRForQuery(id, similarSongs["id"].head(10), splitGenres))
    rrAt100List.append(evaluation.calcRRForQuery(id, similarSongs["id"].head(100), splitGenres))

    percentDeltaMean10List.append(evaluation.percentDeltaMean(id, similarSongs["id"].head(10), spotifyList))
    percentDeltaMean100List.append(evaluation.percentDeltaMean(id, similarSongs["id"].head(100), spotifyList))




parser = argparse.ArgumentParser(prog=constants.PROGRAM_NAME, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('features', choices=['tf-idf', 'word2vec', 'bert', 'blf_spectral', 'blf_correlation', 'blf_deltaspectral', 'blf_logfluc', 'blf_spectralcontrast', 'blf_vardeltaspectral', 'essentia', 'incp', 'mfcc_bow', 'mfcc_stats', 'resnet', 'vgg19', 'early', 'late'], type=str, help="Please choose one set of features")
parser.add_argument('similarityFunction', choices=['cosine', 'inner-product', 'jaccard', 'baseline'], type=str, help="Please choose one similarity function")
parser.add_argument('artistSong', type=str, help="Please enter \"Artist;SongTitle\" for a single song query or \"all\" to compute similarities for all songs.")
# parser.add_argument('lateFusion', choices=['lateFusion=borda', 'lateFusion=none'], help="Please choose late fusion method (ignored for batch and all).")

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
    case "early":
        features = FeaturesType.EARLY
    case "late":
        features = FeaturesType.LATE

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
    splitGenres = evaluation.splitGenresForAllSongs(genres.genres)
    spotifyList = pd.read_csv(constants.SPOTIFY_METADATA_PATH, sep="\t", index_col=0, header=0)
    # print('test sharing song genres for song with "ambient pop" only')
    # sharingSongIdsTest = evaluation.getGenreSharingSongsFromId('31uY67hbgx3mbTo6', splitGenres)

    # 'fallen angel' and 'gabba' occurs only once as a genre, rare: 'vocal trance'
    # print('test sharing song genres for song with "gabba" only') # for testing: add line barbTest12345678	['gabba', 'fallen angel'] to id_genres_mmsr.tsv
    # sharingSongIdsTest = evaluation.getGenreSharingSongsFromId('barbTest12345678', splitGenres)
    # print('sharingSongIdsTest: ', sharingSongIdsTest) # expect: yecA7HU1xXlV7fkZ (for 'gabba') and yYjf30gaGSeOfhOP (for 'fallen angel')

    precisionSumAt10 = 0
    precisionSumAt100 = 0

    ndcgSumAt10 = 0
    ndcgSumAt100 = 0

    rrAt10 = []
    rrAt100 = []

    if args.artistSong != "all" and args.artistSong != "batch":
        print("main called for single query id")

        id = getIdForString(args.artistSong)

        topXSimilarities = 100  # used for the number of returned similarities (per similarity function / per voter) and
        # for the final crop to top elements. e.g. Top100. should be at least 100 to get a correct ndcg@100,
        # precision@100 etc.

        similarSongs = []

        if args.features != "late":
            similarSongs = doQueryWithId(id, topXSimilarities, features, similarityFunction)
            print('similarSongs: ', similarSongs)

            ndcgAt10 = evaluation.calcNDCG(id, similarSongs.head(10), splitGenres, allIds)
            ndcgAt100 = evaluation.calcNDCG(id, similarSongs.head(100), splitGenres, allIds)

            if similarSongs.empty:
                precisionAt10 = 0
                precisionAt100 = 0

            else:
                precisionAt10 = evaluation.calcPrecisionForQuery(id, similarSongs["id"].head(10), splitGenres)
                precisionAt100 = evaluation.calcPrecisionForQuery(id, similarSongs["id"].head(100), splitGenres)

            print("ID: " + str(id))
            print("\tPrecision@10: " + str(precisionAt10))
            print("\tPrecision@100: " + str(precisionAt100))
            print("\tNDCG@10: " + str(ndcgAt10))
            print("\tNDCG@100: " + str(ndcgAt100))
            print("\t%∆Mean@10: " + str(evaluation.percentDeltaMean(id, similarSongs["id"].head(10), spotifyList)))
            print("\t%∆Mean@100: " + str(evaluation.percentDeltaMean(id, similarSongs["id"].head(100), spotifyList)))
        else:
            # 1 result set of similiarSongs per voter
            similarSongsAllVoters = doQueryWithId(id, topXSimilarities, features, similarityFunction)

            voterNames = list(map(lambda x: x[0], LATE_FUSION_LABEL_FEATURESTYPE_FUNCS_AND_PATHS))
            resultLateFusion = lateFusion.modifiedBordaCount(genres.genres, topXSimilarities, similarSongsAllVoters, voterNames)

            similarSongs = resultLateFusion[resultLateFusion['meanScore'].notna()]  # all rows where meanScore != NaN

            print('Top ', topXSimilarities, ': \n (sorted by mean score)')

            printResults = resultLateFusion.copy(deep=True)
            printResults.drop(columns=['id'], inplace=True)  # id column is still there as index thus not needed as column when printing
            print(printResults.head(topXSimilarities).to_string())  # to_string() avoids truncating ouf output

            print('number x of ranked songs in fused Top x: ', len(similarSongs))

            # for "ndcg" on meanScore use dataframe including id as column but also NaNs
            ndcgAt10 = evaluation.calcNDCGAfterLateFusion(id, resultLateFusion.head(10), splitGenres, allIds)
            ndcgAt100 = evaluation.calcNDCGAfterLateFusion(id, resultLateFusion.head(100), splitGenres, allIds)

            # print("\tNDCG@10: " + str(ndcgAt10))
            # print("\tNDCG@100: " + str(ndcgAt100))

    elif args.artistSong == "batch":

        print("main called for batch computation with 1000 batches for all feature sets\n")

        featureList = [FeaturesType.LYRICS_TF_IDF,
                       FeaturesType.LYRICS_WORD2VEC,
                       FeaturesType.LYRICS_BERT,
                       FeaturesType.BLF_CORRELATION,
                       FeaturesType.BLF_DELTASPECTRAL,
                       FeaturesType.BLF_LOGFLUC,
                       FeaturesType.BLF_SPECTRAL,
                       FeaturesType.BLF_SPECTRALCONTRAST,
                       FeaturesType.BLF_VARDELTASPECTRAL,
                       FeaturesType.ESSENTIA,
                       FeaturesType.MFCC_BOW,
                       FeaturesType.MFCC_STATS,
                       FeaturesType.INCP,
                       FeaturesType.RESNET,
                       FeaturesType.VGG19,
                       FeaturesType.EARLY,
                       # FeaturesType.LATE  # no similarity values available, but scores with different range thus metrics must be calculated differently or skipped (metrics where used to select what is used in late fusion)
                       ]

        similarityList = [  # SimilarityFunctionType.BASELINE,
            SimilarityFunctionType.COSINE_SIMILARITY,
            SimilarityFunctionType.INNER_PRODUCT,
            SimilarityFunctionType.JACCARD
        ]

        if constants.NEWSIMILARITIES:
            # this part is for the first time running the program to calculate similarities for all datasets
            # and similarity functions


            for simfunc in similarityList:
                print("Similarity Function: ", simfunc)
                for feat in featureList:
                    print("    Feature: ", feat)
                    doQueryWithId(allIds[1], 100, feat, simfunc)

        # end of the database check run similarity computation

        cols = list()
        cols = ['batch_number']
        for bn in range(0, constants.BATCH_SIZE):
            cols.append(f"{bn}")
        zero_data = np.zeros(shape=(constants.BATCH_COUNT, constants.BATCH_SIZE+1), dtype=int)
        batches = pd.DataFrame(zero_data, columns=cols)
        batches.set_index("batch_number", inplace=True)
        ###########################
        # generation of 1000 random subsamples
        idsets = list()
        for y in range(0, constants.BATCH_COUNT):

            randIdNr = np.random.randint(len(allIds), size=constants.BATCH_SIZE)
            ids = list()
            colnr = 0
            for x in randIdNr:
                ids.append(allIds[x])
                batches.index.values[y] = y
                batches.iloc[[y],[colnr]] = allIds[x]
                if colnr < constants.BATCH_SIZE:
                    colnr += 1
            idx = np.array(ids)
            idsets.append(idx)
        ############################
        nrOfProcessed = len(ids)
        batchDate = datetime.date.today()
        batchTime = time.time()
        randBatchName = f"data/batch_runs/run_{batchDate}_{batchTime}_batches.csv"
        evalBatchName = f"data/batch_runs/run_{batchDate}_{batchTime}_eval.csv"
        batches.to_csv(randBatchName, sep=";")
        cols=["ID Set", "features", "similarity_function", "precision_10", "precision_100",
              "ndcg_10", "ndcg_100", "mrr_10", "mrr_100","delta_mean_10", "delta_mean_100"]
        eval_batch = pd.DataFrame(columns=cols)

        # check if results are already computed - if not, this triggers their computation
        _ = doQueryWithId(ids[0], constants.NR_OF_SONGS, features, similarityFunction)

        # https://superfastpython.com/multiprocessing-pool-share-with-workers/ - "Example of Sharing the Multiprocessing Pool With Workers"
        # https://superfastpython.com/multiprocessing-pool-apply_async/#How_to_Use_apply_async
        # we need to use a pool for proper concurrency (1 process per id is too much) and a manager for sharing variables
        for idrun in range(0, len(idsets)):
            for simfunc in similarityList:
                for features in featureList:
                    print("\n")
                    print("Computing precisions".center(80, "="))
                    print("INFO:")
                    print("Random ID-SET:      #", idrun + 1, "/", len(idsets))
                    print("Dataset:             ", features)
                    print("Similarity function: ", simfunc)
                    print("Computation progress".center(80, "-"))
                    with Manager() as manager:
                        rrAt10List = manager.list()
                        rrAt100List = manager.list()
                        precisionSumAt10Values = manager.list()
                        precisionSumAt100Values = manager.list()
                        ndcgSumAt10Values = manager.list()
                        ndcgSumAt100Values = manager.list()
                        percentDeltaMeanAt10 = manager.list()
                        percentDeltaMeanAt100 = manager.list()
                        print("computing...")
                        with manager.Pool() as pool:
                            # https: // docs.python.org / 3 / library / multiprocessing.html - Using a pool of workers
                            multipleResults = [
                                pool.apply_async(eval_songs, args=(queryId, splitGenres, allIds, precisionSumAt10Values,
                                                                   precisionSumAt100Values, ndcgSumAt10Values,
                                                                   ndcgSumAt100Values, rrAt10List, rrAt100List,
                                                                   percentDeltaMeanAt10, percentDeltaMeanAt100,
                                                                   spotifyList, features, simfunc))
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
                        print("\tMRR@10: " + str(evaluation.calcMRR(rrAt10List)))  # ATTENTION: the managed shared variable is not available outside of the scope (only between "with")
                        print("\tMRR@100: " + str(evaluation.calcMRR(rrAt100List)))
                        print("\t%∆Mean@10: " + str(statistics.median(percentDeltaMeanAt10)))
                        print("\t%∆Mean@100: " + str(statistics.median(percentDeltaMeanAt100)))
                        evaluations_list = [idrun, str(features).lstrip("FeaturesType."),
                                            str(simfunc).lstrip("SimilarityFunctionType."),
                                       precisionSumAt10 / nrOfProcessed, precisionSumAt100 / nrOfProcessed,
                                       ndcgSumAt10 / nrOfProcessed, ndcgSumAt100 / nrOfProcessed,
                                       evaluation.calcMRR(rrAt10List), evaluation.calcMRR(rrAt100List),
                                       statistics.median(percentDeltaMeanAt10), statistics.median(percentDeltaMeanAt100)
                                       ]
                        eval_batch.loc[len(eval_batch)] = evaluations_list
                    #eval_batch.append()

        eval_batch.to_csv(evalBatchName, sep=";")
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

        featureList = [FeaturesType.LYRICS_TF_IDF,
                       FeaturesType.LYRICS_WORD2VEC,
                       FeaturesType.LYRICS_BERT,
                       FeaturesType.BLF_CORRELATION,
                       FeaturesType.BLF_DELTASPECTRAL,
                       FeaturesType.BLF_LOGFLUC,
                       FeaturesType.BLF_SPECTRAL,
                       FeaturesType.BLF_SPECTRALCONTRAST,
                       FeaturesType.BLF_VARDELTASPECTRAL,
                       FeaturesType.ESSENTIA,
                       FeaturesType.MFCC_BOW,
                       FeaturesType.MFCC_STATS,
                       FeaturesType.INCP,
                       FeaturesType.RESNET,
                       FeaturesType.VGG19
                       ]

        if constants.NEWSIMILARITIES:
            # this part is for the first time running the program to calculate similarities for all datasets
            # and similarity functions
            simList = [SimilarityFunctionType.COSINE_SIMILARITY, SimilarityFunctionType.INNER_PRODUCT, SimilarityFunctionType.JACCARD]

            for simfunc in simList:
                print("Similarity Function: ", simfunc)
                for feat in featureList:
                    print("    Feature: ", feat)
                    doQueryWithId(allIds[1], constants.NR_OF_SONGS, feat, simfunc)

        # end of the database check run similarity computation

        ids = allIds[0:1000]  # TODO: run on all ids or provide additional parameter
        nrOfProcessed = len(ids)


        # check if results are already computed - if not, this triggers their computation
        _ = doQueryWithId(ids[0], constants.NR_OF_SONGS, features, similarityFunction)


        # https://superfastpython.com/multiprocessing-pool-share-with-workers/ - "Example of Sharing the Multiprocessing Pool With Workers"
        # https://superfastpython.com/multiprocessing-pool-apply_async/#How_to_Use_apply_async
        # we need to use a pool for proper concurrency (1 process per id is too much) and a manager for sharing variables
        print("\n")
        print("Computing precisions".center(80, "="))
        print("INFO:")
        print("Dataset:             ", features)
        print("Similarity function: ", args.similarityFunction.upper())
        print("Computation progress".center(80, "-"))
        with Manager() as manager:
            rrAt10List = manager.list()
            rrAt100List = manager.list()
            precisionSumAt10Values = manager.list()
            precisionSumAt100Values = manager.list()
            ndcgSumAt10Values = manager.list()
            ndcgSumAt100Values = manager.list()
            percentDeltaMeanAt10 = manager.list()
            percentDeltaMeanAt100 = manager.list()

            with manager.Pool() as pool:
                # https: // docs.python.org / 3 / library / multiprocessing.html - Using a pool of workers
                print("computing...")
                multipleResults = [pool.apply_async(eval_songs, args=(queryId, splitGenres, allIds, precisionSumAt10Values,
                                                                        precisionSumAt100Values, ndcgSumAt10Values,
                                                                        ndcgSumAt100Values, rrAt10List, rrAt100List,
                                                                      percentDeltaMeanAt10, percentDeltaMeanAt100,
                                                                      spotifyList, features)) for queryId in tqdm(ids)]
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
            print("\t%∆Mean@10: " + str(statistics.median(percentDeltaMeanAt10)))
            print("\t%∆Mean@100: " + str(statistics.median(percentDeltaMeanAt100)))


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



