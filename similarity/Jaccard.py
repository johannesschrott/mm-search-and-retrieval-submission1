import numpy as np

import deprecation
import similarity
import database
import logging
from similarity import *
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances_chunked
import pandas as pd

class Jaccard:
    '''Computes the cosine similarity using matrix multiplication (matrices are divided into smaller matrices)
        Results are directly stored into the database.'''
    def do_computation_using_matrix(self, nrResults: int, featuresType, featuresPath: str):

        data: np.ndarray = pd.read_csv(featuresPath, sep="\t", index_col=0).astype(dtype="float16").values

        j = 0
        for result in tqdm(pairwise_distances_chunked(data, data, metric="jaccard", n_jobs=-1)):
            #logging.info(f"Jaccard for {j} of {data.shape[0]} computed.")
            result = 1-result

            k = 0
            for result_row in result:
                values_for_db = []
                song1id = j + k

                nans = np.isnan(result_row).sum()

                try:
                    for similarity_index in np.argpartition(result_row, -(nrResults + 1 + nans))[-(
                            nrResults + 1 + nans):]:  # gets the nrResults indices of the largest elements
                        song2id = int(similarity_index)
                        if song1id != song2id:
                            values_for_db.append((song1id, song2id, float(result_row[similarity_index]),
                                                  similarity.SimilarityFunctionType.JACCARD.value,
                                                  featuresType.value))
                        # float and int conversions are needed so that SQLite stores in the correct data type
                except ValueError:
                    pass  # if there is a NaN (occuring when vector length 0)

                k = k + 1
                database.insert_similarities(values_for_db)

            j = j + result.shape[0]

    def computeHighestSimilaritiesWithMatrix(self, queryIdStr: str, nrResults: int, featuresType, featuresPath) -> pd.DataFrame:
            queryIdInt = database.get_row_nr_from_id(queryIdStr)

            sorted_res = database.get_n_best_similarity_values(queryIdInt, nrResults, similarity.SimilarityFunctionType.JACCARD.value, featuresType.value)

            if len(sorted_res) == 0:
                self.do_computation_using_matrix(nrResults, featuresType, featuresPath)
                sorted_res = database.get_n_best_similarity_values(queryIdInt, nrResults,
                                                            similarity.SimilarityFunctionType.JACCARD.value,
                                                            featuresType.value)

            result = pd.DataFrame(columns=['id', 'similarity'])

            for entry in sorted_res:
                result = pd.concat([result, pd.Series({'id': database.get_id_from_row_nr(entry[1]), 'similarity': entry[2]}).to_frame().T], axis=0, ignore_index=True)

            return result

