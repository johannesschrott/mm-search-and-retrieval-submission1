
import pandas as pd
import numpy as np
import database
import logging
from tqdm import tqdm
import similarity


class Inner_product:
    '''Computes the inner product, which is used as similarity value using matrix multiplication (matrices are divided into smaller matrices)
       Results are directly stored into the database.'''
    def do_computation_using_matrix(self, nrResults: int, featuresType, featuresPath):

        logging.info("Starting inner product similarity computation")

        data = pd.read_csv(featuresPath, sep="\t", index_col=0).astype(dtype="float32") # numba jit compilation does not support float 16 :(

        j = 0
        values_for_db = []
        for row in tqdm(data.values):
            if (j % 1000 - 1 == 0):
                values_for_db = []

            result = np.inner(row.transpose(), data)

            song1id = j

            nans = np.isnan(result).sum()

            try:
                highest_indices = np.argpartition(result, -(nrResults + 1 + nans))[-(nrResults + 1 + nans):]  # gets the nrResults indices of the largest elements
                for similarity_index in highest_indices:
                    song2id = int(similarity_index)
                    if song1id != song2id:
                        values_for_db.append((song1id, song2id, float(result[similarity_index]),
                                              similarity.SimilarityFunctionType.INNER_PRODUCT.value,
                                              featuresType.value))
                    # float and int conversions are needed so that SQLite stores in the correct data type
            except ValueError:
                pass  # if there is a NaN (occuring when vector length 0)

            if (j % 1000 == 0 or j == data.shape[0]-1):
                database.insert_similarities(values_for_db)
                #logging.info(f"Add batch of results to database. Already processed and stored rows: {j}")
            j = j + 1

    def computeHighestSimilaritiesWithMatrix(self, queryIdStr: str, nrResults: int, featuresType, featuresPath) -> pd.DataFrame:
        queryIdInt = database.get_row_nr_from_id(queryIdStr)

        sorted_res = database.get_n_best_similarity_values(queryIdInt, nrResults, similarity.SimilarityFunctionType.INNER_PRODUCT.value, featuresType.value)

        if len(sorted_res) == 0:
            self.do_computation_using_matrix(nrResults, featuresType, featuresPath)
            sorted_res = database.get_n_best_similarity_values(queryIdInt, nrResults,
                                                        similarity.SimilarityFunctionType.INNER_PRODUCT.value,
                                                        featuresType.value)

        result = pd.DataFrame(columns=['id', 'similarity'])

        for entry in sorted_res:
            result = pd.concat([result, pd.Series({'id': database.get_id_from_row_nr(entry[1]), 'similarity': entry[2]}).to_frame().T], axis=0, ignore_index=True)

        return result