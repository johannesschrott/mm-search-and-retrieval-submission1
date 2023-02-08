import logging

import numpy as np
import torch
from torchmetrics.functional import pairwise_cosine_similarity

import database
import similarity
import pandas as pd
from tqdm import tqdm
from similarity import *



class Cos_sim:

    '''Computes the cosine similarity using matrix multiplication (matrices are divided into smaller matrices)
    Results are directly stored into the database.'''
    def do_computation_using_matrix(self, nrResults: int, featuresType, featuresPath):

        data = pd.read_csv(featuresPath, sep="\t", index_col=0).astype(dtype="float32")
        print("========================")
        print("Calculating similarities")
        if torch.cuda.is_available():
            #logging.info("Using CUDA")
            print("-------using CUDA-------")
            device = torch.device("cuda:0")
        elif torch.has_mps:
            print("-------using MPS--------")
            #logging.info("Using MPS")
            device = torch.device("mps")
        else:
            print("-------using CPU--------")
            device = torch.device("cpu")

        torch_data_all = torch.tensor(data.values)
        torch_data_all = torch_data_all.to(device)

        data_parts = []

        computation_step_size = 1000
        if featuresType == similarity.FeaturesType.VGG19:
            computation_step_size = 1000 # step size 1000 is too large for 4GB RAM GPU

        for i in range(0, data.shape[0], computation_step_size):
            data_parts.append(pd.DataFrame(data.iloc[i: data.shape[0] if data.shape[0] < i+computation_step_size else i+computation_step_size, :]))

        # As computation takes a lot of RAM force immediate deletion of variables not needed any longer
        del data

        j = 0
        for part in tqdm(data_parts):
            torch_data_part = torch.tensor(part.values)
            torch_data_part = torch_data_part.to(device)
            torch_similarities = pairwise_cosine_similarity(torch_data_part,torch_data_all)
            torch_similarities = torch_similarities.to("cpu")

            result = torch_similarities.numpy()

            k = 0
            values_for_db = []

            for result_row in result:
                song1id = j+k

                nans = np.isnan(result_row).sum()

                try:
                    if nrResults == constants.NR_OF_SONGS:
                        for similarity_index in range(0, nrResults):  # gets the nrResults indices of the largest elements
                            song2id = int(similarity_index)
                            if song1id != song2id:
                                values_for_db.append((song1id, song2id, float(result_row[similarity_index]),
                                                      similarity.SimilarityFunctionType.COSINE_SIMILARITY.value,
                                                      featuresType.value))
                            # float and int conversions are needed so that SQLite stores in the correct data type
                    else:
                        for similarity_index in np.argpartition(result_row, -(nrResults+1+nans))[-(nrResults+1+nans):]: # gets the nrResults indices of the largest elements
                            song2id = int(similarity_index)
                            if song1id != song2id:
                              values_for_db.append((song1id, song2id, float(result_row[similarity_index]), similarity.SimilarityFunctionType.COSINE_SIMILARITY.value, featuresType.value))
                            # float and int conversions are needed so that SQLite stores in the correct data type

                except ValueError as e:
                    print('Error: ', e)
                    #print(result_row[similarity_index])
                    #print(similarity.SimilarityFunctionType.COSINE_SIMILARITY.value)
                    #print(featuresType.value)
                    #print(e)
                    pass # if there is a NaN (occuring when vector length 0)

                k = k + 1

            database.insert_similarities(values_for_db)
            j = j + computation_step_size
        print("========================")


    def computeHighestSimilaritiesWithMatrix(self, queryIdStr: str, nrResults: int, featuresType, featuresPath) -> pd.DataFrame:
        queryIdInt = database.get_row_nr_from_id(queryIdStr)

        sorted_res = database.get_n_best_similarity_values(queryIdInt, nrResults, similarity.SimilarityFunctionType.COSINE_SIMILARITY.value, featuresType.value)

        if len(sorted_res) == 0:
            self.do_computation_using_matrix(nrResults, featuresType, featuresPath)
            sorted_res = database.get_n_best_similarity_values(queryIdInt, nrResults,
                                                        similarity.SimilarityFunctionType.COSINE_SIMILARITY.value,
                                                        featuresType.value)

        result = pd.DataFrame(columns=['id', 'similarity'])

        for entry in sorted_res:
            result = pd.concat([result, pd.Series({'id': database.get_id_from_row_nr(entry[1]), 'similarity': entry[2]}).to_frame().T], axis=0, ignore_index=True)

        return result