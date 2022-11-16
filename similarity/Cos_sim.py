from sklearn.metrics.pairwise import cosine_similarity
from similarity.SimilarityFunction import SimilarityFunction
import pandas as pd
import numpy as np

class Cos_sim(SimilarityFunction):
    def computeSimilarity(self, queryRepresentation: pd.DataFrame, documentRepresentation: pd.DataFrame) -> np.ndarray.dtype:
        return cosine_similarity(queryRepresentation,documentRepresentation)
