
from similarity.SimilarityFunction import SimilarityFunction
import pandas as pd
import numpy as np
class inner_product(SimilarityFunction):
    def computeSimilarity(self, queryRepresentation: pd.DataFrame, documentRepresentation: pd.DataFrame) -> np.ndarray.dtype:
        queryRepresentation = queryRepresentation.to_numpy()
        documentRepresentation = np.transpose(documentRepresentation.to_numpy())
        return np.dot(queryRepresentation,documentRepresentation)
