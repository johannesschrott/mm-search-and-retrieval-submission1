import numpy as np

from similarity.SimilarityFunction import SimilarityFunction
import pandas as pd
import sklearn as sk


class Jaccard(SimilarityFunction):
    def computeSimilarity(self, queryRepresentation: pd.DataFrame, documentRepresentation: pd.DataFrame):
        """Computes the similarity between the `queryRepresentation` and the `documentRepresenation` using Jaccard"""
        queryRepresentation = self.convertWeightsToOccurence(queryRepresentation)
        documentRepresentation = self.convertWeightsToOccurence(documentRepresentation)

        intersection = len(list(set(queryRepresentation).intersection(documentRepresentation)))
        union = (len(queryRepresentation) + len(documentRepresentation)) - intersection
        return 0 if union == 0 else float(intersection)/union

    def convertWeightsToOccurence(self, dataFrameWithWeights: pd.DataFrame):
        dataFrameWithWeights = dataFrameWithWeights.transpose()
        dataFrameWithWeights["occurring"] = dataFrameWithWeights.iloc[:, 0].map(lambda weight: weight > 0)
        dataFrameWithWeights = dataFrameWithWeights.loc[dataFrameWithWeights["occurring"] == True].index.values


        return dataFrameWithWeights
