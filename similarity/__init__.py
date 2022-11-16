from enum import Enum

import pandas as pd
import json
import os

import constants
from similarity.BERT import BERT
from similarity.Jaccard import Jaccard
from similarity.Word2Vec import Word2Vec
from similarity.TF_IDF import TF_IDF
from similarity.Cos_sim import Cos_sim
from similarity.inner_product import inner_product


class FeaturesType(Enum):
    """Enum that provides all possible feature types (for usage in parameters)"""
    TF_IDF = 0
    WORD2VEC = 1
    BERT = 2



class SimilarityFunctionType(Enum):
    """Enum that provides all possible types of the similarity function (for usage in parameters)"""
    COSINE_SIMILARITY = 1
    JACCARD = 2
    INNER_PRODUCT = 3




def doQueryWithId(queryId: str, nrResults: int, featuresType: FeaturesType,
                  similarityFunctionType: SimilarityFunctionType):
    if os.path.exists("queryResults.json") and os.path.getsize("queryResults.json") > 0:
        with open("queryResults.json", 'r', encoding='utf-8') as queryResultsFile:

            data = queryResultsFile.read()
        queryResultsJson = json.loads(data)
    else:
        queryResultsJson = {}

    similarityFunction: SimilarityFunction = None
    documentsFeatures: Features = None
    result = pd.DataFrame({"id": [], "similarity": []})



    # Only do a recomputation in case there is no stored value in the json
    if not (featuresType.name in queryResultsJson):
        queryResultsJson[featuresType.name] = {}
    if not (similarityFunctionType.name in queryResultsJson[featuresType.name]):
        queryResultsJson[featuresType.name][similarityFunctionType.name] = {}
    if not (queryId in queryResultsJson[featuresType.name][similarityFunctionType.name]):
        print("Computation of similarities for "+queryId+" is needed.")

        match featuresType:
            case FeaturesType.WORD2VEC:
                documentsFeatures = Word2Vec()
            case FeaturesType.TF_IDF:
                documentsFeatures = TF_IDF()
            case FeaturesType.BERT:
                documentsFeatures = BERT()

        match similarityFunctionType:
            case SimilarityFunctionType.JACCARD:
                similarityFunction = Jaccard()
            case SimilarityFunctionType.COSINE_SIMILARITY:
                similarityFunction = Cos_sim()
            case SimilarityFunctionType.INNER_PRODUCT:
                similarityFunction = inner_product()

        queryFeatures = documentsFeatures.getFeatureForId(queryId)
        documentsFeatures.resetIterator()

        for documentFeatures in documentsFeatures:
            documentId = documentFeatures[0]
            documentRepresentation = documentFeatures[1].to_frame().transpose()

            if (documentId != queryId):  # do not compute similarity between the same document
                documentSimilarityResult = pd.DataFrame({"id": [documentFeatures[0]],
                                                         "similarity": [similarityFunction.computeSimilarity(
                                                             queryRepresentation=queryFeatures,
                                                             documentRepresentation=documentRepresentation)]})
                result = pd.concat([result, documentSimilarityResult])

        result.sort_values(by="similarity", inplace=True, ascending=False)

        with open("queryResults.json", 'w', encoding='utf-8') as queryResultsFile:

            queryResultsJson[featuresType.name][similarityFunctionType.name][queryId] = json.loads(result.head(nrResults).to_json(orient="records"))
            json.dump(queryResultsJson, queryResultsFile)
            print("Write results")

    print("Use stored similarites for "+queryId)
    result = pd.DataFrame.from_dict(queryResultsJson[featuresType.name][similarityFunctionType.name][queryId])

    return result.head(nrResults)  # return only the `nrResult` best documents (max. 100)

## Returns the id for the concatenated "artist songTitle"

