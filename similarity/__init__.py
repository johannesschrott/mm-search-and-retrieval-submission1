from enum import Enum

import constants
from similarity.Jaccard import Jaccard

from similarity.Cos_sim import Cos_sim
from similarity.Inner_product import Inner_product
import pandas as pd

class FeaturesType(Enum):
    """Enum that provides all possible feature types (for usage in parameters)"""
    LYRICS_TF_IDF = 0
    LYRICS_WORD2VEC = 1
    LYRICS_BERT = 2
    BLF_CORRELATION = 3
    BLF_DELTASPECTRAL = 4
    BLF_LOGFLUC = 5
    BLF_SPECTRALCONTRAST = 6
    BLF_VARDELTASPECTRAL = 7
    ESSENTIA = 8
    INCP = 9
    MFCC_BOW = 10
    MFCC_STATS = 11
    RESNET = 12
    VGG19 = 13
    BLF_SPECTRAL = 14
    EARLY = 25
    LATE = 26
    EARLY_W2V_BERT = 27
    EARLY_ESSENTIA_VDS = 28





class SimilarityFunctionType(Enum):
    """Enum that provides all possible types of the similarity function (for usage in parameters)"""
    COSINE_SIMILARITY = 1
    JACCARD = 2
    INNER_PRODUCT = 3
    BASELINE = 4


# (label, FeaturesType, similarityFunction, featuresPathFromConstants)
LATE_FUSION_LABEL_FEATURESTYPE_FUNCS_AND_PATHS = [('COSINE BLF_CORRELATION', FeaturesType.BLF_CORRELATION, Cos_sim(), constants.AUDIO_BLF_CORRELATION_PATH),
                               ('COSINE BLF_DELTASPECTRAL', FeaturesType.BLF_DELTASPECTRAL, Cos_sim(), constants.AUDIO_BLF_DELTASPECTRAL_PATH),
                               ('COSINE BLF_LOGFLUC', FeaturesType.BLF_LOGFLUC, Cos_sim(), constants.AUDIO_BLF_LOGFLUC_PATH),
                               ('INNER_PRODUCT BLF_SPECTRALCONTRAST', FeaturesType.BLF_SPECTRALCONTRAST, Inner_product(), constants.AUDIO_BLF_SPECTRALCONTRAST_PATH),
                               ('COSINE BLF_VARDELTASPECTRAL', FeaturesType.BLF_VARDELTASPECTRAL, Cos_sim(), constants.AUDIO_BLF_VARDELTASPECTRAL_PATH),
                               ('COSINE ESSENTIA', FeaturesType.ESSENTIA, Cos_sim(), constants.AUDIO_ESSENTIA_PATH),
                               ('JACCARD ESSENTIA', FeaturesType.ESSENTIA, Jaccard(), constants.AUDIO_ESSENTIA_PATH)]


# returns <nrResults> best documents for selected similarityFunction or baseline,
# if FeaturesType.LATE is selected as featuresType, an array of <nrResults> best documents
# (1 array per voter) are returned.
# a "voter" is defined as feature+similarity function combination from variable
# LATE_FUSION_LABEL_FEATURESTYPE_FUNCS_AND_PATHS.
# similarityFunctionType parameter is ignored, if featuresType == FeaturesType.Late, as also given by
# LATE_FUSION_LABEL_FEATURESTYPE_FUNCS_AND_PATHS.
def doQueryWithId(queryId: str, nrResults: int, featuresType: FeaturesType,
                  similarityFunctionType: SimilarityFunctionType):


    similarityFunction = None


    featuresPath: str = None

    match featuresType:
        case FeaturesType.LYRICS_WORD2VEC:
            featuresPath = constants.LYRICS_WORD2VEC_PATH
        case FeaturesType.LYRICS_TF_IDF:
            featuresPath = constants.LYRICS_TFIDF_PATH
        case FeaturesType.LYRICS_BERT:
            featuresPath = constants.LYRICS_BERT_PATH
        case FeaturesType.VGG19:
            featuresPath = constants.VIDEO_VGG19_PATH
        case FeaturesType.BLF_CORRELATION:
            featuresPath = constants.AUDIO_BLF_CORRELATION_PATH
        case FeaturesType.BLF_SPECTRAL:
            featuresPath = constants.AUDIO_BLF_SPECTRAL_PATH
        case FeaturesType.BLF_DELTASPECTRAL:
            featuresPath = constants.AUDIO_BLF_DELTASPECTRAL_PATH
        case FeaturesType.BLF_LOGFLUC:
            featuresPath = constants.AUDIO_BLF_LOGFLUC_PATH
        case FeaturesType.BLF_SPECTRALCONTRAST:
            featuresPath = constants.AUDIO_BLF_SPECTRALCONTRAST_PATH
        case FeaturesType.BLF_VARDELTASPECTRAL:
            featuresPath = constants.AUDIO_BLF_VARDELTASPECTRAL_PATH
        case FeaturesType.ESSENTIA:
            featuresPath = constants.AUDIO_ESSENTIA_PATH
        case FeaturesType.INCP:
            featuresPath = constants.VIDEO_INCP_PATH
        case FeaturesType.MFCC_BOW:
            featuresPath = constants.AUDIO_MFCC_BOW_PATH
        case FeaturesType.MFCC_STATS:
            featuresPath = constants.AUDIO_MFCC_STATS_PATH
        case FeaturesType.RESNET:
            featuresPath = constants.VIDEO_RESNET_PATH
        case FeaturesType.EARLY:
            featuresPath = constants.EARLY_PATH
        case FeaturesType.LATE:
            featuresPath = None
        case FeaturesType.EARLY_ESSENTIA_VDS:
            featuresPath = constants.EARLY_ESSENTIA_VARDELTASPECTRAL
        case FeaturesType.EARLY_W2V_BERT:
            featuresPath = constants.EARLY_LYRICS_W2V_BERT

    match similarityFunctionType:
        case SimilarityFunctionType.JACCARD:
            similarityFunction = Jaccard()
        case SimilarityFunctionType.COSINE_SIMILARITY:
            similarityFunction = Cos_sim()
        case SimilarityFunctionType.INNER_PRODUCT:
            similarityFunction = Inner_product()

    if (similarityFunctionType.value != SimilarityFunctionType.BASELINE.value):
        if featuresType != FeaturesType.LATE: # on similarity function + features selection
            result = similarityFunction.computeHighestSimilaritiesWithMatrix(queryId, nrResults, featuresType, featuresPath)
            return result.head(nrResults)  # return only the `nrResult` best documents (max. 100)
        else:  # best suited combination of similarity functions + features
            result = []
            for featureSimCombo in LATE_FUSION_LABEL_FEATURESTYPE_FUNCS_AND_PATHS:
                # featureSimCombo: (label, FeaturesType, similarityFunction, featuresPathFromConstants)
                nextResult = featureSimCombo[2].computeHighestSimilaritiesWithMatrix(queryId, nrResults, featureSimCombo[1], featureSimCombo[3])
                result.append(nextResult.head(nrResults)) # return only the `nrResult` best documents (max. 100) PER similarity+feature combo
            return result

    else:
        # The very dumb baseline approach
        # iţaussumes all songs with genre rock are similar on the same level
        genres = pd.read_csv(filepath_or_buffer=constants.GENRES_PATH, sep="\t", index_col=0, header=0)
        genres.reset_index(inplace=True)
        ids = genres["id"].tolist()
        genres = genres["genre"].tolist()
        genres_list = []
        for genre in genres:
            genre = genre.replace('[', '')
            genre = genre.replace(']', '')
            genre = genre.replace('\'', '')
            genre = genre.strip(" ").split(', ')
            genres_list.append(genre)
        the_list = list(zip(ids,genres_list))

        result_list = list(filter(lambda genre: "rock" in genre[1], the_list))

        result = pd.DataFrame()
        ids = [result[0] for result in result_list]
        similarity_values = [1.0 for result in result_list]

        result["id"] = ids
        result["similarity"] = similarity_values

        return result.head(nrResults)  # return only the `nrResult` best documents (max. 100)


