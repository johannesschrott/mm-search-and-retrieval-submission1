import database
from similarity import *

print("Hubness At 100:")
for featureType in FeaturesType:
    for similarityType in [SimilarityFunctionType.COSINE_SIMILARITY, SimilarityFunctionType.JACCARD, SimilarityFunctionType.INNER_PRODUCT]:
        if featureType is FeaturesType.LATE:
            pass
        else:
            print(f"  {featureType.name};\t {similarityType.name} Hubness: {database.get_hubness_at_k(k=100,featureType=featureType.value,similarityType=similarityType.value)}")

print("Hubness At 10:")
for featureType in FeaturesType:
    for similarityType in [SimilarityFunctionType.COSINE_SIMILARITY, SimilarityFunctionType.JACCARD, SimilarityFunctionType.INNER_PRODUCT]:
        if featureType is FeaturesType.LATE:
            pass
        else:
            print(f"  {featureType.name};\t {similarityType.name} Hubness: {database.get_hubness_at_k(k=10, featureType=featureType.value, similarityType=similarityType.value)}")
