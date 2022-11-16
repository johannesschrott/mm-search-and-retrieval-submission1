import pandas as pd

from similarity.Features import Features
import constants

class TF_IDF(Features):
    def __init__(self):
        super().__init__()
        # the var features is inherited
        self.features = pd.read_csv(constants.TFIDF_PATH, sep="\t",index_col=0,header=0)
        print("Successfully read TF-IDF features")
