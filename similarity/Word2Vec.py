import pandas as pd

from similarity.Features import Features
import constants


class Word2Vec(Features):
    def __init__(self):
        super().__init__()
        # the var features is inherited
        self.features = pd.read_csv(filepath_or_buffer=constants.WORD2VEC_PATH, sep="\t", index_col=0, header=0)
        print("Successfully read Word2Vec features")
