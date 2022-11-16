import pandas as pd

from similarity.Features import Features
import constants

class BERT(Features):
    def __init__(self):
        super().__init__()
        # the var features is inherited
        self.features = pd.read_csv(constants.BERT_PATH, sep="\t",index_col=0,header=0)
        print("Successfully read BERT features")
