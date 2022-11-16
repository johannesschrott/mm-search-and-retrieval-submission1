import pandas as pd



class Features:
    """The class Features is a super class that provides common functionality for all kinds of features."""

    features: pd.DataFrame = None
    current: int = None
    rowNames = None
    def __init__(self):
        """Child classes should use the initializer for loading the features into the global variable `features`."""
        self.current = 0
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if self.rowNames is None:
            self.rowNames = self.features.index.values

        if self.current < len(self.features.axes[0]):
            res = (self.rowNames[self.current], self.features.loc[self.rowNames[self.current]])
            self.current = self.current + 1
            return res
        else:
            raise StopIteration

    def resetIterator(self):
        self.current = 0
    def getFeatureForId(self, _id: str) -> pd.DataFrame:
        """Returns the features for a specific song which is specified by its ID."""
        return self.features.loc[[_id]]