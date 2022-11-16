import pandas as pd
import constants

class Genres:
    """The class Genres is a super class that provides common functionality."""

    genres: pd.DataFrame = None

    def __init__(self):
        print("init genres from ", constants.GENRES_PATH)
        self.genres = pd.read_csv(filepath_or_buffer=constants.GENRES_PATH, sep="\t",index_col=0, header=0)
        """.transpose()"""
        #print("self.genres: ", self.genres)

    def getGenreForId(self, _id: str) -> pd.DataFrame:
        """Returns the genres for a specific song which is specified by its ID."""
        print("getGenreForId")
        return self.genres[_id]

    def getAllGenres(self):
        print("getAllGenres")
        return self.genres

