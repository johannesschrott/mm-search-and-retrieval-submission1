import logging
import pandas as pd
import sqlite3
import constants


'''Determines the id for the row number in the dataset.
Example: the first song in the dataset (= row 0) has id "9jbSytob9XRzwvB6"
'''
def get_id_from_row_nr(rowNr: int) -> str:
    connection = sqlite3.connect(constants.DATABASE_PATH)
    cur = connection.cursor()

    try:
        cur.execute(f"SELECT {constants.ROW_ID_COL_ID} FROM {constants.ROW_ID_TABLENAME} WHERE {constants.ROW_ID_COL_ROWNR} = {rowNr}")

        result = cur.fetchone()
        result = result[0]
        #logging.info(f"ID for row nr. {rowNr}:\t{result}")
        connection.commit()
        connection.close()


    except sqlite3.OperationalError:
        logging.warning(f"Database table {constants.ROW_ID_TABLENAME} does not exist and will be created and populated.")
        create_row_nr_id_table(cur)
        connection.commit()
        connection.close()
        result = (get_id_from_row_nr(rowNr))


    return result




'''Determines the the row number of a song given by its id in the dataset.
Example: the first song with id "9jbSytob9XRzwvB6" has row nr 0
'''
def get_row_nr_from_id(id: str) -> int:
    connection = sqlite3.connect(constants.DATABASE_PATH)
    cur = connection.cursor()

    try:
        cur.execute(f"SELECT {constants.ROW_ID_COL_ROWNR} FROM {constants.ROW_ID_TABLENAME} WHERE {constants.ROW_ID_COL_ID} = '{id}'")

        result = cur.fetchone()
        result = result[0]
        #logging.info(f"Row nr. for id. {id}:\t{result}")
        connection.commit()
        connection.close()

    except sqlite3.OperationalError:
        logging.warning(f"Database table {constants.ROW_ID_TABLENAME} does not exist and will be created and populated.")
        create_row_nr_id_table(cur)
        connection.commit()
        connection.close()
        result = get_row_nr_from_id(id)



    return result



'''Inserts a similarity value into the database'''
def insert_similarity(song1id: str, song2id: str, similarityValue: float, similarityType: int, featureType: int):
    connection = sqlite3.connect(constants.DATABASE_PATH)
    cur = connection.cursor()

    try:
        cur.execute(f"INSERT INTO {constants.ROW_SONG_SIMILARITIES_TABLENAME} VALUES ({song1id}, {song2id}, {similarityValue}, {similarityType}, {featureType})")
        connection.commit()
        connection.close()

    except sqlite3.OperationalError:
        logging.warning(f"Database table {constants.ROW_SONG_SIMILARITIES_TABLENAME} does not exist and will be created.")
        create_song_similarity_table(cur)
        connection.commit()
        connection.close()
        insert_similarity(song1id,song2id,similarityValue,similarityType)



def insert_similarities(values):
    connection = sqlite3.connect(constants.DATABASE_PATH)
    cur = connection.cursor()

    try:
        cur.executemany(f"INSERT INTO {constants.ROW_SONG_SIMILARITIES_TABLENAME} VALUES (?,?,?,?,?)", values)
        connection.commit()
        connection.close()

    except sqlite3.OperationalError:
        logging.warning(f"Database table {constants.ROW_SONG_SIMILARITIES_TABLENAME} does not exist and will be created.")
        create_song_similarity_table(cur)
        connection.commit()
        connection.close()
        insert_similarities(values)



'''Gets the similarity value computed with a specific similarity function for two given songs'''
def get_similarity_value(song1: str, song2: str, type: int) -> float:
    connection = sqlite3.connect(constants.DATABASE_PATH)
    cur = connection.cursor()

    try:
        cur.execute(f"""
        SELECT {constants.ROW_SONG_SIMILARITIES_SIMILARITY_VALUE} FROM {constants.ROW_SONG_SIMILARITIES_TABLENAME} 
        WHERE 
           ({constants.ROW_SONG_SIMILARITIES_SONG1_ID} = {id} OR {constants.ROW_SONG_SIMILARITIES_SONG2_ID} = {id}) AND
           {constants.ROW_SONG_SIMILARITIES_SIMILARITY_TYPE} = {type}
        """)
        result = cur.fetchone()
        connection.commit()
        connection.close()


    except sqlite3.OperationalError:
        logging.warning(
            f"Database table {constants.ROW_SONG_SIMILARITIES_TABLENAME} does not exist and will be created.")
        create_song_similarity_table(cur)
        connection.commit()
        connection.close()
        result = []


    return result

def get_n_best_similarity_values(song1: int, n:int, similarityType: int, featureType: int) -> list:
    assert 1 <= n <= constants.NR_OF_SONGS

    connection = sqlite3.connect(constants.DATABASE_PATH)
    cur = connection.cursor()

    try:
        cur.execute(f"""
        SELECT {constants.ROW_SONG_SIMILARITIES_SONG1_ID},
            {constants.ROW_SONG_SIMILARITIES_SONG2_ID},
            {constants.ROW_SONG_SIMILARITIES_SIMILARITY_VALUE} 
            FROM {constants.ROW_SONG_SIMILARITIES_TABLENAME} 
        WHERE 
           {constants.ROW_SONG_SIMILARITIES_SONG1_ID} = {song1} AND
           {constants.ROW_SONG_SIMILARITIES_SIMILARITY_TYPE} = {similarityType} AND
           {constants.ROW_SONG_SIMILARITIES_FEATURE_TYPE} = {featureType}
        ORDER BY {constants.ROW_SONG_SIMILARITIES_SIMILARITY_VALUE} DESC
        LIMIT {n}
        """)

        result = cur.fetchall()

    except sqlite3.OperationalError:
        logging.warning(
            f"Database table {constants.ROW_SONG_SIMILARITIES_TABLENAME} does not exist and will be created.")
        create_song_similarity_table(cur)
        connection.commit()
        result = get_similarity_value(song1, n, similarityType)

    connection.commit()
    connection.close()

    return result

################
# Table Create #
################


def create_row_nr_id_table(cur):
    try:
        # create table
        logging.info(f"Create a new table {constants.ROW_ID_TABLENAME}")
        cur.execute(
            f"CREATE TABLE {constants.ROW_ID_TABLENAME}({constants.ROW_ID_COL_ROWNR} INT, {constants.ROW_ID_COL_ID} TEXT)")

        # load the correspondences
        data = pd.read_csv(constants.GENRES_PATH, sep="\t", index_col=0)
        data.reset_index(inplace=True)
        indices = data["id"].tolist()
        for i in range(0, len(data)):
            cur.execute(
                f"INSERT INTO {constants.ROW_ID_TABLENAME}({constants.ROW_ID_COL_ROWNR}, {constants.ROW_ID_COL_ID}) VALUES (?,?)",
                (i, indices[i]))
    except sqlite3.Error as e:
        logging.error(e)

def create_song_similarity_table(cur):
    try:
        # create table
        logging.info(f"Create a new table {constants.ROW_SONG_SIMILARITIES_TABLENAME}")
        cur.execute(
            f"""
            CREATE TABLE {constants.ROW_SONG_SIMILARITIES_TABLENAME}(
              {constants.ROW_SONG_SIMILARITIES_SONG1_ID} INTEGER, 
              {constants.ROW_SONG_SIMILARITIES_SONG2_ID} INTEGER,
              {constants.ROW_SONG_SIMILARITIES_SIMILARITY_VALUE} REAL,
              {constants.ROW_SONG_SIMILARITIES_SIMILARITY_TYPE} INTEGER,
              {constants.ROW_SONG_SIMILARITIES_FEATURE_TYPE} INTEGER
            )
            """)

        # load the correspondences
       # data = pd.read_csv(constants.GENRES_PATH, sep="\t", index_col=0)
        #for i in range(0, len(data)):
        #    cur.execute(
         #       f"INSERT INTO {constants.ROW_ID_TABLENAME}({constants.ROW_ID_COL_ROWNR, constants.ROW_ID_COL_ID}) VALUES (?,?)",
          #      (i, data[i]))
    except sqlite3.Error as e:
        logging.error(e)

