IDS_PATH = "data/id_information_mmsr.tsv"
GENRES_PATH = "data/id_genres_mmsr.tsv"
LYRICS_TFIDF_PATH = "data/id_lyrics_tf-idf_mmsr.tsv"
LYRICS_WORD2VEC_PATH = "data/id_lyrics_word2vec_mmsr.tsv"
LYRICS_BERT_PATH = "data/id_lyrics_bert_mmsr.tsv"
AUDIO_BLF_CORRELATION_PATH = "data/id_blf_correlation_mmsr.tsv"
AUDIO_BLF_DELTASPECTRAL_PATH = "data/id_blf_deltaspectral_mmsr.tsv"
AUDIO_BLF_LOGFLUC_PATH = "data/id_blf_logfluc_mmsr.tsv"
AUDIO_BLF_SPECTRAL_PATH = "data/id_blf_spectral_mmsr.tsv"
AUDIO_BLF_SPECTRALCONTRAST_PATH = "data/id_blf_spectralcontrast_mmsr.tsv"
AUDIO_BLF_VARDELTASPECTRAL_PATH = "data/id_blf_vardeltaspectral_mmsr.tsv"
AUDIO_ESSENTIA_PATH = "data/id_essentia_mmsr.tsv"
AUDIO_MFCC_BOW_PATH = "data/id_mfcc_bow_mmsr.tsv"
AUDIO_MFCC_STATS_PATH = "data/id_mfcc_stats_mmsr.tsv"
VIDEO_INCP_PATH = "data/id_incp_mmsr.tsv"
VIDEO_RESNET_PATH = "data/id_resnet_mmsr.tsv"
VIDEO_VGG19_PATH = "data/id_vgg19_mmsr.tsv"


NR_OF_SONGS = 68641

PROGRAM_NAME = "Multimedia Search and Retrieval WS2022, Group C"
PROGRAM_VERSION = "Submission 2"

######################
# Database Constants #
######################

DATABASE_PATH = "data/database.sqlite"
# Constants for the row <-> id mapping table
ROW_ID_TABLENAME = "row_id_correspondence"
ROW_ID_COL_ROWNR = "nr"
ROW_ID_COL_ID = "id"

# Constants for the song similarity table
ROW_SONG_SIMILARITIES_TABLENAME = "song_similarities"
ROW_SONG_SIMILARITIES_SONG1_ID = "song1Id"
ROW_SONG_SIMILARITIES_SONG2_ID = "song2Id"
ROW_SONG_SIMILARITIES_SIMILARITY_VALUE = "similarityValue"
ROW_SONG_SIMILARITIES_SIMILARITY_TYPE = "similarityType"
ROW_SONG_SIMILARITIES_FEATURE_TYPE = "featureType"

# Constant for computing whole db of similarities
NEWSIMILARITIES = True