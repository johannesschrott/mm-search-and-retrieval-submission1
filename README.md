# Multimedia Search and Retrieval Project

Group C. Members:

* Dominik Lugmair, k11776355@students.jku.at
* Akintunde Ojutiku, k12124446@students.jku.at
* Barbara Wakolbinger, k00955298@students.jku.at
* David Wirrer, k11809802@students.jku.at
* Johannes Schrott, k11904214@students.jku.at

## General remark
All features (.tsv files from moodle) as well as the database containing the similarities MUST be placed in the directory `data` on the top level of this repositiory. The file `constants.py` provides the possibility to look up the paths
and some other constants and also allows adjusting them.
Due to their size, all files in the `data` directory are NOT pushed to GitHub. 
The similarity database can be computed with the `main.py` file (see below how to do so) or can be obtained from us (please write an email; be aware that the size of the database is around 13 GB).

## How to run the project

### `main.py`: Compute similarities + evaluation
While the intention of our framework is to be reusable in other
(Python) programs, it also features a command line interface that
can be used through running the main.py file in the projects root.
main.py needs to be run with parameters for type of features, the type of similarity function
and an artist/song title.

The available similarity functions and feature types can be retrieved when running `main.py` without any arguments.

The Artist/Songname parameter is a string containing 1. the artist's name, 2. a semicolon and 3. the song title.

Example:
`python3 main.py tf-idf cosine "Nirvana;Come As You Are"`

`python3 main.py late cosine "Nirvana;Come As You Are"` 

Because of parameter 'late': uses the feature + similarity function combinations found to be best suited for borda count and specified in `__init__.py` (see below)

Parameter 'cosine' (i.e. the similarity function) is ignored, as soon as 'late' is used.
'late' cannot be used together with batch or all.

### `statistics_main.py`: Compute statistics over the dataset
Run this python script without any arguments to compute some statistics over the dataset. A discussion of the results of the script is done in the project report.


### `precisions_recall_plots.py`: Generate precision recall plots
This script generates an interactive precsion and recall plot. To do so, in the first run `app.run_server(debug=False)` 
needs to be commented and `get_precisions_and_recalls()` must be uncommented in order to calculate the data for the plots. 
In a second run `app.run_server(debug=False)` must to be uncommented and `get_precisions_and_recalls()` needs to be commented in order to start a built-in webserver.
The website this webserver delivers contains an interactive plot where all shown curves can be ticked on or off individually.


### `rank_order_correlations.py`: Rank order correlations 

This python script computes rank order correlations between all possible combinations between features and similarity functions. Simply run this file without any arguments, the results will be printed to the console.
Inside the script the variable `nr_of_songs` defines the number of songs of which the correlations are calculated. As a result always the mean of all songs for a specific k is returned.

## How to specifiy which features + similarity function combinations are used for borda count
In __init__.py specify them in the array LATE_FUSION_LABEL_FEATURESTYPE_FUNCS_AND_PATHS.
Each array entry shall have the form (label, FeaturesType, similarityFunction, featuresPathFromConstants).
label is arbitrary and used as column header in the dataframe with fused scores and also for printing to the terminal and writing into the results csv-file (templ_lateFusion_sortedByMean.csv and temp_lateFusion_notSorted.csv).
FeaturesType.<FEATURE_TYPE_NAME> must fit constants.<FEATURE_FILE_PATH>.

Example of 2 voters:
`LATE_FUSION_LABEL_FEATURESTYPE_FUNCS_AND_PATHS = [('COSINE BLF_CORRELATION', FeaturesType.BLF_CORRELATION, Cos_sim(), constants.AUDIO_BLF_CORRELATION_PATH),`
`('JACCARD BLF_DELTASPECTRAL', FeaturesType.BLF_DELTASPECTRAL, Jaccard(), constants.AUDIO_BLF_DELTASPECTRAL_PATH)]`

## How to specify what is used for the batch (and all) run.
In constants.py set the number of runs per retrieval algorithm (similarty function + feature combination) via constant `BATCH_COUNT` and the number of sample queries per batch via constant `BATCH_SIZE`. 
Each batch is re-used for the other retrieval algorithms to provide comparable accuracy metrics.
In main.py it can be redefined which features (variable `featureList`), similarity functions (`similarityList`) are to be combined for the batch run.

all: For running all queries there are the variables `featureList` and `simList`.

All those variables can only be set by directly adapting the source code as only intended to be changed by an experienced user, familiar with the technical field.

Restrictions:
Please, note that `FeaturesType.LATE` and `FeaturesType.EARLY` are not working for batch and all.


## Troubleshooting
In case there are errors due to missing similarities, check whether constant `NEWSIMILARITIES` of constants.py is set to true for re-calculating missing similarities.

If still not solved, do a batch run, preceding the intended run, with all feature + similarity function combinations that are needed for the actual run.

If still not working, try the previous steps after dropping the database, so that recalcing the needed similarities is forced.

