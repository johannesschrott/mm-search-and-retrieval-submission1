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

