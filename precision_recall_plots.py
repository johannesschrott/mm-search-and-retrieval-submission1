import logging

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import torch
from torchmetrics.functional import retrieval_precision_recall_curve

from multiprocessing import Manager

import constants
import database
import similarity
from evaluation.Evaluation import Evaluation
from evaluation.Genres import Genres
from tqdm import tqdm
from similarity import FeaturesType


'''Defines size of the head of songs that are used to compute the precision and recall'''
nr_of_songs = 1000

app = Dash(__name__)


app.layout = html.Div([
    html.H1('Interactive precision recall plots'),
    html.P(f"The first {nr_of_songs} songs of the data set were used to compute the values for the plot."),
    dcc.Graph(id="graph"),
    dcc.Checklist(
        id="checklist",
        labelClassName="bp4-control bp4-checkbox",
        inputClassName="bp4-control-indicator",
        options=[sim.name for sim in similarity.SimilarityFunctionType],
        value=[sim.name for sim in similarity.SimilarityFunctionType],
        inline=True
    ),
])
@app.callback(
    Output("graph", "figure"),
    Input("checklist", "value"))
def update_line_chart(sim_funcs):
    df = get_precisions_and_recalls() # replace with your own data source
    df.sort_values(by=["Recall"])
    mask = df.similarity_function.isin(sim_funcs)
    fig = px.line(df[mask],
        x="Recall", y="Precision", color="SimFunc with Feature",height=510)
    fig.update_layout(yaxis_tickformat=".2%", xaxis_tickformat=".2%")
    fig.update_xaxes(range=[0,0.002],nticks=11)
    fig.update_yaxes(range=[0,1],nticks=11)
    return fig


def get_precisions_and_recalls():
    #result = pd.DataFrame({"Similarity Function": pd.Series(dtype="str"),
    #                      "Feature": pd.Series(dtype="str"),
    #                      "Precision": pd.Series(dtype=float),
    #                       "Recall": pd.Series(dtype=float)})
    precision_name = f"data/batch_runs/precision_recall_run_{nr_of_songs}_eval.csv"
    try:
        result = pd.read_csv(precision_name)
    except FileNotFoundError:
        # if the results have never been computed, they need to be computed
        sim_func_list = []
        sim_func_x_feature_list = []
        precision_list = []
        recall_list = []

        eval = Evaluation()
        genres = Genres()
        splitGenres = eval.splitGenresForAllSongs(genres.genres)
        ids = pd.read_csv(constants.IDS_PATH, sep="\t", index_col=3, header=0)["id"].values

        simList = [similarity.SimilarityFunctionType.BASELINE,
                   similarity.SimilarityFunctionType.COSINE_SIMILARITY,
                   similarity.SimilarityFunctionType.INNER_PRODUCT,
                   similarity.SimilarityFunctionType.JACCARD
                   ]

        for sim_func in tqdm(simList):
            logging.info(f"Progress on features in similarity type {sim_func}")
            featureList = [FeaturesType.LYRICS_TF_IDF,
                           FeaturesType.LYRICS_WORD2VEC,
                           FeaturesType.LYRICS_BERT,
                           FeaturesType.BLF_CORRELATION,
                           FeaturesType.BLF_DELTASPECTRAL,
                           FeaturesType.BLF_LOGFLUC,
                           FeaturesType.BLF_SPECTRAL,
                           FeaturesType.BLF_SPECTRALCONTRAST,
                           FeaturesType.BLF_VARDELTASPECTRAL,
                           FeaturesType.ESSENTIA,
                           FeaturesType.MFCC_BOW,
                           FeaturesType.MFCC_STATS,
                           FeaturesType.INCP,
                           FeaturesType.RESNET,
                           FeaturesType.VGG19,
                           FeaturesType.EARLY,
                           FeaturesType.EARLY_ESSENTIA_VDS,
                           FeaturesType.EARLY_W2V_BERT
                           ]

            for feature in tqdm(featureList):
                precisions_at_k = [0 for i in range(0,100)]
                recalls_at_k = [0 for i in range(0,100)]
                #for i in range(0,nr_of_songs):

                with Manager() as manager:
                    precisions_at_k = manager.list([0 for _ in range(0,100)])
                    recalls_at_k = manager.list([0 for _ in range(0,100)])

                    with manager.Pool() as pool:
                        multipleResults = [
                            pool.apply_async(songLoop, args=(eval, feature,
                                                             i, ids, precisions_at_k,
                                                             recalls_at_k, sim_func, splitGenres))
                            for i in range(0, nr_of_songs)]
                        #songLoop(eval, feature, i, ids, precisions_at_k, recalls_at_k, sim_func, splitGenres)

                        #print("Catching results of computation...")
                        [res.wait() for res in multipleResults]

                    #print(precisions_at_k)
                    precisions_at_k = [precision/nr_of_songs for precision in precisions_at_k]
                    recalls_at_k = [recall/nr_of_songs for recall in recalls_at_k]

                for precision in precisions_at_k:
                    precision_list.append(precision)
                    sim_func_list.append(sim_func.name)
                    sim_func_x_feature_list.append(f"{sim_func.name}: {feature.name}")

                for recall in recalls_at_k:
                    recall_list.append(recall)

        result = pd.DataFrame()
        result["similarity_function"] = sim_func_list
        result["SimFunc with Feature"] = sim_func_x_feature_list
        result["Precision"] = precision_list
        result["Recall"] = recall_list

        result.to_csv(precision_name)

    return result


def songLoop(eval, feature, i, ids, precisions_at_k, recalls_at_k, sim_func, splitGenres):
    row_id_similarities = similarity.doQueryWithId(ids[0], 100, feature, sim_func).values
    row_id_similarities = [(row_id_similarity[0], row_id_similarity[1]) for row_id_similarity in row_id_similarities]
    relevant_songs = eval.getGenreSharingSongsFromId(ids[i], splitGenres)
    retrieved_songs = [row_id_similarity[0] for row_id_similarity in row_id_similarities]
    retrieved_sim_vals = [row_id_similarities[retrieved_songs.index(ids[i])][1] if ids[i] in retrieved_songs else 0.0
                          for i in range(0, constants.NR_OF_SONGS)]
    all_relevants = [ids[i] in relevant_songs if ids[i] in relevant_songs else False for i in
                     range(0, constants.NR_OF_SONGS)]
    retrieved_torch = torch.tensor(retrieved_sim_vals)
    relevant_torch = torch.tensor(all_relevants)
    precisions_torch, recalls_torch, _ = retrieval_precision_recall_curve(retrieved_torch, relevant_torch, max_k=100)
    precisions = precisions_torch.numpy()
    recalls = recalls_torch.numpy()
    for i in range(0, 100):
        precisions_at_k[i] = precisions_at_k[i] + precisions[i]
        recalls_at_k[i] = recalls_at_k[i] + recalls[i]



#
# df = px.data.gapminder()  # replace with your own data source
# mask = df.continent.isin(continents)
# print(df)
# test = df[mask]
# print[test]

if __name__ == '__main__':
# Uncomment this to run interactive plot as website
    app.run_server(debug=True)

# uncomment this to create a new data source for precision and recall using the defined nr of songs
    #get_precisions_and_recalls()

