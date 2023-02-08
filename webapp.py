import dash as d
import dash_bootstrap_components as dbc
from fuzzywuzzy import fuzz

import pandas as pd

import constants
from similarity import *

app = d.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

app.title = constants.PROGRAM_NAME

####################
# Data preperation #
####################

options_df = pd.read_csv(constants.IDS_PATH, sep="\t", index_col=3, header=0)
options_df = options_df.reset_index()
options_df["label"] = "Title: "+options_df["song"]+"; Interpret: "+options_df["artist"] #+"; Album name: "+options_df["album_name"]
options_df["value"] = options_df["id"]

youtube_df =  pd.read_csv(constants.YOUTUBE_URL_PATH, sep="\t", index_col=0, header=0)
spotify_df =  pd.read_csv(constants.SPOTIFY_METADATA_PATH, sep="\t", index_col=0, header=0)


id_file_df = pd.read_csv(constants.IDS_PATH, sep="\t", index_col=3, header=0)

for_retrieval_by_ids = id_file_df.copy()
for_retrieval_by_ids.set_index("id", inplace=True)
def getSongForId(string: str):
    return for_retrieval_by_ids.loc[string]["song"]

def getAlbumForId(string: str):
    return for_retrieval_by_ids.loc[string]["album_name"]

def getArtistForId(string: str):
    return for_retrieval_by_ids.loc[string]["artist"]

def getYouTubeLink(string: str):
    return youtube_df.loc[string]["url"]

def getSpotifyId(string: str):
    return spotify_df.loc[string]["spotify_id"]

##########
# Layout #
##########

dropdown_selection = dbc.Container([d.html.Br(), dbc.Form([dbc.Row([dbc.Label("Query:", html_for="song_selection", width=2,md=1),
                                        dbc.Col([d.dcc.Dropdown(id="song_selection",placeholder=constants.WEB_DROPDOWN_SELECT_PLACEHOLDER,persistence=True)],width=10,md=11)
                                        ])],className="mb-3",)])

navbar = dbc.NavbarSimple(
    brand=constants.PROGRAM_NAME,
    brand_href="#",
    color="primary",
    dark=True,
    fixed=True
    )


search_result = d.dcc.Loading(
            id="search_result_loading",
            type="default",
            children=d.html.Div(id="search_loading_output"),
            fullscreen=False
        )


main_content = d.html.Div(children=[dropdown_selection,search_result])



app.layout = d.html.Div(children=[
    navbar,
    main_content
])

#################
# Program Logic #
#################

@app.callback(
    d.Output("song_selection", "options"),
    d.Input("song_selection", "search_value")
)
def update_search_options(search_value):
    if not search_value:
        raise d.exceptions.PreventUpdate

    return [{"label": row.label, "value": row.value} for row in options_df.itertuples() if str.upper(search_value) in str.upper(row.label)][:10]

@app.callback(
    d.Output("search_loading_output", "children"),
    d.Input("song_selection", "value")
)
def update_search_result(value):
    if not value:
        raise d.exceptions.PreventUpdate
    else:
        similarSongs = doQueryWithId(value, 100, FeaturesType.EARLY, SimilarityFunctionType.COSINE_SIMILARITY)

        result_page = [d.html.H1(children=f"Results for \"{getSongForId(value)}\" by {getArtistForId(value)}:"),
                       d.html.Div([d.html.A(href=getYouTubeLink(value),target="_blank",
                                                 children=[
                                                     dbc.Button(
                                                         children=[f"Show \"{getSongForId(value)}\" on ", d.html.I(className="bi bi-youtube"), " YouTube"],
                                                     id=f"youtube_{value}")
                                                 ]),
                                                    dbc.Tooltip(
                                                        f"Show the music video of \"{getSongForId(value)}\" from YouTube",
                                                        target=f"youtube_{value}",
                                                        placement="top"),
                                        ]),d.html.Br()]

        if (constants.WEB_APP_PRODUCTION):
            table_header = [
                d.html.Thead(d.html.Tr([
                    d.html.Th("Rank"),
                    d.html.Th("Song title"),
                    d.html.Th("Artist"),
                   # d.html.Th("Similarity value"),
                    d.html.Th("Listen on YouTube"),
                    d.html.Th("Listen on Spotify")

                ]))
            ]
        else:
            table_header = [
                d.html.Thead(d.html.Tr([
                    d.html.Th("Rank"),
                    d.html.Th("Song ID"),
                    d.html.Th("Song title"),
                    d.html.Th("Artist"),
                    d.html.Th("Similarity value"),
                    d.html.Th("Listen on YouTube"),
                    d.html.Th("Listen on Spotify")

                ]))
            ]

        table_rows = []

        i = 1

        for song in similarSongs.itertuples():
            if (constants.WEB_APP_PRODUCTION):
                table_rows.append(d.html.Tr([
                    d.html.Td(f"{i}."),
                    d.html.Td(getSongForId(song.id)),
                    d.html.Td(getArtistForId(song.id)),
                    #d.html.Td("{0:.4f}".format(song.similarity)),
                    d.html.Td(children=[d.html.Div([d.html.A(href=getYouTubeLink(song.id),target="_blank",
                                                 children=[
                                                     dbc.Button(
                                                         children=[d.html.I(className="bi bi-youtube"), " YouTube"],
                                                     id=f"youtube_{song.id}")
                                                 ]),
                                                    dbc.Tooltip(
                                                        f"Show the music video of \"{getSongForId(song.id)}\" from YouTube",
                                                        target=f"youtube_{song.id}",
                                                        placement="top"),
                                        ])]),
                        d.html.Td(children=[d.html.Div([d.html.A(href=f"https://open.spotify.com/track/{getSpotifyId(song.id)}",
                                                                 id=f"spotify_{song.id}", target="_blank",
                                                     children=[
                                                     dbc.Button(
                                                         children=[d.html.I(className="bi bi-spotify"), " Spotify"])
                                                 ]),dbc.Tooltip(
            f"Listen to \"{getSongForId(song.id)}\" on Spotify",
            target=f"spotify_{song.id}",
                            placement="top"
        ),
                                                        ])
                                        ])
                ]))
            else:
                table_rows.append(d.html.Tr([
                    d.html.Td(i),
                    d.html.Td(song.id),
                    d.html.Td(getSongForId(song.id)),
                    d.html.Td(getArtistForId(song.id)),
                    d.html.Td(song.similarity),
                    d.html.Td(children=[d.html.A(href=getYouTubeLink(song.id),
                                                 children=[
                                                    dbc.Button(children=[d.html.I(className="bi bi-youtube")," YouTube"])
                                        ])
                    ]),
                    d.html.Td(children=[d.html.A(href=f"https://open.spotify.com/track/{getSpotifyId(song.id)}",
                                                 children=[
                                                     dbc.Button(children=[d.html.I(className="bi bi-spotify"), " Spotify"])
                                                 ])
                                        ])
                ]))
            i += 1

        table_body = [d.html.Tbody(table_rows)]

        table = dbc.Table(table_header + table_body, bordered=True,striped=True,responsive=True)

        result_page.append(table)

        return dbc.Container(result_page)

if __name__ == "__main__":
    app.run_server(debug=True,port=8080,host="0.0.0.0")