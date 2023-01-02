import os

import dash_bootstrap_components as dbc
import plotly.express as px
from dash import html, dcc

from app.figures_generator import get_phonemes_figure, get_phonemes_data

# define the colors for the plots
COLORS = {0: '#1f77b4', 1: '#ff7f0e'}

# get the names of the dense models and vocabulary sizes from the directories
dense_models = set()
vocab_sizes = set()
for dense_vocab in os.listdir("../assets/2d_area"):
    dense_models.update({dense_vocab.split("_")[0]})
    vocab_sizes.update({dense_vocab.split("_")[1].split(".")[0]})
vocab_sizes = sorted([int(x) for x in vocab_sizes])
# create options for the dropdown menus
DENSE_OPTIONS = [{"label": d, "value": d} for d in dense_models]
UNIT_OPTIONS = [{"label": v, "value": v} for v in vocab_sizes]
DATASETS = ['LIBRISPEECH', 'LJSPEECH']
FIGURE_TYPE_OPTIONS = [{"label": "Relations", "value": 'Relations'},
                       {"label": "Phonemes", "value": 'Phonemes'}]
COLOR_BY_OPTIONS = [{"label": "Phonemes", "value": 'Phonemes'},
                    {"label": "Phonemes Families", "value": 'Phonemes Families'}]
PCA_TSNE_OPTIONS = [{"label": "PCA", "value": "PCA"}, {"label": "T-SNE", "value": "T-SNE"}]
DIM_OPTIONS = [{"label": "2D", "value": "2D"}, {"label": "3D", "value": "3D"}]


def get_audio_table():
    """
    Returns the HTML elements for the audio table.
    """
    # create the table headers
    audio_table = [html.B("DataSet", style={'grid-column-start': 1, 'grid-column-end': 2, 'grid-row': "1"}),
                   html.B(DATASETS[0], style={'grid-column-start': 1, 'grid-column-end': 2, 'grid-row': "2"}),
                   html.B(DATASETS[1], style={'grid-column-start': 1, 'grid-column-end': 2, 'grid-row': "3"})]

    # create the audio players for each dataset
    for i, ds in enumerate(DATASETS):
        audio_table.append(html.Audio(id=f"audio-{ds}-raw", src="", controls=True, autoPlay=True,
                                      style={'grid-column-start': 4, 'grid-column-end': 6, 'grid-row': f"{i + 2}"}))
    return audio_table


def get_control_panel():
    fig_type = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H6("Figure Type", className="card-title"),
                    dcc.RadioItems(
                        options=FIGURE_TYPE_OPTIONS,
                        value=FIGURE_TYPE_OPTIONS[1]['value'],
                        id="opt_figure_type",
                    ),
                ]
            ),
        ],
    )

    config = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H6("Configuration", className="card-title"),
                    dcc.RadioItems(
                        options=DENSE_OPTIONS,
                        value=DENSE_OPTIONS[0]['value'],
                        id="opt_dense",
                    ),
                    dcc.Checklist(
                        options=UNIT_OPTIONS,
                        value=[UNIT_OPTIONS[1]['value']],
                        id="opt_units",
                    ),

                ]
            ),
        ],
    )

    color_by = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H6("Color By", className="card-title"),
                    dcc.RadioItems(
                        options=COLOR_BY_OPTIONS,
                        value=COLOR_BY_OPTIONS[0]['value'],
                        id="opt_color_by",
                    ),
                ]
            ),
        ],
    )

    decomp = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H6("Dimensionality Reduction", className="card-title"),
                    dcc.RadioItems(
                        options=PCA_TSNE_OPTIONS, value=PCA_TSNE_OPTIONS[0]['value'],
                        id="opt_pca_tsne",
                    ),
                    dcc.RadioItems(
                        options=DIM_OPTIONS, value=DIM_OPTIONS[0]['value'],
                        id="opt_dim",
                    )

                ]
            ),
        ],
    )

    audio_card = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H6("Audio", className="card-title"),
                    dcc.RadioItems(DATASETS, DATASETS[1], id="opt_autoplay",
                                   inline=True),
                    html.Audio(id=f"play-audio", src="", controls=True, autoPlay=True),
                    html.Div(style={'display': 'flex'}, children=[
                        html.Div("Search Unit:"),
                        dcc.Input(id='mark-unit', value="")]),
                    html.Div(style={'display': 'flex'}, children=[
                        html.Div("Selected Unit:"),
                        html.Div(id='selected-unit-name', children="")]),

                ]
            ),
        ],
    )

    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H4("Control Panal", className="card-title"),
                    fig_type, config, color_by, audio_card, decomp
                ]
            )
        ])


def get_html():
    controls = get_control_panel()
    return dbc.Container(
        [
            html.Div(style={'display': 'flex', 'justify-content': 'space-between'}, children=[
                html.Div(children=[
                    html.H2("Analysing Discrete Self Supervised Speech Representation for Spoken Language Modeling"),
                    html.H2(children=[dcc.Link(children="Textless NLP", href="https://speechbot.github.io/")], ),

                ]),
                html.Img(src="assets/logo.jpeg", style={'width': "100px", 'height': "100px"}),
            ]),
            html.Hr(),

            dbc.Row([
                dbc.Col(dbc.Form([controls])),
                dbc.Col(
                    dcc.Loading(dcc.Graph(id='2d-fig',
                                          figure=get_phonemes_figure(get_phonemes_data('hubert', str(100)),
                                                                     'Phonemes')))),
                dbc.Col(dcc.Graph(id='tree-fig', figure=px.treemap(width=150, height=600)))
            ]),
        ],
        fluid=True,
    )
