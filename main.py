"""
The main script to run the app.
Note: This script must follow the preprocessing script.
How to run the app:
python main.py
open  http://0.0.0.0:8080/ in the browser
"""

from typing import List, Tuple, Union

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import Dash, Output, Input, State

from app.figures_generator import get_main_figure, get_phonemes_figure, get_tree_map, get_phonemes_data
from app.html_components import get_html


def get_unit_and_vocab(click_data: dict, opt_units: List[int]) -> Tuple[Union[str, None], Union[int, None]]:
    """
    Given a dictionary containing data about a click event and a list of vocabulary sizes,
    returns a tuple containing the unit and vocabulary size corresponding to the click event.
    If the click data is invalid or the unit or vocabulary size cannot be determined,
    returns None for those values.
    """
    vocab_size = unit = None  # Initialize variables to None
    if click_data and 'text' in click_data['points'][0] and 'curveNumber' in click_data['points'][0]:
        # If the click data is valid and contains the necessary information
        unit = click_data['points'][0]['text']  # Get the unit from the click data
        vocab_size = opt_units[click_data['points'][0]['curveNumber']]  # Get the vocabulary size from the click data
    return unit, vocab_size  # Return the unit and vocabulary size

dash_app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = dash_app.server


@dash_app.callback(
    Output("2d-fig", "figure"),
    Input("opt_figure_type", "value"),
    Input("opt_dense", "value"),
    Input("opt_color_by", "value"),
    Input("opt_pca_tsne", "value"),
    Input("opt_units", "value"),
    Input("opt_dim", "value"),
    Input("mark-unit", "value"),
    Input("2d-fig", "clickData"),
    State("2d-fig", "relayoutData"),
)
def update_main_figure(
        opt_figure_type: str,
        opt_dense: str,
        opt_color_by: str,
        opt_pca_tsne: str,
        opt_units: List[int],
        opt_dim: str,
        mark_unit: str,
        click_data: dict,
        limits: dict,
) -> dict:
    """
    Given various input values and state data, returns a figure object to update the 2d-fig figure.
    The figure object is either a plot of relationships or a plot of phonemes, depending on the value of
    opt_figure_type.
    """
    if opt_figure_type == "Relations":
        # If the opt_figure_type is "Relations", get the unit and vocabulary size from the click data
        unit, vocab_size = get_unit_and_vocab(click_data, opt_units)
        # Generate the main figure using the unit and vocabulary size
        fig_2d = get_main_figure(
            opt_dense, opt_pca_tsne, opt_units, unit, vocab_size, limits, opt_dim, mark_unit
        )
    else:
        # If the opt_figure_type is not "Relations"
        # Get the ID of the triggering component
        triger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
        if triger_id == "2d-fig":
            # If the triggering component is the 2d-fig figure, return no update
            return dash.no_update
        # If the triggering component is not the 2d-fig figure, generate the phonemes figure
        fig_2d = get_phonemes_figure(get_phonemes_data(opt_dense, opt_units[0]), opt_color_by)

    return fig_2d  # Return the figure object


@dash_app.callback(
    Output("tree-fig", "figure"),
    Input("opt_units", "value"),
    Input("2d-fig", "clickData"),
    Input("opt_figure_type", "value"),
    Input("opt_dense", "value"),
)
def update_treemap_figure(
        opt_units: List[int], click_data: dict, opt_figure_type: str, opt_dense: bool
):
    """
    Given various input values, returns a figure object to update the tree-fig figure.
    The figure object is either a treemap plot or a placeholder figure, depending on the values of opt_figure_type
    and the triggering component.
    """
    # Get the ID of the triggering component
    triger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if opt_figure_type != "Relations":
        # If the opt_figure_type is not "Relations"
        if triger_id == "opt_figure_type":
            # If the triggering component is opt_figure_type, return a placeholder treemap plot
            return px.treemap(width=150, height=600)
        # If the triggering component is not opt_figure_type, return no update
        return dash.no_update
    if triger_id != "2d-fig":
        # If the opt_figure_type is "Relations" but the triggering component is not 2d-fig, return no update
        return dash.no_update
    # If the opt_figure_type is "Relations" and the triggering component is 2d-fig, get the unit and vocabulary size from the click data
    unit, vocab_size = get_unit_and_vocab(click_data, opt_units)
    # Generate the treemap plot using the unit and vocabulary size
    return get_tree_map(opt_dense, vocab_size, unit)


@dash_app.callback(
    [Output("selected-unit-name", "children"), Output("play-audio", "src")],
    [
        Input("tree-fig", "hoverData"),
        Input("2d-fig", "hoverData"),
        Input("opt_autoplay", "value"),
        Input("opt_figure_type", "value"),
        Input("opt_dense", "value"),
        State("opt_units", "value"),
    ],
)
def update_single_selection_audio(
        fig_tree_hover: dict,
        fig_2d_hover: dict,
        opt_autoplay: bool,
        opt_figure_type: str,
        opt_dense: bool,
        opt_units: List[int],
) -> Tuple[str, str]:
    """
    Given various input values and state data, returns a tuple containing the name of the selected unit and the
    audio file src.
    The selected unit and audio file src are determined based on the triggering component and the hover data from
    either the tree-fig or 2d-fig figure.
    """
    # Get the ID of the triggering component
    triger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if triger_id == "tree-fig":
        # If the triggering component is tree-fig
        # Get the label of the hovered data point and split it to get the unit and vocabulary size
        label = fig_tree_hover["points"][0]["label"]
        unit = label.split("(")[0]
        vocab_size = label.split("(")[1].replace(")", "")

    elif triger_id == "2d-fig":
        # If the triggering component is 2d-fig
        if opt_figure_type == "Relations":
            # If the opt_figure_type is "Relations", get the vocabulary size from the hover data and the unit
            # from the click data
            vocab_size = opt_units[fig_2d_hover["points"][0]["curveNumber"]]
            unit = fig_2d_hover["points"][0]["text"]
        else:
            # If the opt_figure_type is not "Relations", get the vocabulary size from the first element of
            # opt_units and the unit from the hover data
            vocab_size = opt_units[0]
            unit = fig_2d_hover["points"][0]["curveNumber"]
    else:
        # If the triggering component is not tree-fig or 2d-fig, return empty strings
        return ("", "")

    # Format the selected unit string using the unit, opt_dense, and vocabulary size
    selected_unit = f"{unit} ({opt_dense}{vocab_size})"
    # Format the audio file src using the opt_autoplay, vocabulary size, opt_dense, and unit
    src = f"assets/audio/{opt_autoplay}_{vocab_size}_{opt_dense}/{unit}.wav"
    return selected_unit, src  # Return the selected unit string and audio file src


dash_app.layout = get_html()
if __name__ == "__main__":
    dash_app.run_server(debug=False, use_reloader=False)