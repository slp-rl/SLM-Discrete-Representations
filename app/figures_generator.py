from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from preprocessing.phonemes_manager import phone_name_to_color, phone_name_to_family_color, phone_txt_to_family_txt, \
    phone_name_to_family_name

COLORS = {50: '#1F77B4', 100: '#FF7F0E', 200: '#8C564B'}


def get_phonemes_data(dense_model: str, vocab: str) -> pd.DataFrame:
    """
    Reads in the unit areas and unit-to-phoneme mapping data from CSV files and
    returns a single Pandas DataFrame with the concatenated data.

    Parameters:
    - dense_model: the dense model name
    - vocab: the vocabulary size

    Returns:
    - A Pandas DataFrame with the concatenated data.
    """
    # Construct the file paths
    units_ares_path = f"../assets/2d_area/{dense_model}_{vocab}.csv"
    units_to_phonemes_path = f"../assets/units_phonemes/{dense_model}_{vocab}_stats.csv"

    # Read in the data from the CSV files
    units_ares = pd.read_csv(units_ares_path, index_col=0)
    units_to_phonemes = pd.read_csv(units_to_phonemes_path, index_col=0)

    # Concatenate the data into a single DataFrame
    df = pd.concat([units_ares, units_to_phonemes], axis=1)

    return df


def row_to_text(row: pd.Series) -> str:
    """
    Converts a row of data from a Pandas DataFrame into a string with the top three phonemes and their
    percentages.

    Parameters:
    - row: a row of data from a Pandas DataFrame.

    Returns:
    - A string with the top three phonemes and their percentages.
    """
    text = f"{row['top_0_phoneme']} ({row['top_0_percentage']:.0%}) \n"
    text += f"{row['top_1_phoneme']} ({row['top_1_percentage']:.0%}) \n"
    text += f"{row['top_2_phoneme']} ({row['top_2_percentage']:.0%}) \n"
    return text


def get_phonemes_figure(data: pd.DataFrame, color_by: str) -> go.Figure:
    """
    Generates a Plotly figure with phonemes plotted as filled shapes with labels and hover text.
    The color of the shapes and labels can be set to either phonemes or phoneme families.

    Parameters:
    - data: a Pandas DataFrame with the phonemes data
    - color_by: either 'Phonemes' or 'Phonemes Families', specifying how the shapes and labels
                should be colored

    Returns:
    - A Plotly figure with the plotted phonemes.
    """
    fig = go.Figure()

    # Determine the color function, text function, and name function based on the 'color_by' parameter
    if color_by == 'Phonemes':
        color_func = phone_name_to_color
        txt_func = lambda x: x
        name_func = lambda x: x
    else:  # Phonemes Families
        color_func = phone_name_to_family_color
        txt_func = phone_txt_to_family_txt
        name_func = phone_name_to_family_name

    # Iterate over the rows in the data
    for i, row in data.iterrows():
        unit = int(i)
        x = [float(x) for x in row['X'].split(",")]
        y = [float(y) for y in row['Y'].split(",")]
        phoneme = row['top_0_phoneme']
        text = row_to_text(row)

        # Add a trace with the phoneme shape and hover text
        fig.add_trace(go.Scatter(
            x=x, y=y,
            fill='toself',
            fillcolor=color_func(phoneme),
            hoveron='fills',  # select where hover is active
            line_color='black',
            text=txt_func(text),
            name=unit,
            mode='lines',
            hoverinfo='text'
        ))

        # Add a label for the phoneme
        fig.add_annotation(
            x=np.mean(x),
            y=np.mean(y),
            text=name_func(phoneme),
            showarrow=False,
            font=dict(
                color="#ffffff"
            ),
            align="center",
            bordercolor="#ffffff",
            borderwidth=2,
            borderpad=2,
            bgcolor=color_func(phoneme),
            opacity=0.8
        )

    # Update the layout of the figure
    fig.update_layout(
        autosize=False,
        width=600,
        height=600
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=go.layout.Margin(l=0, r=0, b=0, t=0))
    return fig


def get_tree_map(dense_model: str, vocab: str, unit: str) -> go.Figure:
    """
    Generates a Plotly figure with a treemap visualization of the clusters of a given unit and vocabulary.

    Parameters:
    - dense_model: the name of the dense model
    - vocab: the vocabulary size
    - unit: the unit for which the clusters should be plotted

    Returns:
    - A Plotly figure with the treemap visualization of the clusters.
    """
    # Read the cluster data from a CSV file
    data = pd.read_csv(f"../assets/dimension_reduction/{dense_model}_clusters.csv", index_col=0)

    # Filter the data to include only the clusters of the given unit and vocabulary
    data_filter = data[(data[f'target_units_{vocab}'].astype(int) == int(unit))]

    # Extract the names of the units in the clusters
    names = [f'{row["src_unit"]}({row["src_vocab"]})' for _, row in data_filter.iterrows()]

    # Set the root name of the treemap to the name of the given unit
    root_name = f'{unit}({vocab})'

    # Set the parent of each unit to either the root name or an empty string
    parents = [root_name if x != root_name else "" for x in names]

    # Create the treemap figure
    fig = go.Figure(go.Treemap(
        labels=names,
        parents=parents,
        marker=dict(
            colors=[COLORS[int(vocab)] for vocab in data_filter['src_vocab']],
        ),
    ))

    # Update the layout of the figure
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    fig.update_layout(
        autosize=False,
        width=150,
        height=600,
    )
    fig.update_traces(root_color=COLORS[int(vocab)])

    return fig


def get_main_figure(dense_model: str, pca_tsne: str, units_options: List[str], click_unit: Optional[str] = None,
                    click_vocab: Optional[str] = None, limits: Optional[Dict[str, Any]] = None,
                    graph_dim: str = "2D", mark_unit: str = "") -> go.Figure:
    """
    Generates a Plotly figure with a scatter plot visualization of the units in a given dense model, with the option to
    filter the units by vocabulary size and to mark a specific unit.

    Parameters:
    - dense_model: the name of the dense model
    - pca_tsne: the dimensionality reduction technique used to generate the coordinates of the units
    - units_options: a list of vocabulary sizes to include in the plot
    - click_unit: the unit to mark in the plot (if provided)
    - click_vocab: the vocabulary size of the unit to mark (if provided)
    - limits: the limits of the x and y axes of the plot (if provided)
    - graph_dim: the dimensionality of the plot ("2D" or "3D")

    """
    centers_2d = pd.read_csv(f'../assets/dimension_reduction/{dense_model}_{pca_tsne}_{graph_dim}.csv')
    clusters = pd.read_csv(f"../assets/dimension_reduction/{dense_model}_clusters.csv", index_col=0)

    centers_2d = centers_2d.merge(clusters, on=['src_vocab', 'src_unit'])

    centers_2d.loc[:, 'marker_size'] = 7
    if click_vocab and click_unit:
        is_selected = centers_2d[f'target_units_{click_vocab}'].astype(int) == int(click_unit)
        centers_2d['marker_size'] += 7 * is_selected

    fig_2d = go.Figure()
    filter_data = []
    for size in units_options:
        centers_filter = centers_2d[centers_2d['src_vocab'].astype(int) == int(size)]
        filter_data.append(centers_filter)
        if mark_unit:
            units_df = centers_filter[centers_filter['src_unit'].astype(int) == int(mark_unit)]
            if len(units_df):
                unit_index = units_df.index[0]
                centers_filter.loc[unit_index, 'marker_size'] = 21

        options = dict(x=centers_filter['X'], y=centers_filter['Y'], mode='markers+text',
                       text=centers_filter['src_unit'],
                       hoverinfo='text', textposition="top center", name=f'{size}',
                       textfont=dict(
                           color=COLORS[int(size)]),
                       marker=dict(size=centers_filter['marker_size'],
                                   color=[COLORS[int(size)]] * len(centers_filter)))
        if graph_dim == "3D":
            fig_2d.add_trace(go.Scatter3d(z=centers_filter['Z'], **options))
        else:
            fig_2d.add_trace(go.Scatter(**options))

    if limits and 'xaxis.range[0]' in limits:
        fig_2d.update_xaxes(range=[limits['xaxis.range[0]'], limits['xaxis.range[1]']])
        fig_2d.update_yaxes(range=[limits['yaxis.range[0]'], limits['yaxis.range[1]']])
    fig_2d.update_traces(textposition='top center')

    fig_2d.update_layout(
        autosize=False,
        width=600,
        height=600,
    )
    fig_2d.update_layout(margin=go.layout.Margin(l=0, r=0, b=0, t=0))
    return fig_2d
