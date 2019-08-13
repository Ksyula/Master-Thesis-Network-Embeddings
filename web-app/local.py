# -*- coding: utf-8 -*-

"""
The Local version of the app.
"""

import base64
import io
import os
import time
import sys
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from node2vec import Node2Vec
from itertools import zip_longest
import umap
from sklearn.decomposition import PCA
import hdbscan

# networks
import random_network
import disconnected_components
import disconnected_components_with_time
import connected_stars
import disconnected_stars
import star_graph
import grid_graph
import sklearn.cluster as cluster

def merge(a, b):
    return dict(a, **b)
def omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}
def getDegreeByNode(walk, G):
    Degree_list = []
    for node in walk:
        Degree_list = Degree_list + [str(G.degree[int(node)]) + '_']
    return (Degree_list)
def getTimeClassByNode(walk, G):
    TimeClass_list = []
    for node in walk:
        TimeClass_list = TimeClass_list + [G.nodes[int(node)]['TimeClass']]
    return (TimeClass_list)
def input_field(title, state_id, state_value, state_max, state_min):
    """Takes as parameter the title, state, default value and range of an input field, and output a Div object with the given specifications."""
    return html.Div([
        html.P(title,
               style={
                   'display': 'inline-block',
                   'verticalAlign': 'mid',
                   'marginRight': '5px',
                   'margin-bottom': '0px',
                   'margin-top': '0px'}),
        html.Div([
            dcc.Input(
                id=state_id,
                type='number',
                value=state_value,
                max=state_max,
                min=state_min,
                size=7)],
            style={
                'display': 'inline-block',
                'margin-top': '0px',
                'margin-bottom': '0px'}
        )])
def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={'margin': '0px 5px 30px 0px'},
        children=[
            f"{name}:",
            html.Div(style={'margin-left': '5px'}, children=[
                dcc.Slider(id=f'slider-{short}',
                           min=min,
                           max=max,
                           marks=marks,
                           step=step,
                           value=val)
            ])
        ])

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}


# Generate the default scatter plot
tsne_df = pd.read_csv("data/tsne_3d.csv", index_col=0)

data = []

for idx, val in tsne_df.groupby(tsne_df.index):
    idx = int(idx)

    scatter = go.Scatter3d(
        name=f"Digit {idx}",
        x=val['x'],
        y=val['y'],
        z=val['z'],
        mode='markers',
        marker=dict(
            size=2.5,
            symbol='circle-dot'
        )
    )
    data.append(scatter)

# Layout for the t-SNE graph
tsne_layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)


local_layout = html.Div([
    # In-browser storage of global variables
    html.Div(
        id="data-df-and-message",
        style={'display': 'none'}
    ),

    html.Div(
        id="label-df-and-message",
        style={'display': 'none'}
    ),

    # Main app
    html.Div([
        html.H2(
            children='Network Embedding',
            id='title1',
            style={
                'float': 'left',
                'margin-top': '20px',
                'margin-bottom': '20px',
                'margin-left': '7px'
            }
        ),
        html.Img(
            src="http://www.sensafety.org/img/core-img/snet-logo.png",
            style={
                'height': '80px',
                'float': 'right',
                'margin': '0px 50px 0px 0px'
            }
        )
    ],
        className="row"
    ),

    html.Div([
        html.Div([
            # The network
            dcc.Graph(
                id='network-2d-plot',
                figure={
                    'data': random_network.data,
                    'layout': random_network.layout
                },
                style={
                    'height': '50vh',
                },
            )
        ],
            id="network-plot-div",
            className="eight columns"
        ),
        html.Div([
            html.H4(
                children='Node2vec',
                id='network_embedding_h4'
            ),
            dcc.Dropdown(
                id='dropdown-network',
                searchable=False,
                options=[

                    {'label': 'Random network', 'value': 'random_network'},
                    {'label': 'Star', 'value': 'star_graph'},
                    {'label': 'Disconnected stars', 'value': 'disconnected_stars'},
                    {'label': 'Connected stars', 'value': 'connected_stars'},
                    {'label': 'Disconnected components', 'value': 'disconnected_components'},
                    {'label': 'Disconnected components with time', 'value': 'disconnected_components_with_time'},
                    {'label': 'Grid', 'value': 'grid_graph'}
                    ],
                value='random_network'
            ),
            html.H6(
                children='Features:',
                id='features_h6',
                style={'margin': '15px 0px 0px 0px'}
            ),
            dcc.Checklist(
                id='features_node2vec',
                options=[
                    {'label': 'Location', 'value': 'Location'},
                    {'label': 'Degree', 'value': 'Degree'},
                    {'label': 'Time', 'value': 'Time'}
                ],
                values=['Location'],
                labelStyle={
                    'display': 'inline-block',
                    'margin-right': '7px',
                    'margin-left': '7px',
                    'font-weight': 300
                    },
                style={
                    'display': 'inline-block',
                    'margin-left': '7px'
                }
            ),
            NamedSlider(
                name="Dimensions",
                short="dimensions",
                min=5,
                max=128,
                step=None,
                val=10,
                marks={i: i for i in [5, 10, 32, 64, 128]}
            ),
            NamedSlider(
                name="Walk length",
                short="walk_length",
                min=6,
                max=14,
                step=None,
                val=8,
                marks={i: i for i in [6, 8, 10, 12, 14]}
            ),
            NamedSlider(
                name="Number of walks per node",
                short="num_walks",
                min=3,
                max=6,
                step=None,
                val=4,
                marks={i: i for i in [3, 4, 5, 6]}
            ),
            html.Button(
                children='Generate embeddings',
                id='network-embedding-generation-button',
                n_clicks=0
            ),
            html.Div(id='output-state')
        ],
            className="four columns",
            style={
                'padding': 20,
                'margin': 5,
                'borderRadius': 5,
                'border': 'thin lightgrey solid',

                # Remove possibility to select the text for better UX
                'user-select': 'none',
                '-moz-user-select': 'none',
                '-webkit-user-select': 'none',
                '-ms-user-select': 'none'
            }
        )
    ],
        className="row"
    ),

    html.Div([
        html.H2(
            children='Dimensionality Reduction',
            id='title2',
            style={
                'float': 'left',
                'margin-top': '20px',
                'margin-bottom': '20px',
                'margin-left': '7px'
            }
        )
    ],
        className="row"
    ),
################## TSNE
    html.Div([
        html.Div([
            # Data about the graph
            html.Div(
                id="kl-divergence",
                style={'display': 'none'}
            ),

            html.Div(
                id="end-time",
                style={'display': 'none'}
            ),

            html.Div(
                id="error-message",
                style={'display': 'none'}
            ),

            # The graph
            dcc.Graph(
                id='tsne-3d-plot',
                figure={
                    'data': data,
                    'layout': tsne_layout
                },
                style={
                    'height': '60vh',
                },
            )
        ],
            id="tsne-plot-div",
            className="eight columns"
        ),

        html.Div([

            html.H4( "t-SNE Parameters",
                id='dim_reduction'
            ),

            input_field("Perplexity:", "perplexity-state", 20, 50, 5),

            input_field("Number of Iterations:", "n-iter-state", 400, 5000, 250),

            input_field("Learning Rate:", "lr-state", 10, 1000, 10),

            html.Button(
                id='tsne-train-button',
                n_clicks=0,
                children='Start Training t-SNE'
            ),

            dcc.Upload(
                id='upload-data',
                children=html.A('Upload your input data here.'),
                style={
                    'height': '45px',
                    'line-height': '45px',
                    'border-width': '1px',
                    'border-style': 'dashed',
                    'border-radius': '5px',
                    'text-align': 'center',
                    'margin-top': '5px',
                    'margin-bottom': '5 px'
                },
                multiple=False,
                max_size=-1
            ),

            dcc.Upload(
                id='upload-label',
                children=html.A('Upload your labels here.'),
                style={
                    'height': '45px',
                    'line-height': '45px',
                    'border-width': '1px',
                    'border-style': 'dashed',
                    'border-radius': '5px',
                    'text-align': 'center',
                    'margin-top': '5px',
                    'margin-bottom': '5px'
                },
                multiple=False,
                max_size=-1
            ),

            html.Div([
                html.P(id='upload-data-message',
                       style={
                           'margin-bottom': '0px'
                       }),

                html.P(id='upload-label-message',
                       style={
                           'margin-bottom': '0px'
                       }),

                html.Div(id='training-status-message',
                         style={
                             'margin-bottom': '0px',
                             'margin-top': '0px'
                         }),

                html.P(id='error-status-message')
            ],
                id='output-messages',
                style={
                    'margin-bottom': '2px',
                    'margin-top': '2px'
                }
            )
        ],
            className="four columns",
            style={
                'padding': 20,
                'margin': 5,
                'borderRadius': 5,
                'border': 'thin lightgrey solid',

                # Remove possibility to select the text for better UX
                'user-select': 'none',
                '-moz-user-select': 'none',
                '-webkit-user-select': 'none',
                '-ms-user-select': 'none'
            }
        )
    ],
        className="row"
    ),
################## UMAP
    html.Div([
        html.Div([
            # The graph
            dcc.Graph(
                id='umap-3d-plot',
                figure={
                    'data': data,
                    'layout': tsne_layout
                },
                style={
                    'height': '60vh',
                },
            )
        ],
            id="umap-plot-div",
            className="eight columns"
        ),

        html.Div([
            html.H4(
                'UMAP Parameters',
                id='umap_h4'
            ),
            input_field("# neighbors:", "n_neighbors", 20, 200, 2),      # controls how UMAP balances local versus global structure in the data
            NamedSlider(                                                 # controls how tightly UMAP is allowed to pack points together
                name="Minimum distance",
                short="min_dist",
                min=0.0,
                max=1.0,
                step=0.1,
                val=0.2,
                marks={i: i for i in [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]}
            ),
            # n_neighbors,min_dist, distance_metric
            dcc.Dropdown(
                id='distance_metric',                          # controls how distance is computed in the ambient space of the input dat
                searchable=False,
                options=[
                    # TODO: Generate more data
                    {'label': 'euclidean', 'value': 'euclidean'},
                    {'label': 'manhattan', 'value': 'manhattan'},
                    {'label': 'mahalanobis', 'value': 'mahalanobis'}
                ],
                value='euclidean',
                style = {
                        'margin-top': '15px'
                        }
            ),
            html.Button(
                id='umap-train-button',
                n_clicks=0,
                children='Start Training UMAP',
                style = {
                        'margin-top': '15px'
                        }
            ),
        ],
            className="four columns",
            style={
                'padding': 20,
                'margin': 5,
                'borderRadius': 5,
                'border': 'thin lightgrey solid',

                # Remove possibility to select the text for better UX
                'user-select': 'none',
                '-moz-user-select': 'none',
                '-webkit-user-select': 'none',
                '-ms-user-select': 'none'
            }
        )
    ],
        className="row"
    ),
################## PCA
    html.Div([
        html.Div([
            # The graph
            dcc.Graph(
                id='pca-3d-plot',
                figure={
                    'data': data,
                    'layout': tsne_layout
                },
                style={
                    'height': '60vh',
                },
            )
        ],
            id="pca-plot-div",
            className="eight columns"
        ),

        html.Div([
            html.H4(
                'PCA',
                id='pca_h4'
            ),
            html.Button(
                id='pca-train-button',
                n_clicks=0,
                children='Start Training PCA',
                style={
                    'margin-top': '15px'
                }
            ),
        ],
            className="four columns",
            style={
                'padding': 20,
                'margin': 5,
                'borderRadius': 5,
                'border': 'thin lightgrey solid',

                # Remove possibility to select the text for better UX
                'user-select': 'none',
                '-moz-user-select': 'none',
                '-webkit-user-select': 'none',
                '-ms-user-select': 'none'
            }
        )
    ],
        className="row"
    ),
######### Clustering

    html.Div([
        html.H2(
            children='Cluster Analysis',
            id='title3',
            style={
                'float': 'left',
                'margin-top': '20px',
                'margin-bottom': '20px',
                'margin-left': '7px'
            }
        )
    ],
        className="row"
    ),

    html.Div([
        html.Div([
            # The graph
            dcc.Graph(
                id='clustering-3d-plot',
                figure={
                    'data': data,
                    'layout': tsne_layout
                },
                style={
                    'height': '60vh',
                },
            )
        ],
            id="clustering-plot-div",
            className="eight columns"
        ),

        html.Div([
            html.H4(
                'Cluster analysis',
                id='clustering_h4'
            ),
            dcc.Dropdown(
                id='dropdown-clustering',
                searchable=False,
                options=[
                    {'label': 'k-means', 'value': 'k_means'},
                    {'label': 'HDBSCAN', 'value': 'hdbscan'}
                    ],
                value='k_means',
                style={'margin': '0px 0px 15px 0px'}
            ),

            dcc.Dropdown(
                id='dropdown-dimreduction',
                searchable=False,
                options=[
                    {'label': 't-SNE', 'value': 'tsne'},
                    {'label': 'UMAP', 'value': 'umap'},
                    {'label': 'PCA', 'value': 'pca'}
                ],
                value='umap',
                style={'margin': '0px 0px 15px 0px'}
            ),

            input_field("k = ", "k_parameter", 2, 10, 2),
            NamedSlider(
                name="Minimum cluster size:",
                short="min_cl_size",
                min=5,
                max=40,
                step=5,
                val=10,
                marks={i: i for i in [5, 10, 15, 20, 25, 30, 35, 40]}
            ),
            NamedSlider(
                name="Minimum number of samples:",
                short="min_num_samp",
                min=5,
                max=40,
                step=5,
                val=10,
                marks={i: i for i in [5, 10, 15, 20, 25, 30, 35, 40]}
            ),
            NamedSlider(
                name="Alpha",
                short="alpha",
                min=0.1,
                max=1.3,
                step=0.1,
                val=0.2,
                marks={i: i for i in [0.1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2, 1.3]}
            ),

            html.Button(
                id='cluster-analysis-button',
                n_clicks=0,
                children='Run cluster analysis',
                style={
                    'margin-top': '15px'
                }
            ),
        ],
            className="four columns",
            style={
                'padding': 20,
                'margin': 5,
                'borderRadius': 5,
                'border': 'thin lightgrey solid',

                # Remove possibility to select the text for better UX
                'user-select': 'none',
                '-moz-user-select': 'none',
                '-webkit-user-select': 'none',
                '-ms-user-select': 'none'
            }
        )
    ],
        className="row"
    ),
],
    className="container",
    style={
        'width': '90%',
        'max-width': 'none',
        'font-size': '1.5rem'
    }
)

#########################################################

def local_callbacks(app):
    @app.callback(Output('clustering-plot-div', 'children'),
                  [Input('cluster-analysis-button', 'n_clicks')],
                  [State('dropdown-clustering', 'value'),
                   State('dropdown-dimreduction', 'value'),
                   State('k_parameter', 'value'),
                   State('slider-min_cl_size', 'value'),
                   State('slider-min_num_samp', 'value'),
                   State('slider-alpha', 'value'),
                   State('n_neighbors', 'value'),
                   State('slider-min_dist', 'value'),
                   State('distance_metric', 'value'),
                   State('perplexity-state', 'value'),
                   State('n-iter-state', 'value'),
                   State('lr-state', 'value')])
    def cluster_analysis(n_clicks, clustering, dimreduction,
                         k_parameter,                               # k-means params
                         min_cl_size, min_num_samp, alpha,          # HDBSCAN params
                         n_neighbors, min_dist, metric,             # UMAP params
                         perplexity, n_iter, learning_rate        # t-SNE params
                         ):
        if n_clicks > 0:
            data_df = np.array(pd.read_csv("/Users/ksenia/Documents/Abschlussarbeit/Network_embedding/app/dash/data/output_embeddings.csv"))
            #### k-means
            if clustering == 'k_means':
                if dimreduction == 'umap':
                    umap_ = umap.UMAP(n_neighbors=n_neighbors,
                                      min_dist=min_dist,
                                      n_components=3,
                                      metric=metric,
                                      random_state=42)
                    embeddings = umap_.fit_transform(data_df)
                    kmeans_labels = pd.DataFrame(cluster.KMeans(n_clusters=k_parameter, random_state=42).fit_predict(embeddings), columns=['label'])

                    umap_data_df = pd.DataFrame(embeddings, columns=['x', 'y', 'z'])

                    combined_df = umap_data_df.join(kmeans_labels)
                elif dimreduction == 'tsne':
                    tsne = TSNE(n_components=3,
                                perplexity=perplexity,
                                learning_rate=learning_rate,
                                n_iter=n_iter)

                    embeddings = tsne.fit_transform(data_df)
                    kmeans_labels = pd.DataFrame(cluster.KMeans(n_clusters=k_parameter, random_state=42).fit_predict(embeddings), columns=['label'])

                    tsne_data_df = pd.DataFrame(embeddings, columns=['x', 'y', 'z'])

                    combined_df = tsne_data_df.join(kmeans_labels)

                elif dimreduction == 'pca':

                    pca = PCA(n_components=3, svd_solver='full')
                    embeddings = pca.fit_transform(data_df)
                    kmeans_labels = pd.DataFrame(cluster.KMeans(n_clusters=k_parameter, random_state=42).fit_predict(embeddings), columns=['label'])

                    pca_data_df = pd.DataFrame(embeddings, columns=['x', 'y', 'z'])
                    combined_df = pca_data_df.join(kmeans_labels)
            #### HDBSCAN
            elif clustering == 'hdbscan':
                if dimreduction == 'umap':
                    umap_ = umap.UMAP(n_neighbors=n_neighbors,
                                      min_dist=min_dist,
                                      n_components=3,
                                      metric=metric,
                                      random_state=42)
                    embeddings = umap_.fit_transform(data_df)

                    hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cl_size,
                                           min_samples=min_num_samp,
                                           alpha=alpha,
                                           metric='euclidean')
                    hdbscanoutput = hdbs.fit(embeddings)

                    hdbscan_labels = pd.DataFrame(hdbscanoutput.labels_, columns=['label'])

                    umap_data_df = pd.DataFrame(embeddings, columns=['x', 'y', 'z'])

                    combined_df = umap_data_df.join(hdbscan_labels)
                elif dimreduction == 'tsne':
                    tsne = TSNE(n_components=3,
                                perplexity=perplexity,
                                learning_rate=learning_rate,
                                n_iter=n_iter)

                    embeddings = tsne.fit_transform(data_df)

                    hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cl_size,
                                           min_samples=min_num_samp,
                                           alpha=alpha,
                                           metric='euclidean')
                    hdbscanoutput = hdbs.fit(embeddings)

                    hdbscan_labels = pd.DataFrame(hdbscanoutput.labels_, columns=['label'])

                    tsne_data_df = pd.DataFrame(embeddings, columns=['x', 'y', 'z'])

                    combined_df = tsne_data_df.join(hdbscan_labels)

                elif dimreduction == 'pca':

                    pca = PCA(n_components=3, svd_solver='full')
                    embeddings = pca.fit_transform(data_df)
                    hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cl_size,
                                           min_samples=min_num_samp,
                                           alpha=alpha,
                                           metric='euclidean')
                    hdbscanoutput = hdbs.fit(embeddings)

                    hdbscan_labels = pd.DataFrame(hdbscanoutput.labels_, columns=['label'])
                    pca_data_df = pd.DataFrame(embeddings, columns=['x', 'y', 'z'])
                    combined_df = pca_data_df.join(hdbscan_labels)

            data = []

            # Group by the values of the label
            for idx, val in combined_df.groupby('label'):
                scatter = go.Scatter3d(
                    name=idx,
                    x=val['x'],
                    y=val['y'],
                    z=val['z'],
                    mode='markers',
                    marker=dict(
                        size=2.5,
                        symbol='circle-dot'
                    )
                )
                data.append(scatter)

        return [
        # The clustered graph
        dcc.Graph(
            id='clustering-3d-plot',
            figure={
                'data': data,
                'layout': tsne_layout
            },
            style={
                'height': '60vh',
            },
            )
        ]

    @app.callback(Output('network-plot-div', 'children'),
                  [Input('dropdown-network', 'value')])
    def update_network(network):
        if network == 'random_network':
            return [
                # The graph
                dcc.Graph(
                    id='network-2d-plot',
                    figure={
                        'data': random_network.data,
                        'layout': random_network.layout
                    },
                    style={
                        'height': '50vh',
                    },
                )
            ]
        elif network == 'star_graph':
            return [
                # The graph
                dcc.Graph(
                    id='network-2d-plot',
                    figure={
                        'data': star_graph.data,
                        'layout': star_graph.layout
                    },
                    style={
                        'height': '50vh',
                    },
                )
            ]
        elif network == 'disconnected_stars':
            return [
                # The graph
                dcc.Graph(
                    id='network-2d-plot',
                    figure={
                        'data': disconnected_stars.data,
                        'layout': disconnected_stars.layout
                    },
                    style={
                        'height': '50vh',
                    },
                )
            ]
        elif network == 'connected_stars':
            return [
                # The graph
                dcc.Graph(
                    id='network-2d-plot',
                    figure={
                        'data': connected_stars.data,
                        'layout': connected_stars.layout
                    },
                    style={
                        'height': '50vh',
                    },
                )
            ]
        elif network == 'disconnected_components':
            return [
                # The graph
                dcc.Graph(
                    id='network-2d-plot',
                    figure={
                        'data': disconnected_components.data,
                        'layout': disconnected_components.layout
                    },
                    style={
                        'height': '50vh',
                    },
                )
            ]
        elif network == 'disconnected_components_with_time':
            return [
                # The graph
                dcc.Graph(
                    id='network-2d-plot',
                    figure={
                        'data': disconnected_components_with_time.data,
                        'layout': disconnected_components_with_time.layout
                    },
                    style={
                        'height': '50vh',
                    },
                )
            ]
        elif network == 'grid_graph':
            return [
                # The graph
                dcc.Graph(
                    id='network-2d-plot',
                    figure={
                        'data': grid_graph.data,
                        'layout': grid_graph.layout
                    },
                    style={
                        'height': '50vh',
                    },
                )
            ]

    # Network embedding Button Click --> Generate embeddings
    @app.callback(Output('output-state', 'children'),
                  [Input('network-embedding-generation-button', 'n_clicks')],
                  [State('dropdown-network','value'),
                   State('features_node2vec', 'values'),
                   State('slider-dimensions', 'value'),
                   State('slider-walk_length', 'value'),
                   State('slider-num_walks', 'value')
                   ])
    def output_walks(n_clicks, network, features, dimensions, walk_length, num_walks):

        if network == 'random_network':
            g = random_network.G
        elif network == 'disconnected_components':
            g = disconnected_components.G
        elif network == 'disconnected_components_with_time':
            g = disconnected_components_with_time.G
        elif network == 'connected_stars':
            g = connected_stars.G
        elif network == 'disconnected_stars':
            g = disconnected_stars.G
        elif network == 'star_graph':
            g = star_graph.G
        elif network == 'grid_graph':
            g = grid_graph.G

        if n_clicks > 0:

            node2vec = Node2Vec(g,
                                dimensions=dimensions,
                                walk_length=walk_length,  # How many nodes are in each random walk
                                num_walks=num_walks,  # Number of random walks to be generated from each node in the graph
                                workers=4)
            print("Original walks:\n", "\n\n".join(map(str, node2vec.walks[0:3])), file=sys.stderr)
            if features == ['Location', 'Degree', 'Time']:
                print('Location, Degree, Time', file=sys.stderr)
                degree_formatted_walks = [getDegreeByNode(walk, g) for walk in node2vec.walks]
                time_formatted_walks = [getTimeClassByNode(walk, g) for walk in node2vec.walks]
                formatted_walks = [[j for i in zip_longest(a, b, c, fillvalue=None) for j in i][:-1] for a, b, c in
                                   list(zip(node2vec.walks, degree_formatted_walks, time_formatted_walks))]
                print("location_Degree_Time_formatted_walks:\n", "\n\n".join(map(str, formatted_walks[0:3])), file=sys.stderr)
                node2vec.walks = formatted_walks
                model = node2vec.fit(window=10, min_count=1)
                nodes_str = [x for x in model.wv.vocab if str.isdigit(x)]
                print(nodes_str[0:5], file=sys.stderr)
                embeddings = np.array([model.wv[x] for x in nodes_str])

                nodes = [int(i) for i in nodes_str]
                labels = [g.nodes[node]['TimeClass'] for node in nodes]
                print(embeddings.shape, file=sys.stderr)

            elif features == ['Location', 'Degree']:
                print('Location, Degree', file=sys.stderr)

                degree_formatted_walks = [getDegreeByNode(walk, g) for walk in node2vec.walks]
                formatted_walks = [[j for i in zip_longest(a, b, fillvalue=None) for j in i][:-1] for a, b in list(zip(node2vec.walks, degree_formatted_walks))]
                print("Degree_location_formatted_walks:\n", "\n\n".join(map(str, formatted_walks[0:3])), file=sys.stderr)
                node2vec.walks = formatted_walks
                model = node2vec.fit(window=10, min_count=1)
                nodes_str = [x for x in model.wv.vocab if str.isdigit(x)]
                print(nodes_str[0:5], file=sys.stderr)
                embeddings = np.array([model.wv[x] for x in nodes_str])

                nodes = [int(i) for i in nodes_str]
                labels = [g.degree[node] for node in nodes]
                print(embeddings.shape, file=sys.stderr)

            elif features == ['Location', 'Time']:
                print('Location, Time', file=sys.stderr)
                time_formatted_walks = [getTimeClassByNode(walk, g) for walk in node2vec.walks]
                formatted_walks = [[j for i in zip_longest(a, b, fillvalue=None) for j in i][:-1] for a, b in list(zip(node2vec.walks, time_formatted_walks))]
                print("Time_location_formatted_walks:\n", "\n\n".join(map(str, formatted_walks[0:3])), file=sys.stderr)
                node2vec.walks = formatted_walks
                model = node2vec.fit(window=10, min_count=1)
                nodes_str = [x for x in model.wv.vocab if str.isdigit(x)]
                print(nodes_str[0:5], file=sys.stderr)
                embeddings = np.array([model.wv[x] for x in nodes_str])

                nodes = [int(i) for i in nodes_str]
                labels = [g.nodes[node]['TimeClass'] for node in nodes]
                print("labels = ", labels)
                print(embeddings.shape, file=sys.stderr)

            else:
                print('Location', file=sys.stderr)
                model = node2vec.fit(window=10, min_count=1)
                nodes_str = [x for x in model.wv.vocab if str.isdigit(x)]
                print(nodes_str[0:5], file=sys.stderr)
                embeddings = np.array([model.wv[x] for x in nodes_str])
                labels = [int(i) for i in nodes_str]
                print(embeddings.shape, file=sys.stderr)
            # Writing the data:
            np.savetxt('/Users/ksenia/Documents/Abschlussarbeit/Network_embedding/app/dash/data/output_embeddings.csv', np.insert(embeddings, 0, np.arange(dimensions, dtype=int), axis=0), delimiter=',', fmt='%.4f')
            with open('/Users/ksenia/Documents/Abschlussarbeit/Network_embedding/app/dash/data/output_labels.csv', 'w') as f:
                f.write("0\n")
                for item in labels:
                    f.write("%s\n" % item)
            return u'''
            The Button has been pressed {} times,
            Features are {}, 
            Dimensions = {}, 
            walk_length = {},
            num_walks = {}.
            '''.format(n_clicks, features, dimensions, walk_length, num_walks)

    def parse_content(contents, filename):
        """This function parses the raw content and the file names, and returns the dataframe containing the data, as well
        as the message displaying whether it was successfully parsed or not."""
        if contents is None:
            return None, ""
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
                print('df = ', df)
            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                return None, 'The file uploaded is invalid.'
        except Exception as e:
            print(e)
            return None, 'There was an error processing this file.'
        return df, f'{filename} successfully processed.'

    # Uploaded data --> Hidden Data Div
    @app.callback(Output('data-df-and-message', 'children'),
                  [Input('upload-data', 'contents'),
                   Input('upload-data', 'filename')])
    def parse_data(contents, filename):
        data_df, message = parse_content(contents, filename)
        if data_df is None:
            return [None, message]
        elif data_df.shape[1] < 3:
            return [None, f'The dimensions of {filename} are invalid.']
        return [data_df.to_json(orient="split"), message]

    # Uploaded labels --> Hidden Label div
    @app.callback(Output('label-df-and-message', 'children'),
                  [Input('upload-label', 'contents'),
                   Input('upload-label', 'filename')])
    def parse_label(contents, filename):
        label_df, message = parse_content(contents, filename)
        if label_df is None:
            return [None, message]
        elif label_df.shape[1] != 1:
            return [None, f'The dimensions of {filename} are invalid.']
        return [label_df.to_json(orient="split"), message]

    # Hidden Data Div --> Display upload status message (Data)
    @app.callback(Output('upload-data-message', 'children'),
                  [Input('data-df-and-message', 'children')])
    def output_upload_status_data(data):
        return data[1]

    # Hidden Label Div --> Display upload status message (Labels)
    @app.callback(Output('upload-label-message', 'children'),
                  [Input('label-df-and-message', 'children')])
    def output_upload_status_label(data):
        return data[1]

    #t-SNE Button Click --> Update t-SNE graph with states
    @app.callback(Output('tsne-plot-div', 'children'),
                  [Input('tsne-train-button', 'n_clicks')],
                  [State('perplexity-state', 'value'),
                   State('n-iter-state', 'value'),
                   State('lr-state', 'value'),
                   State('data-df-and-message', 'children'),
                   State('label-df-and-message', 'children')
                   ])
    def update_tsne_graph(n_clicks, perplexity, n_iter, learning_rate, data_div, label_div):
        """Run the t-SNE algorithm upon clicking the training button"""

        error_message = None  # No error message at the beginning
        # Fix for startup POST
        if n_clicks <= 0 and (data_div is None or label_div is None):
            global data
            kl_divergence, end_time = None, None

        else:
            print("n_clicks___ = ", n_clicks)
            data_df = np.array(pd.read_csv("/Users/ksenia/Documents/Abschlussarbeit/Network_embedding/app/dash/data/output_embeddings.csv"))
            label_df = pd.read_csv("/Users/ksenia/Documents/Abschlussarbeit/Network_embedding/app/dash/data/output_labels.csv")

            # Fix the range of possible values
            if n_iter > 1000:
                n_iter = 1000
            elif n_iter < 250:
                n_iter = 250

            if perplexity > 50:
                perplexity = 50
            elif perplexity < 5:
                perplexity = 5

            if learning_rate > 1000:
                learning_rate = 1000
            elif learning_rate < 10:
                learning_rate = 10

            # Start timer
            start_time = time.time()

            # Then, apply t-SNE with the input parameters
            tsne = TSNE(n_components=3,
                        perplexity=perplexity,
                        learning_rate=learning_rate,
                        n_iter=n_iter)

            try:
                data_tsne = tsne.fit_transform(data_df)
                kl_divergence = tsne.kl_divergence_

                # Combine the reduced t-sne data with its label
                tsne_data_df = pd.DataFrame(data_tsne, columns=['x', 'y', 'z'])

                label_df.columns = ['label']

                combined_df = tsne_data_df.join(label_df)

                data = []

                # Group by the values of the label
                for idx, val in combined_df.groupby('label'):
                    scatter = go.Scatter3d(
                        name=idx,
                        x=val['x'],
                        y=val['y'],
                        z=val['z'],
                        mode='markers',
                        marker=dict(
                            size=2.5,
                            symbol='circle-dot'
                        )
                    )
                    data.append(scatter)

                end_time = time.time() - start_time

            # Catches Heroku server timeout
            except:
                error_message = "We were unable to train the t-SNE model due to timeout. Try to run it again, or to clone the repo and run the program locally."
                kl_divergence, end_time = None, None

        return [
            # Data about the graph
            html.Div([
                kl_divergence
            ],
                id="kl-divergence",
                style={'display': 'none'}
            ),

            html.Div([
                end_time
            ],
                id="end-time",
                style={'display': 'none'}
            ),

            html.Div([
                error_message
            ],
                id="error-message",
                style={'display': 'none'}
            ),

            # The graph
            dcc.Graph(
                id='tsne-3d-plot',
                figure={
                    'data': data,
                    'layout': tsne_layout
                },
                style={
                    'height': '60vh',
                },
            )
        ]

    #UMAP Button Click --> Update UMAP graph with states
    @app.callback(Output('umap-plot-div', 'children'),
                  [Input('umap-train-button', 'n_clicks')],
                  [State('n_neighbors', 'value'),
                   State('slider-min_dist', 'value'),
                   State('distance_metric', 'value'),
                   State('data-df-and-message', 'children'),
                   State('label-df-and-message', 'children')
                   ])
    def update_umap_graph(n_clicks, n_neighbors, min_dist, distance_metric, data_div, label_div):
        """Run the UMAP algorithm upon clicking the training button"""
        error_message = None
        # Fix for startup POST
        if n_clicks <= 0 and (data_div is None or label_div is None):
            global data
        else:
            print("n_clicks_umap = ", n_clicks)
            data_df = pd.read_csv("/Users/ksenia/Documents/Abschlussarbeit/Network_embedding/app/dash/data/output_embeddings.csv")
            label_df = pd.read_csv("/Users/ksenia/Documents/Abschlussarbeit/Network_embedding/app/dash/data/output_labels.csv")


            # Fix the range of possible values
            if n_neighbors > 200:
                n_neighbors = 200
            elif n_neighbors < 2:
                n_neighbors = 2

           # apply UMAP with the input parameters
            umap_ = umap.UMAP(n_neighbors = n_neighbors,
                              min_dist = min_dist,
                              n_components = 3,
                              metric = distance_metric)
            try:
                data_umap = umap_.fit_transform(data_df)

                # Combine the reduced umap data with its label
                umap_data_df = pd.DataFrame(data_umap, columns=['x', 'y', 'z'])

                label_df.columns = ['label']

                combined_df = umap_data_df.join(label_df)

                data = []

                # Group by the values of the label
                for idx, val in combined_df.groupby('label'):
                    scatter = go.Scatter3d(
                        name=idx,
                        x=val['x'],
                        y=val['y'],
                        z=val['z'],
                        mode='markers',
                        marker=dict(
                            size=2.5,
                            symbol='circle-dot'
                        )
                    )
                    data.append(scatter)

            # Catches Heroku server timeout
            except:
                error_message = "We were unable to train the UMAP model due to timeout. Try to run it again."

        return [
            # The graph
            dcc.Graph(
                id='umap-3d-plot',
                figure={
                    'data': data,
                    'layout': tsne_layout
                },
                style={
                    'height': '60vh',
                },
            )
        ]


    #PCA Button Click --> Update UMAP graph with states
    @app.callback(Output('pca-plot-div', 'children'),
                  [Input('pca-train-button', 'n_clicks')],
                  [State('data-df-and-message', 'children'),
                   State('label-df-and-message', 'children')
                   ])
    def update_pca_graph(n_clicks, data_div, label_div):
        """Run the PCA algorithm upon clicking the training button"""
        error_message = None
        # Fix for startup POST
        if n_clicks <= 0 and (data_div is None or label_div is None):
            global data
        else:
            print("n_clicks_pca = ", n_clicks)
            data_df = pd.read_csv("/Users/ksenia/Documents/Abschlussarbeit/Network_embedding/app/dash/data/output_embeddings.csv")
            label_df = pd.read_csv("/Users/ksenia/Documents/Abschlussarbeit/Network_embedding/app/dash/data/output_labels.csv")

           # apply PCA with the input parameters
            pca = PCA(n_components = 3, svd_solver='full')

            try:
                data_pca = pca.fit_transform(data_df)

                # Combine the reduced umap data with its label
                pca_data_df = pd.DataFrame(data_pca, columns=['x', 'y', 'z'])

                label_df.columns = ['label']

                combined_df = pca_data_df.join(label_df)

                data = []

                # Group by the values of the label
                for idx, val in combined_df.groupby('label'):
                    scatter = go.Scatter3d(
                        name=idx,
                        x=val['x'],
                        y=val['y'],
                        z=val['z'],
                        mode='markers',
                        marker=dict(
                            size=2.5,
                            symbol='circle-dot'
                        )
                    )
                    data.append(scatter)

            # Catches Heroku server timeout
            except:
                error_message = "We were unable to train the UMAP model due to timeout. Try to run it again."

        return [
            # The graph
            dcc.Graph(
                id='pca-3d-plot',
                figure={
                    'data': data,
                    'layout': tsne_layout
                },
                style={
                    'height': '60vh',
                },
            )
        ]


    # Updated graph --> Training status message
    @app.callback(Output('training-status-message', 'children'),
                  [Input('end-time', 'children'),
                   Input('kl-divergence', 'children')])
    def update_training_info(end_time, kl_divergence):
        # If an error message was output during the training.

        if end_time is None or kl_divergence is None or end_time[0] is None or kl_divergence[0] is None:
            return None
        else:
            end_time = end_time[0]
            kl_divergence = kl_divergence[0]

            return [
                html.P(f"t-SNE trained in {end_time:.2f} seconds.",
                       style={'margin-bottom': '0px'}),
                html.P(f"Final KL-Divergence: {kl_divergence:.2f}",
                       style={'margin-bottom': '0px'})
            ]

    @app.callback(Output('error-status-message', 'children'),
                  [Input('error-message', 'children')])
    def show_error_message(error_message):
        if error_message is not None:
            return [
                html.P(error_message[0])
            ]

        else:
            return []
