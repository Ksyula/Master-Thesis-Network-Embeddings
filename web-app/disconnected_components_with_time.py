import networkx as nx
import plotly.graph_objs as go
from math import floor

n_G = 200
n_H = 200
k_G = 4
k_H = 5
G = nx.watts_strogatz_graph(n_G, # The number of nodes
                     k_G,   # Each node is connected to k nearest neighbors in ring topology
                     0.3,   # The probability of rewiring each edge
                     seed=None) # Seed for random number generator (default=None)
H = nx.watts_strogatz_graph(n_H, # The number of nodes
                     k_H,   # Each node is connected to k nearest neighbors in ring topology
                     0.3,   # The probability of rewiring each edge
                     seed=None) # Seed for random number generator (default=None)
def mapping(x):
    return x+n_G
H = nx.relabel_nodes(H, mapping)

G = nx.compose(G, H)
pos = nx.spring_layout(G)

nx.set_node_attributes(G, 'NA', 'TimeClass')
nx.set_node_attributes(G, 'NA', 'Color')
lab = ["a", "b", "c", "d", "e", "f", "g", "h"]
color = ['1', '2', '3', '4', '5', '6', '7', '8']
for i in G.nodes:
    for node in G.neighbors(i):
        if G.nodes[node]['TimeClass'] == 'NA':
            G.nodes[node]['TimeClass'] = lab[floor(i/133)]
            G.nodes[node]['Color'] = color[floor(i / 133)]

pos = nx.spring_layout(G)
for key in pos:
    pos[key] = list(pos[key])
nx.set_node_attributes(G, pos, 'pos')

edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')
for edge in G.edges():
    x0, y0 = G.node[edge[0]]['pos']
    x1, y1 = G.node[edge[1]]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='Rainbow',
        reversescale=False,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Time',
            xanchor='left',
            titleside='right'
        ),
        line=dict(width=2)))
for node in G.nodes():
    x, y = G.node[node]['pos']
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    node_trace['marker']['color'] += G.nodes[node]['Color']
data = [edge_trace, node_trace]
layout = go.Layout(
                title='<br>Two components',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5, r=5, t=40),
                annotations=[dict(
                    text="400 nodes / time classes",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002)],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))