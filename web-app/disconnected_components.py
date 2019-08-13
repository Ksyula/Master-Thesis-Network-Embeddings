import networkx as nx
import plotly.graph_objs as go
import random

n_G = 200
# use a 2D Gaussian distribution of node positions with mean (0, 0) and standard deviation 2
p = {i: (random.gauss(-2, 1), random.gauss(-2, 1)) for i in range(n_G)}
G = nx.random_geometric_graph(n_G, 0.5, pos=p)
# Take the largest connected component
G= [G.subgraph(c) for c in nx.connected_components(G)][0]
nx.set_node_attributes(G, 'a', 'TimeClass')

n_H = 300
# use a 2D Gaussian distribution of node positions with mean (0, 0) and standard deviation 2
p = {i: (random.gauss(2, 1), random.gauss(2, 1)) for i in range(n_H)}
H = nx.random_geometric_graph(n_H, 0.5, pos=p)
# Take the largest connected component
H = [H.subgraph(c) for c in nx.connected_components(H)][0]
def mapping(x):
    return x+200
H = nx.relabel_nodes(H, mapping)
nx.set_node_attributes(H, 'b', 'TimeClass')

G = nx.compose(G, H)

# Store position as node attribute data for random_geometric_graph
# edge traces
edge_trace_G = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

for edge in G.edges():
    x0, y0 = G.node[edge[0]]['pos']
    x1, y1 = G.node[edge[1]]['pos']
    edge_trace_G['x'] += tuple([x0, x1, None])
    edge_trace_G['y'] += tuple([y0, y1, None])

edge_trace_H = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5, color='#000'),
    hoverinfo='none',
    mode='lines')

for edge in H.edges():
    x0, y0 = H.node[edge[0]]['pos']
    x1, y1 = H.node[edge[1]]['pos']
    edge_trace_H['x'] += tuple([x0, x1, None])
    edge_trace_H['y'] += tuple([y0, y1, None])

# node traces
node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='Reds',
        reversescale=False,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line=dict(width=2)))

for node in G.nodes():
    x, y = G.node[node]['pos']
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])

# Color node points by the number of connections.
for node, adjacencies in enumerate(G.adjacency()):
    node_trace['marker']['color'] += tuple([len(adjacencies[1])])
    node_info = 'id: ' + str(node) + ' \n degree: ' + str(len(adjacencies[1]))
    node_trace['text'] += tuple([node_info])

data = []

data = [edge_trace_G, edge_trace_H, node_trace]

layout = go.Layout(
    title='<br>Two components network',
    titlefont=dict(size=16),
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    annotations=[dict(
        text="500 nodes",
        showarrow=False,
        xref="paper", yref="paper",
        x=0.005, y=-0.002)],
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))