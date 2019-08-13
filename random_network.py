import networkx as nx
import numpy as np
import plotly.graph_objs as go
###############################################
# Generate the default network plot
G = nx.random_geometric_graph(n=200, radius=0.125)
# Store position as node attribute data for random_geometric_graph
pos=nx.get_node_attributes(G, 'pos')

nx.set_node_attributes(G, 'a', 'TimeClass')
for node in np.random.choice(G.node, size=round(len(G)/2), replace=False, p=None):
    G.node[node]['TimeClass'] = 'b'

# Add edges as disconnected lines in a single trace and nodes as a scatter trace
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
network_data = []

data = [edge_trace, node_trace]

layout = go.Layout(
                title='<br>Random Network',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[dict(
                    text="200 nodes; p=1/8",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002)],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
###############################################