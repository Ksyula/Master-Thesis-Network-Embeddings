import networkx as nx
import plotly.graph_objs as go

G = nx.grid_2d_graph(30, 30)
pos = dict((n, n) for n in G.nodes())
nx.set_node_attributes(G, pos, 'pos')
G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

nx.set_node_attributes(G, 'a', 'TimeClass')
for node in G.node:
    if (G.degree[node] < 4): G.node[node]['TimeClass'] = 'b'


edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5, color= '#888'),
    hoverinfo='none',
    mode='lines')

for edge in G.edges():
    x0, y0 = G.node[edge[0]]['pos']
    x1, y1 = G.node[edge[1]]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
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
data = [edge_trace, node_trace]
layout = go.Layout(
                title='<br>Grid graph',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[dict(
                    text=" ",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002)],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))