## Master thesis on Network Embedding learning in Financial networks. 
It has been written as a graduate work on Wirtschaftsinformatik (Business informatics) program Data Analysis specialisation in Technical University Berlin.

### Theme:

_Unsupervised node segmentation infinancial transaction networks_

### Abstract:
The relations between parties in the finance industry is commonly modelled by a complex financial network with different participants connected by edges representing transactions. For example, in retail banking participants are consumers and enterprises, in interbank networks – individuals and legal entities. Due to the sensitive nature of data and strong privacy regulations, such participants are almost always lacking characteristics for a comprehensive multivariate analysis of financial data. Fortunately, structural features remain available and facilitate the exploration of anonymized financial networks. Modern fast network embedding frameworks derive structural features based on nodes' local neighbourhoods within a network. They find a mapping function which converts network's node to a low-dimensional latent representation. However, the majority of the network embedding methods process only static networks, while real-world financial networks are dynamic.

The current thesis presents a concept for unsupervised node segmentation in financial networks. It exploits network embedding framework for structural feature learning from dynamic networks. The concept is implemented in the form of the data processing pipeline, which produces several outputs for one input data set. The result is the best-evaluated set of groups of users similar in the sense of defined structural and time features.

This work consists of a compilation of background study, the development of a concept, prototypical implementation, description of experiments on real-world data and evaluation of the concept on a dynamic financial network. The experiments demonstrated the capability of the pipeline to reveal new insights about the input network based on a combination of features.
