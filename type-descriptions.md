This document contains the descriptions of several common types referenced
throughout the in-code docstrings.

*Merge directive:* This describes which nodes of a network are to be merged and
in what order. It is a list of tuples of layer indices and partitions. Each
partition is a list of lists of indices to be merged. The merging is done in the
order in which the layer indices appears.

*Linkage list:* This arranges the linkage matrices for the network. It is a
list of tuples of layer indices and a list of linkage elements. The list of
linkage elements has one linkage element for each kind (pos, neg, etc.) of
neuron. Each linkage element is a tuple of 2 things. The first is a list of the
indices of neurons that have been clustered in the element. This list also acts
as a map from the index of the neurons in the cluster to the original neuron
index. The second is a the linkage matrix. If there is only one neuron of the
given kind, the linkage matrix is None.
