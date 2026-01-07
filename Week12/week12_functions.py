import numpy as np
np.set_printoptions(precision=3) # only 3 decimals in print
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')
from tqdm import tqdm

plt.rc("axes", labelsize = 11)
plt.rc("xtick", labelsize = 10, top = True, direction="in")
plt.rc("ytick", labelsize = 10, right = True, direction="in")
plt.rc("axes", titlesize = 13)
plt.rc("legend", fontsize = 10, loc = "best")
plt.rc('animation', html='jshtml')

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import SumAggregation


def nice_plot_nodes(ax):
    ax.set(xticks=[], yticks=[])

def plot_graph(ax, graph, s=1000):
    xmin = float(torch.min(graph.pos[:,0]))
    xmax = float(torch.max(graph.pos[:,0]))
    ymin = float(torch.min(graph.pos[:,1]))
    ymax = float(torch.max(graph.pos[:,1]))
    ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1))

    # Plot the nodes:
    for idx, (pos, x) in enumerate(zip(graph.pos, graph.x)):
        text_offset = pos + s**0.5 * torch.tensor([0.01, 0.01])
        ax.scatter(*pos.T, c=f'C{int(x)}', s=s, edgecolors='black')
        ax.text(*pos.T, int(x), color='white', ha='center', va='center')
        ax.text(*text_offset.T, f'{idx}', color='black', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.2'))
    
    # Plot the edges:
    for idx, (i1,i2) in enumerate(graph.edge_index.T):
        ax.plot([graph.pos[i1,0], graph.pos[i2,0]], [graph.pos[i1,1], graph.pos[i2,1]], c='black', zorder=0)
    

def elements_for_random_graph(num_nodes):
    cutoff = 2.5

    box_size=10
    positions = []
    for i in range(num_nodes):
        new_position = torch.rand(1, 2) * box_size
        if len(positions) > 0:
            all_positions = torch.vstack(positions)
            while torch.any(torch.cdist(all_positions, new_position) < 0.75*cutoff) or \
            torch.all(torch.cdist(all_positions, new_position) > cutoff):
                new_position = torch.rand(1, 2) * box_size        
        positions.append(new_position)

    positions = torch.vstack(positions)

    edge_index = []
    for i in range(len(positions)):
        for j in range(len(positions)):
            edge_index.append([i, j])

    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.int64).reshape(2, -1)
    x = torch.tensor(list(range(num_nodes)),dtype=torch.float32)
    return edge_index, positions, x


def is_identical_graphs(graph1, graph2, atol=1e-5):
    # Collect all the eigenvalues for each graph in the list
    list_of_graphs = [graph1, graph2]
    eigenvalue_tensors = []
    for graph in list_of_graphs:
        adj_matrix = to_dense_adj(graph.edge_index)
        eigenvalues = torch.linalg.eigvalsh(adj_matrix)
        eigenvalues = torch.sort(eigenvalues).values
        eigenvalue_tensors.append(eigenvalues)
    
    eigenvalue_tensors = torch.vstack(eigenvalue_tensors)

    # Check if all the eigenvalues are the same
    return torch.allclose(eigenvalue_tensors[0], eigenvalue_tensors[1], atol=atol)


def keep_short_edges(edge_index, positions):
    edge_index_to_keep = []
    for edge in edge_index.T:
        source = positions[edge[0]]
        target = positions[edge[1]]
        if torch.linalg.norm(source - target) < 2.5:
            edge_index_to_keep.append(edge)
    edge_index_to_keep = torch.tensor(np.array(edge_index_to_keep).T)
    return edge_index_to_keep


def random_graph(num_nodes):
    edge_index, positions, x = elements_for_random_graph(num_nodes)
    edge_index = keep_short_edges(edge_index, positions)
    graph = Data(x=x, edge_index=edge_index, pos=positions)
    return graph


def random_graphs(num_graphs, num_nodes, patience=1000):
    graph0 = random_graph(num_nodes)
    list_of_graphs = [graph0]

    attempts = 0
    while len(list_of_graphs) < num_graphs and attempts < patience:
        proposed_graph = random_graph(num_nodes)
        
        # Check if the proposed graph's eigenvalues are identical to any in the list
        identical_graphs = False
        for graph in list_of_graphs:
            if is_identical_graphs(proposed_graph, graph):  # Uses sorted eigenvalues
                identical_graphs = True
                break  # No need to check further

        # Append the graph if it has unique eigenvalues
        if not identical_graphs:
            list_of_graphs.append(proposed_graph)
            attempts = 0  # Reset attempts on success
        else:
            attempts += 1  # Increment attempts if graph is not added

    return list_of_graphs