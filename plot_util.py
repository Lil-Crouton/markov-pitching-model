import numpy as np
import networkx as nx

def define_edges(player_rates, avg_rates, rate_diffs):
    dec = 0
    # Define edges between nodes that reflect state transition matrix
    elist = [(1,2),(1,3),(2,4),(2,5),(3,5),(3,6),(4,7),(4,8),(5,8),(5,9),(6,6),(6,9),\
            (6,13),(7,10),(7,14),(8,10),(8,11),(9,9),(9,11),(9,13),(10,12),(10,14),\
            (11,11),(11,12),(11,13),(12,12),(12,13),(12,14)]

    player_labels = [f"{int(round(element*100,dec))}%" for element in player_rates]
    avg_labels = [f"{int(round(element*100,dec))}%" for element in avg_rates]

    rate_labels = []
    for rate_diff in rate_diffs:
        if rate_diff > 0:
            rate_labels.append(f"+{int(round(rate_diff,dec))}%")
        else:
            rate_labels.append(f"{int(round(rate_diff,dec))}%")

    edges_with_diff_labels = [(elist[i][0], elist[i][1], {'label': rate_labels[i]}) for i in range(len(elist))]
    edges_with_rate_labels = [(elist[i][0], elist[i][1], {'label': player_labels[i]}) for i in range(len(elist))]

    return edges_with_diff_labels, edges_with_rate_labels, player_labels, avg_labels


def get_color_intensities(rate_diffs, player):
    color_intensities = []
    if player == 'average':
        for diff in rate_diffs:
            color_intensities.append(255)
    else:
        # Normalize rate_diffs to range between -1 and 1
        capped_diffs = np.clip(rate_diffs, -10, 10)
        normalized_diffs = 2 * (capped_diffs - np.min(capped_diffs)) / (np.max(capped_diffs) - np.min(capped_diffs)) - 1
        for diff in normalized_diffs:
            color_intensities.append(int(255*diff))
    
    return color_intensities

def create_graph(edges_with_labels):
    G = nx.DiGraph()
    # Add Edges
    G.add_edges_from(edges_with_labels)
    label_mapping = {1:'[0,0]',2:'[1,0]',3:'[0,1]',4:'[2,0]',5:'[1,1]',6:'[0,2]',\
                    7:'[3,0]',8:'[2,1]',9:'[1,2]',10:'[3,1]',11:'[2,2]',12:'[3,2]',\
                        13:'OUT',14:'WALK'}
    G = nx.relabel_nodes(G, label_mapping)
    
    return G

def define_node_locations(scale_factor):
    pos = {
        '[0,0]': (150, 600*scale_factor),
        '[0,1]': (180, 500*scale_factor),
        '[1,0]': (120, 500*scale_factor),
        '[1,1]': (150, 400*scale_factor),
        '[0,2]': (210, 400*scale_factor),
        '[2,0]': (90, 400*scale_factor),
        '[1,2]': (180, 300*scale_factor),
        '[2,1]': (120, 300*scale_factor),
        '[2,2]': (150, 200*scale_factor),
        '[3,0]': (60, 300*scale_factor),
        '[3,1]': (90, 200*scale_factor),
        '[3,2]': (120, 100*scale_factor),
        'OUT': (210, 0),
        'WALK': (60, 0)
    }
    return pos