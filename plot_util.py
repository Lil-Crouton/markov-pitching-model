import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.offline as pyo

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

def create_edge_trace(x0,y0,x1,y1,color,width):
    return  go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        mode='lines',
        line=dict(color=color, width=width),
        hoverinfo='none'
    )

def plot_edges(G, pos, color_intensities, edge_annotations, player):
    edge_traces = []
    loop_edge_colors = []
    loop_text_colors = []
    good_idx = [0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,0,1,0,1,0] # 1 = good, 0 = bad

    for i, edge in enumerate(G.edges()):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if (good_idx[i] and color_intensities[i] > 0) or (not good_idx[i] and color_intensities[i] < 0):
            edge_color = f'rgba(255,{255-abs(color_intensities[i])},{255-abs(color_intensities[i])},1)'
            text_color = 'red'
        else:
            edge_color = f'rgba({255-abs(color_intensities[i])},{255-abs(color_intensities[i])},255,1)'
            text_color = 'blue'
        if player == 'average':
            edge_color = 'gray'
            text_color = 'black'
        edge_trace = create_edge_trace(x0,y0,x1,y1,edge_color,width=3)
        edge_traces.append(edge_trace)

        if edge[0] == edge[1]:
            x_pos = x0+23
            y_pos = y0+5
            loop_edge_colors.append(edge_color)
            loop_text_colors.append(text_color)
        else:
            x_pos = (x0+x1)/2
            y_pos = (y0+y1)/2
        # Edge annotations (midpoint for label placement)
        edge_annotations.append(
            dict(
                x=x_pos,
                y=y_pos,
                xref='x',
                yref='y',
                text = '<b><i>'+str(G.edges[edge]['label'])+'</b></i>',
                showarrow=False,
                font=dict(color=text_color, size=14, family='Arial'),
                bgcolor='white',
                borderpad=0
            )
        )
        #static edge trace
    edge_traces_single_color = []
    for i, edge in enumerate(G.edges()):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_color = 'rgba(0, 0, 0, 0.2)'  # Change the color here
        edge_trace_single_color = create_edge_trace(x0, y0, x1, y1, edge_color, width=3)
        edge_traces_single_color.append(edge_trace_single_color)

    return edge_traces, edge_traces_single_color, loop_edge_colors, loop_text_colors


def plot_nodes (G, pos, count_diff, player_counts):
    # Create node trace
    node_trace = go.Scatter(
        x=[],
        y=[],
        mode='markers+text',
        hoverinfo='text',
        hovertext=[],
        hoverlabel=dict(
            bgcolor='rgba(255, 206, 231, 1)',
            bordercolor='rgba(255, 66, 164, 1)',
            font_size=14,
            font_family='Arial',
            font_color='black'
        ),
        marker=dict(
            showscale=False,
            color=[],
            size=[],
            symbol='circle',
            line_width=0.5,
            line_color='black',
            opacity=1),
        text=[],
        textposition="middle center",
        textfont=dict(
            family="Arial",
            size=18,
            color="black"
        )
    )

    node_diameter = 90
    # Add nodes to node trace
    for i,node in enumerate(G.nodes()):
        x,y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        # Customize node size and color
        node_trace['marker']['color'] += tuple(['rgb(220,220,220,1)'])
        node_trace['marker']['size'] += tuple([node_diameter + 10*count_diff[node]])
        count_diff[node] = round(count_diff[node],2)
        if count_diff[node] > 0:
            node_trace['hovertext'] += tuple([f"{player_counts[node]} pitches<br>+{count_diff[node]}% relative to average"])
        else:
            node_trace['hovertext'] += tuple([f"{player_counts[node]} pitches <br>{count_diff[node]}% relative to average"])
        if node == 'OUT' or node == 'WALK':
            node_trace['text'] += tuple([str(node)])
        else:
            node_trace['text'] += tuple([str(node)[1]+'-'+str(node)[-2]])
    
    return node_trace


def plot_text(G, pos, player_inplay_rates, avg_inplay_rates, diff_select):
    dec = 0
    if diff_select:
        inplay_rates = [round((player_inplay_rates[i] - avg_inplay_rates[i])*100,dec) for i in range(len(player_inplay_rates))]
    else:
        inplay_rates = [round(player_inplay_rates[i]*100,dec) for i in range(len(player_inplay_rates))]
    inplay_rate_labels = []

    for inplay_rate in inplay_rates:
        if inplay_rate > 0 and diff_select:
            inplay_rate_labels.append(f"+{int(round(inplay_rate,dec))}%")
        else:
            inplay_rate_labels.append(f"{int(round(inplay_rate,dec))}%")

    inplay_rate_labels = {
        '[0,0]': inplay_rate_labels[0],
        '[1,0]': inplay_rate_labels[1],
        '[0,1]': inplay_rate_labels[2],
        '[2,0]': inplay_rate_labels[3],
        '[1,1]': inplay_rate_labels[4],
        '[0,2]': inplay_rate_labels[5],
        '[3,0]': inplay_rate_labels[6],
        '[2,1]': inplay_rate_labels[7],
        '[1,2]': inplay_rate_labels[8],
        '[3,1]': inplay_rate_labels[9],
        '[2,2]': inplay_rate_labels[10],
        '[3,2]': inplay_rate_labels[11],
        'WALK': "",
        'OUT': ""
    }
    # Create text trace
    text_trace = go.Scatter(
        x=[],
        y=[],
        mode='text',
        hoverinfo='none',
        text=[],
        textposition="bottom center",
        textfont=dict(
            family="Arial",
            size=11,
            color="black"
        )
    )
    for node in G.nodes():
        x,y = pos[node]
        text_trace['x'] += tuple([x])
        text_trace['y'] += tuple([y])
        if node == 'OUT' or node == 'WALK':
            text_trace['text'] += tuple([''])
        else:
            text_trace['text'] += tuple(['<br>'+'IN PLAY:'+'<b><br>'+str(inplay_rate_labels[node])+'</b>'])
    
    return text_trace

def plot_info(G, player_counts_df, player_rates, avg_counts_df, avg_rates, pct_player_counts, pct_avg_counts, player):

    total_plr_pitches = int(player_counts_df.iloc[:, 1:].sum().sum()) - 2 # subtract 3 for the 1s in the markov state transition matrix
    total_plr_strikes = int(player_counts_df.iloc[0]['[0-1]'] + player_counts_df.iloc[1]['[1-1]'] + player_counts_df.iloc[2]['[0-2]'] + player_counts_df.iloc[3]['[2-1]'] + \
                        player_counts_df.iloc[4]['[1-2]'] + player_counts_df.iloc[5]['OUT'] + player_counts_df.iloc[5]['[0-2]'] + player_counts_df.iloc[6]['[3-1]'] + player_counts_df.iloc[7]['[2-2]'] + \
                        player_counts_df.iloc[8]['OUT'] + player_counts_df.iloc[8]['[1-2]'] + player_counts_df.iloc[9]['[3-2]'] + player_counts_df.iloc[10]['OUT'] + player_counts_df.iloc[10]['[2-2]'] + \
                        player_counts_df.iloc[11]['OUT'] + player_counts_df.iloc[11]['[3-2]'])
    plr_strike_rate = round(total_plr_strikes/total_plr_pitches*100,1)

    total_avg_pitches = int(avg_counts_df.iloc[:, 1:].sum().sum())
    total_avg_strikes = int(avg_counts_df.iloc[0]['[0-1]'] + avg_counts_df.iloc[1]['[1-1]'] + avg_counts_df.iloc[2]['[0-2]'] + avg_counts_df.iloc[3]['[2-1]'] + \
                        avg_counts_df.iloc[4]['[1-2]'] + avg_counts_df.iloc[5]['OUT'] + avg_counts_df.iloc[5]['[0-2]'] + avg_counts_df.iloc[6]['[3-1]'] + avg_counts_df.iloc[7]['[2-2]'] + \
                        avg_counts_df.iloc[8]['OUT'] + avg_counts_df.iloc[8]['[1-2]'] + avg_counts_df.iloc[9]['[3-2]'] + avg_counts_df.iloc[10]['OUT'] + avg_counts_df.iloc[10]['[2-2]'] + \
                        avg_counts_df.iloc[11]['OUT'] + avg_counts_df.iloc[11]['[3-2]'])
    avg_strike_rate = round(total_avg_strikes/total_avg_pitches*100,1)

    total_plr_balls = int(player_counts_df.iloc[0]['[1-0]'] + player_counts_df.iloc[1]['[2-0]'] + player_counts_df.iloc[2]['[1-1]'] + player_counts_df.iloc[3]['[3-0]'] + \
                    player_counts_df.iloc[4]['[2-1]'] + player_counts_df.iloc[5]['[1-2]'] + player_counts_df.iloc[6]['WALK'] + player_counts_df.iloc[7]['[3-1]'] + \
                    player_counts_df.iloc[8]['[2-2]'] + player_counts_df.iloc[9]['WALK'] + player_counts_df.iloc[10]['[3-2]'] + player_counts_df.iloc[11]['WALK'])
    plr_ball_rate = round(total_plr_balls/total_plr_pitches*100,1)

    total_avg_balls = int(avg_counts_df.iloc[0]['[1-0]'] + avg_counts_df.iloc[1]['[2-0]'] + avg_counts_df.iloc[2]['[1-1]'] + avg_counts_df.iloc[3]['[3-0]'] + \
                    avg_counts_df.iloc[4]['[2-1]'] + avg_counts_df.iloc[5]['[1-2]'] + avg_counts_df.iloc[6]['WALK'] + avg_counts_df.iloc[7]['[3-1]'] + \
                    avg_counts_df.iloc[8]['[2-2]'] + avg_counts_df.iloc[9]['WALK'] + avg_counts_df.iloc[10]['[3-2]'] + avg_counts_df.iloc[11]['WALK'])
    avg_ball_rate = round(total_avg_balls/total_avg_pitches*100,1)

    plr_total_inplay = int(player_counts_df['PLAY'].sum())
    plr_inplay_rate = round(plr_total_inplay/total_plr_pitches*100,1)

    avg_total_inplay = int(avg_counts_df['PLAY'].sum())
    avg_inplay_rate = round(avg_total_inplay/total_avg_pitches*100,1)
    
    if player == 'League Average':
        line_1 = f"In 2023, pitchers in the MLB threw <b>{total_plr_pitches}</b> total pitches"
        line_2 = f"<br><b>{total_plr_strikes} Strikes</b>: Accounting for {plr_strike_rate}% of all pitches thrown"
        line_3 = f"<br><br><b>{total_plr_balls} Balls</b>: Accounting for {plr_ball_rate}% of all pitches thrown"
        line_4 = f"<br><br><br><b>{plr_total_inplay} In Play</b>: Accounting for {plr_inplay_rate}% of all pitches thrown"
    else:
        line_1 = f"In 2023, <b>{player}</b> threw <b>{total_plr_pitches}</b> total pitches"
        line_2 = f"<br><b>{total_plr_strikes} Strikes</b>: {plr_strike_rate}% of his total pitches (league avg: {avg_strike_rate}%)"
        line_3 = f"<br><br><b>{total_plr_balls} Balls</b>: {plr_ball_rate}% of his total pitches (league avg: {avg_ball_rate}%)"
        line_4 = f"<br><br><br><b>{plr_total_inplay} In Play</b>: {plr_inplay_rate}% of his total pitches (league avg: {avg_inplay_rate}%)"
    info_text = [line_1, line_2, line_3, line_4]

    info_trace = go.Scatter(
        x=[],
        y=[],
        mode='text',
        hoverinfo='none',
        text=[],
        textposition="bottom right", 
        textfont=dict(
            family="Arial",
            size=16,
            color="black"
        )
    )

    x = [10, 10, 10, 10]
    y = [1000, 1000, 1000, 1000]
    for i, line in enumerate(info_text):
        info_trace['x'] += tuple([x[i]])
        info_trace['y'] += tuple([y[i]])
        info_trace['text'] += tuple([line])
    
    return info_trace    


def add_traces(edge_traces, edge_traces_single_color, node_trace, text_trace, info_trace):
    traces = edge_traces
    traces.extend(edge_traces_single_color)
    node_trace_idx = len(traces)
    traces.extend([node_trace])
    traces.extend([text_trace])
    traces.extend([info_trace])
    # traces.extend([credits_text_trace])
    return traces, node_trace_idx

def plot_loops(pos,node_trace,loop_edge_colors):
    # Add loops to graph with arrow caps
    loops = []
    arrows = []
    two_strike_index = [5,8,12,13]
    for i in range(4):
        node_size = node_trace['marker']['size'][two_strike_index[i]]/10
        loop = dict(
            start_pt = [pos[f'[{i},2]'][0]+node_size+1, pos[f'[{i},2]'][1]+node_size+10],
            end_pt = [pos[f'[{i},2]'][0]+node_size+2, pos[f'[{i},2]'][1]+node_size-20],
            control_pt1 = [pos[f'[{i},2]'][0]+node_size+15, pos[f'[{i},2]'][1]+node_size+15],
            control_pt2 = [pos[f'[{i},2]'][0]+node_size+5, pos[f'[{i},2]'][1]+node_size-30]
        )
        loops.append(loop)
        
        arrow = dict(
            type="path",
            path=f"M {loop['end_pt'][0]-0.5},{loop['end_pt'][1]-1.5} L {loop['end_pt'][0]+1.5},{loop['end_pt'][1]+8.5} L {loop['end_pt'][0]+2},{loop['end_pt'][1]-5} Z",
            fillcolor=loop_edge_colors[i],
            line=dict(color=loop_edge_colors[i]),
            line_width=0
        )
        arrows.append(arrow)
        
        arrow = dict(
        type="path",
        path=f"M {loop['end_pt'][0]-0.5},{loop['end_pt'][1]-1.5} L {loop['end_pt'][0]+1.5},{loop['end_pt'][1]+8.5} L {loop['end_pt'][0]+2},{loop['end_pt'][1]-5} Z",
        fillcolor="rgba(0,0,0,0.2)",
        line=dict(color="rgba(0,0,0,0.2)"),
        line_width=0
        )
        arrows.append(arrow)
        
    shapes = []
    for i,loop in enumerate(loops):
        shapes.append(
            dict(
                type="path",
                path=f"M {loop['start_pt'][0]},{loop['start_pt'][1]} C {loop['control_pt1'][0]},{loop['control_pt1'][1]} {loop['control_pt2'][0]},{loop['control_pt2'][1]} {loop['end_pt'][0]},{loop['end_pt'][1]}",
                line_color=loop_edge_colors[i],
            )
        )
        shapes.append(
            dict(
                type="path",
                path=f"M {loop['start_pt'][0]},{loop['start_pt'][1]} C {loop['control_pt1'][0]},{loop['control_pt1'][1]} {loop['control_pt2'][0]},{loop['control_pt2'][1]} {loop['end_pt'][0]},{loop['end_pt'][1]}",
                line_color="rgba(0,0,0,0.2)"
            )
        )
    shapes.extend(arrows)
    return shapes



def create_figure(traces, edge_annotations, pos, loop_edge_colors, node_trace_idx):
    fig = go.Figure(data=traces,
                    layout=go.Layout(
                    width=900,
                    height=900,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0, pad=0),  # Update the margin values here
                    annotations=edge_annotations,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=5, fixedrange=True),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1, fixedrange=True)))
    shapes = plot_loops(pos, traces[node_trace_idx], loop_edge_colors)
    fig.update_layout(shapes=shapes)
    return fig