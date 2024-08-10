import pandas as pd
import json
import data_util
import plot_util
from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

def main_function(player, diff_select):
    # Load all user configured data for the app
    config_data = json.load(open('config.json'))

    # Get the player id from the mapping in the config data
    player_id = config_data['players'][0][player]

    # Get average dataframes
    avg_counts_df, avg_rates_df = data_util.get_avg_dfs('Dataframes/2023average_counts_df.csv','Dataframes/2023average_rates_df.csv')

    # Get count and rate dataframes for selected player
    player_counts_df, player_rates_df = data_util.get_player_dfs(player_id)

    # Get count differences and percentages for players and averages
    player_counts, avg_counts, count_diff, pct_avg_counts, pct_player_counts = data_util.get_counts(player_counts_df, avg_counts_df)

    # Get rate differences for players and averages
    player_rates, player_inplay_rates, avg_rates, avg_inplay_rates, rate_diffs = data_util.get_rates(player_rates_df, avg_rates_df)

    # Get edge information
    edges_with_diff_labels, edges_with_rate_labels, player_labels, avg_labels = plot_util.define_edges(player_rates, avg_rates, rate_diffs)

    # Define edge color intensities
    color_intensities = plot_util.get_color_intensities(rate_diffs, player_id)

    # Create Network Graph
    if diff_select:
        G = plot_util.create_graph(edges_with_diff_labels)
    else:
        G = plot_util.create_graph(edges_with_rate_labels)

    # Get node positions
    pos = plot_util.define_node_locations(1.5)

    # Get edge and loop traces
    edge_annotations = []
    edge_traces, edge_traces_single_color, loop_edge_colors, loop_text_colors = plot_util.plot_edges(G, pos, color_intensities, edge_annotations, player_id)

    # Get node trace
    node_trace = plot_util.plot_nodes(G, pos, count_diff, player_counts)

    # Get text trace
    text_trace = plot_util.plot_text(G, pos, player_inplay_rates, avg_inplay_rates, diff_select)

    # Get info trace to be shown next to plot
    info_trace = plot_util.plot_info(G, player_counts_df, player_rates, avg_counts_df, avg_rates, pct_player_counts, pct_avg_counts, player)

    # Append all traces together
    traces, node_trace_idx = plot_util.add_traces(edge_traces, edge_traces_single_color, node_trace, text_trace, info_trace)

    # Create figure
    fig = plot_util.create_figure(traces, edge_annotations, pos, loop_edge_colors, node_trace_idx)

if __name__ == '__main__':
    fig = main_function('League Average', diff_select=False)

    markov_text = """
            This chart is a markov chain representation of how a pitcher/batter move through a count with each ""node"" representing a different state of the count.  
            <span style='color: red;'>Red</span> lines indicate that the pitcher is better than average at inducing a "good" outcome.  
            <span style='color: blue;'>Blue</span> lines indicate that the pitcher is worse than average at inducing a "bad" outcome.  
            **Good Outcome**: Moving the count in the pitcher's favor, or inducing an out.  
            **Bad Outcome**: Moving the count in the batter's favor, or inducing a walk.  
            The size of each node is proportional to the number of pitches thrown in that count relative to the league average proportion.  
            In-play rates are shown in the middle of each node and are considered a **neutral outcome**.
            Note: Pitches from starter and relieving pitchers are included in league average calculation.
            """

    players = json.load(open('config.json'))['players'][0]

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([
        html.Div('Pitch Count Dynamics: A Markov Chain Model of 2023 Pitcher Performance', style={'font-family': 'Arial', 'font-size': '36px','padding-left':'20px'}),
        html.Div('Created by Pace Balster. Data obtained from RetroSheet (https://www.retrosheet.org/)', style={'font-family': 'Arial', 'font-size': '12px','padding-left':'20px'}),
        html.Div('Note: Only pitches thrown in starting appearances are counted for players. The following pitch types are counted: B, C, F, K, L, M, O, S, T, X (Reference Retrosheet game log documentation)', style={'font-family': 'Arial', 'font-size': '12px', 'font-style': 'italic','padding-left':'20px'}),
        html.Br(),
        html.Div([
            dcc.Dropdown(
            id='player-dropdown',
            options=[{'label': player, 'value': player} for player in players.keys()],
            value='Kyle Hendricks'
            )
        ], style={'width': '50%', 'display': 'inline-block','font-family': 'Arial', 'font-size': '16px','padding-left':'20px','margin':0}),
        html.Div([
            dbc.Switch(
                id='diff-select',
                label='Display Difference from League Average',
                value=False
            )
        ], style={'width': '100%', 'display': 'inline-block','font-family': 'Arial', 'font-size': '16px','padding-left':'20px','margin':0}),
        html.Div([
            dcc.Graph(
                id='markov-graph',
                figure=fig,
                config={"responsive": True}
            )
        ], style={'width':'100%', 'display': 'inline-block','padding':0,'margin':0}),
        html.Div([
            html.H2("How to Read this Chart"),
            html.Hr(style={"border-top": "2px solid #bbb", "margin-top": "0em", "margin-bottom": "0em"}),
            html.P([
                "This chart is a markov chain representation of how a pitcher/batter move through a count with each ",
                html.Span("node", style={'font-style': 'italic'}),
                " representing a different state of the count. ",
                html.Br(),
                html.Span("Red", style={'color': 'red','font-weight': 'bold', 'font-style': 'italic'}),
                " lines indicate that the pitcher induces ",
                html.Span("good", style={'font-style': 'italic'}),
                " outcomes at a higher rate and ",
                html.Span("bad", style={'font-style': 'italic'}),
                " outcomes at a lower rate compared to league average.",
                html.Br(),
                html.Span("Blue", style={'color': 'blue','font-weight': 'bold', 'font-style': 'italic'}),
                " lines indicate that the pitcher induces ",
                html.Span("good", style={'font-style': 'italic'}),
                " outcomes at a lower rate and ",
                html.Span("bad", style={'font-style': 'italic'}),
                " outcomes at a higher rate compared to league average.",
                html.Br(),
                html.Strong("Good Outcome:"),
                " Moving the count in the pitcher's favor, or inducing an out. ",
                html.Br(),
                html.Strong("Bad Outcome:"),
                " Moving the count in the batter's favor, or inducing a walk.",
                html.Br(),
                html.Span("Note: In-play rates are shown in the middle of each node and are considered a neutral outcome. Foul balls that do not change the count are considered a Bad outcome as they take the pitcher deeper into the count.", style={'font-style': 'italic'}),
                html.Br(),
                html.Strong("Node Sizes:"),
                "The size of each node is proportional to the number of pitches thrown in that count relative to the league average proportion."
            ])
        ], style={'font-family': 'Arial', 'font-size': '16px', 'text-align': 'left', 'padding-left':'20px'}),
    ])

    @app.callback(
        Output('markov-graph', 'figure'),
        [Input('player-dropdown', 'value'), Input('diff-select', 'value')]
    )
    def update_plot(player, diff_select):
        updated_fig = main_function(player, diff_select)
        return updated_fig
    
    server = app.server