import pandas as pd
import json
import data_util
import plot_util


def main_function(player, diff_select):
    # Load all user configured data for the app
    config_data = json.load(open('config.json'))

    # Get the player id from the mapping in the config data
    player_id = config_data['players'][player]

    # Get average dataframes
    avg_counts_df, avg_rates_df = data_util.get_avg_dfs()

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
        G = plot.util.create_graph(edges_with_rate_labels)

    # Get node positions
    pos = plot_util.define_node_locations(1.5)



if __name__ == '__main__':

    fig = main_function('League Average', diff_select=False)


