import pandas as pd
import json
import data_util


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

if __name__ == '__main__':

    fig = main_function('League Average', diff_select=False)


