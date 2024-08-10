import pandas as pd

def get_avg_dfs(avg_counts_file, avg_rates_file):
    """
    Reads in dataframes for the average counts and rate information from .csv files
    """
    avg_counts_df = pd.read_csv(avg_counts_file)
    avg_rates_df = pd.read_csv(avg_rates_file)
    return avg_counts_df, avg_rates_df


def get_player_dfs(player_id):
    """
    Reads in dataframes for the player's counts and rate information from .csv files
    """
    player_counts_df = pd.read_csv(f'Dataframes/2023{player_id}_counts_df.csv')
    player_rates_df = pd.read_csv(f'Dataframes/2023{player_id}_rates_df.csv')
    return player_counts_df, player_rates_df

def get_counts(player_counts_df, avg_counts_df):
    """
    Computes information relatedthe number of times a player went from each count to the next and the difference to league average
    """
    player_counts = [player_counts_df.iloc[0]['[1-0]']+player_counts_df.iloc[0]['[0-1]']+player_counts_df.iloc[0]['PLAY'],
                     player_counts_df.iloc[1]['[2-0]']+player_counts_df.iloc[1]['[1-1]']+player_counts_df.iloc[1]['PLAY'],
                     player_counts_df.iloc[2]['[1-1]']+player_counts_df.iloc[2]['[0-2]']+player_counts_df.iloc[2]['PLAY'],
                     player_counts_df.iloc[3]['[3-0]']+player_counts_df.iloc[3]['[2-1]']+player_counts_df.iloc[3]['PLAY'],
                     player_counts_df.iloc[4]['[2-1]']+player_counts_df.iloc[4]['[1-2]']+player_counts_df.iloc[4]['PLAY'],
                     player_counts_df.iloc[5]['[0-2]']+player_counts_df.iloc[5]['[1-2]']+player_counts_df.iloc[5]['OUT']+player_counts_df.iloc[5]['PLAY'],
                     player_counts_df.iloc[6]['[3-1]']+player_counts_df.iloc[6]['WALK']+player_counts_df.iloc[6]['PLAY'],
                     player_counts_df.iloc[7]['[3-1]']+player_counts_df.iloc[7]['[2-2]']+player_counts_df.iloc[7]['PLAY'],
                     player_counts_df.iloc[8]['[1-2]']+player_counts_df.iloc[8]['[2-2]']+player_counts_df.iloc[8]['OUT']+player_counts_df.iloc[8]['PLAY'],
                     player_counts_df.iloc[9]['[3-2]']+player_counts_df.iloc[9]['WALK']+player_counts_df.iloc[9]['PLAY'],
                     player_counts_df.iloc[10]['[2-2]']+player_counts_df.iloc[10]['[3-2]']+player_counts_df.iloc[10]['OUT']+player_counts_df.iloc[10]['PLAY'],
                     player_counts_df.iloc[11]['[3-2]']+player_counts_df.iloc[11]['OUT']+player_counts_df.iloc[11]['WALK']+player_counts_df.iloc[11]['PLAY']]

    avg_counts = [avg_counts_df.iloc[0]['[1-0]']+avg_counts_df.iloc[0]['[0-1]']+avg_counts_df.iloc[0]['PLAY'],
                  avg_counts_df.iloc[1]['[2-0]']+avg_counts_df.iloc[1]['[1-1]']+avg_counts_df.iloc[1]['PLAY'],
                  avg_counts_df.iloc[2]['[1-1]']+avg_counts_df.iloc[2]['[0-2]']+avg_counts_df.iloc[2]['PLAY'],
                  avg_counts_df.iloc[3]['[3-0]']+avg_counts_df.iloc[3]['[2-1]']+avg_counts_df.iloc[3]['PLAY'],
                  avg_counts_df.iloc[4]['[2-1]']+avg_counts_df.iloc[4]['[1-2]']+avg_counts_df.iloc[4]['PLAY'],
                  avg_counts_df.iloc[5]['[0-2]']+avg_counts_df.iloc[5]['[1-2]']+avg_counts_df.iloc[5]['OUT']+avg_counts_df.iloc[5]['PLAY'],
                  avg_counts_df.iloc[6]['[3-1]']+avg_counts_df.iloc[6]['WALK']+avg_counts_df.iloc[6]['PLAY'],
                  avg_counts_df.iloc[7]['[3-1]']+avg_counts_df.iloc[7]['[2-2]']+avg_counts_df.iloc[7]['PLAY'],
                  avg_counts_df.iloc[8]['[1-2]']+avg_counts_df.iloc[8]['[2-2]']+avg_counts_df.iloc[8]['OUT']+avg_counts_df.iloc[8]['PLAY'],
                  avg_counts_df.iloc[9]['[3-2]']+avg_counts_df.iloc[9]['WALK']+avg_counts_df.iloc[9]['PLAY'],
                  avg_counts_df.iloc[10]['[2-2]']+avg_counts_df.iloc[10]['[3-2]']+avg_counts_df.iloc[10]['OUT']+avg_counts_df.iloc[10]['PLAY'],
                  avg_counts_df.iloc[11]['[3-2]']+avg_counts_df.iloc[11]['OUT']+avg_counts_df.iloc[11]['WALK']+avg_counts_df.iloc[11]['PLAY']]

    player_counts = [int(i) for i in player_counts]
    avg_counts = [int(i) for i in avg_counts]

    pct_avg_counts = [i/sum(avg_counts) for i in avg_counts]
    pct_player_counts = [i/sum(player_counts) for i in player_counts]
    count_diff = [(pct_player_counts[i]-pct_avg_counts[i])*100 for i in range(len(player_counts))]
    count_diff += [0, 0]
    player_counts += [0, 0]

    player_counts = {
        '[0,0]': player_counts[0],
        '[1,0]': player_counts[1],
        '[0,1]': player_counts[2],
        '[2,0]': player_counts[3],
        '[1,1]': player_counts[4],
        '[0,2]': player_counts[5],
        '[3,0]': player_counts[6],
        '[2,1]': player_counts[7],
        '[1,2]': player_counts[8],
        '[3,1]': player_counts[9],
        '[2,2]': player_counts[10],
        '[3,2]': player_counts[11],
        'WALK': player_counts[12],
        'OUT': player_counts[13]
    }
    count_diff = {
        '[0,0]': count_diff[0],
        '[1,0]': count_diff[1],
        '[0,1]': count_diff[2],
        '[2,0]': count_diff[3],
        '[1,1]': count_diff[4],
        '[0,2]': count_diff[5],
        '[3,0]': count_diff[6],
        '[2,1]': count_diff[7],
        '[1,2]': count_diff[8],
        '[3,1]': count_diff[9],
        '[2,2]': count_diff[10],
        '[3,2]': count_diff[11],
        'WALK': count_diff[12],
        'OUT': count_diff[13]
    }
    
    return player_counts, avg_counts, count_diff, pct_avg_counts, pct_player_counts


def get_rates(player_rates_df, avg_rates_df):
    """
    Computes the rate at which players move from one count to another and the difference to league average
    """
    dec = 0
    player_rates = [player_rates_df.iloc[0]['[1-0]'],player_rates_df.iloc[0]['[0-1]'],player_rates_df.iloc[1]['[2-0]'],player_rates_df.iloc[1]['[1-1]'],\
                    player_rates_df.iloc[2]['[1-1]'],player_rates_df.iloc[2]['[0-2]'],player_rates_df.iloc[3]['[3-0]'],player_rates_df.iloc[3]['[2-1]'],\
                    player_rates_df.iloc[4]['[2-1]'],player_rates_df.iloc[4]['[1-2]'],player_rates_df.iloc[5]['[0-2]'],player_rates_df.iloc[5]['[1-2]'],\
                    player_rates_df.iloc[5]['OUT'],player_rates_df.iloc[6]['[3-1]'],player_rates_df.iloc[6]['WALK'],player_rates_df.iloc[7]['[3-1]'],\
                    player_rates_df.iloc[7]['[2-2]'],player_rates_df.iloc[8]['[1-2]'],player_rates_df.iloc[8]['[2-2]'],player_rates_df.iloc[8]['OUT'],\
                    player_rates_df.iloc[9]['[3-2]'],player_rates_df.iloc[9]['WALK'],player_rates_df.iloc[10]['[2-2]'],player_rates_df.iloc[10]['[3-2]'],\
                    player_rates_df.iloc[10]['OUT'],player_rates_df.iloc[11]['[3-2]'],player_rates_df.iloc[11]['OUT'],player_rates_df.iloc[11]['WALK']]

    player_inplay_rates = [player_rates_df.iloc[0]['PLAY'],player_rates_df.iloc[1]['PLAY'],player_rates_df.iloc[2]['PLAY'],player_rates_df.iloc[3]['PLAY'],\
                           player_rates_df.iloc[4]['PLAY'],player_rates_df.iloc[5]['PLAY'],player_rates_df.iloc[6]['PLAY'],player_rates_df.iloc[7]['PLAY'],\
                           player_rates_df.iloc[8]['PLAY'],player_rates_df.iloc[9]['PLAY'],player_rates_df.iloc[10]['PLAY'],player_rates_df.iloc[11]['PLAY']]

    avg_rates = [avg_rates_df.iloc[0]['[1-0]'],avg_rates_df.iloc[0]['[0-1]'],avg_rates_df.iloc[1]['[2-0]'],avg_rates_df.iloc[1]['[1-1]'],\
                 avg_rates_df.iloc[2]['[1-1]'],avg_rates_df.iloc[2]['[0-2]'],avg_rates_df.iloc[3]['[3-0]'],avg_rates_df.iloc[3]['[2-1]'],\
                 avg_rates_df.iloc[4]['[2-1]'],avg_rates_df.iloc[4]['[1-2]'],avg_rates_df.iloc[5]['[0-2]'],avg_rates_df.iloc[5]['[1-2]'],\
                 avg_rates_df.iloc[5]['OUT'],avg_rates_df.iloc[6]['[3-1]'],avg_rates_df.iloc[6]['WALK'],avg_rates_df.iloc[7]['[3-1]'],\
                 avg_rates_df.iloc[7]['[2-2]'],avg_rates_df.iloc[8]['[1-2]'],avg_rates_df.iloc[8]['[2-2]'],avg_rates_df.iloc[8]['OUT'],\
                 avg_rates_df.iloc[9]['[3-2]'],avg_rates_df.iloc[9]['WALK'],avg_rates_df.iloc[10]['[2-2]'],avg_rates_df.iloc[10]['[3-2]'],\
                 avg_rates_df.iloc[10]['OUT'],avg_rates_df.iloc[11]['[3-2]'],avg_rates_df.iloc[11]['OUT'],avg_rates_df.iloc[11]['WALK']]

    avg_inplay_rates = [avg_rates_df.iloc[0]['PLAY'],avg_rates_df.iloc[1]['PLAY'],avg_rates_df.iloc[2]['PLAY'],avg_rates_df.iloc[3]['PLAY'],\
                        avg_rates_df.iloc[4]['PLAY'],avg_rates_df.iloc[5]['PLAY'],avg_rates_df.iloc[6]['PLAY'],avg_rates_df.iloc[7]['PLAY'],\
                        avg_rates_df.iloc[8]['PLAY'],avg_rates_df.iloc[9]['PLAY'],avg_rates_df.iloc[10]['PLAY'],avg_rates_df.iloc[11]['PLAY']]
    
    rate_diffs = [round((player_rates[i] - avg_rates[i])*100,dec) for i in range(len(player_rates))]

    return player_rates, player_inplay_rates, avg_rates, avg_inplay_rates, rate_diffs