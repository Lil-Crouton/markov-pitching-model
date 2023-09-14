from requests_html import HTMLSession
import pandas as pd
from datetime import datetime, timedelta
from pybaseball import playerid_reverse_lookup
import time

def get_player_pitchvalues(player_name, player_id, start_date, end_date):
    url = f'https://www.fangraphs.com/players/{player_name}/{player_id}/game-log?position=OF&season=&gds={start_date}&gde={end_date}&type=7'
    return url

def get_player_plate_discipline(player_name, player_id, start_date, end_date):
    url = f'https://www.fangraphs.com/players/{player_name}/{player_id}/game-log?position=OF&season=&gds={start_date}&gde={end_date}&type=8'
    return url

def get_player_advanced(player_name, player_id, start_date, end_date):
    url = f'https://www.fangraphs.com/players/{player_name}/{player_id}/game-log?position=OF&season=&gds={start_date}&gde={end_date}&type=2'
    return url

def get_game_logs(start_year, end_year):

    years = [str(i) for i in range(start_year,end_year)]
    columns = [0,3,5,6,8,9,10,32,60,101,102,103,104,105,108,111,114,117,120,123,\
            126,129,132,135,138,141,144,147,150,153,156]
    column_names = ['Date','Visiting Team','Visiting Team G#','Home Team', \
                    'Home Team G#','Away Score','Home Score','SO (Away)', \
                    'SO (Home)','SP ID (Away)','SP Name (Away)','SP ID (Home)', \
                    'SP Name (Home)','P1 ID (Away)','P2 ID (Away)','P3 ID (Away)', \
                    'P4 ID (Away)','P5 ID (Away)','P6 ID (Away)','P7 ID (Away)', \
                    'P8 ID (Away)','P9 ID (Away)','P1 ID (Home)','P2 ID (Home)', \
                    'P3 ID (Home)','P4 ID (Home)','P5 ID (Home)','P6 ID (Home)', \
                    'P7 ID (Home)','P8 ID (Home)','P9 ID (Home)']
    x = {columns[i]:column_names[i] for i in range(len(columns))}

    game_logs = pd.DataFrame(columns=column_names)

    for year in years:
        game_log_df = pd.read_csv('gl'+year+'.txt', header=None)
        game_log_df.drop(columns=game_log_df.columns.difference(columns),axis=1,inplace=True)
        game_log_df = game_log_df.rename(columns=x)

        game_logs = pd.concat([game_logs,game_log_df],ignore_index=True)
    game_logs.to_csv('game_logs.csv', index=False)


def get_start_date(end_date):
    # convert end_date to datetime object (it's of the form 'YYYYMMDD')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    # subtract 100 days from end date to get start date (assume end_date is datetime object)
    start_date = end_date - timedelta(days=100)

    # if start_date is earlier in year than March 1, then take the difference and subtract that from September 30 of the previous year
    if start_date.month < 3 or start_date.year < end_date.year:
        day_diff = datetime(start_date.year, 3, 1) - start_date
        start_date = datetime(start_date.year-1, 9, 30) - day_diff

    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    return start_date, end_date

def get_player_info(retroid):
    player = playerid_reverse_lookup(retroid, key_type='retro')
    player.fg_id = player['key_fangraphs'].values[0]
    player.fg_name = player['name_first'].values[0] + '-' + player['name_last'].values[0]
    # if player.fg_name has any spaces, replace them with dashes
    player.fg_name = player.fg_name.replace(' ', '-')
    return player

def scrape_fangraphs(url, stats_type):
    session = HTMLSession()
    r = session.get(url)
    r.html.render(timeout=60)
    tables = r.html.find('div.table-scroll')
    selected_table = tables[1]

    df = pd.read_html(str(selected_table.html), header=0)[0]
    df = df.drop(columns=df.columns[df.columns.str.contains('divider')])
    df = df.iloc[0:1]

    if stats_type == 'advanced':
        column_map = {
        'BB%BB% - Walk Percentage (BB/PA)': 'BB%',
        'K%K% - Strikeout Percentage (SO/PA)': 'K%',
        'BB/KBB/K - Walk to Strikeout Ratio (BB/SO)': 'BB/K',
        'AVGAVG - Batting Average (H/AB)': 'AVG',
        'OBPOBP - On Base Percentage': 'OBP',
        'SLGSLG - Slugging Percentage': 'SLG',
        'OPSOPS - On Base + Slugging Percentage': 'OPS',
        'ISOISO - Isolated Power (SLG-AVG)': 'ISO',
        'SpdSpd - Speed Score (4 Component Version)': 'Spd',
        'BABIPBABIP - Batting Average on Balls in Play': 'BABIP',
        'wSBwSB - Stolen Bases and Caught Stealing runs above average': 'wSB',
        'wRCwRC - Runs Created based wOBA': 'wRC',
        'wRAAwRAA - Runs Above Average based on wOBA': 'wRAA',
        'wOBAwOBA - Weighted On Base Average (Linear Weights)': 'wOBA',
        'wRC+wRC+ - Runs per PA scaled where 100 is average; both league and park adjusted; based on wOBA': 'wRC+'
        }
        df = df.rename(columns=column_map)
        df.drop(columns=['Date','Team','Opp','BO','Pos','BB/K','OPS','ISO','BABIP','wSB','wRC'], inplace=True)

    return df

if __name__ == '__main__':
    need_game_logs = True
    column_names = ['BB%','K%','AVG','OBP','SLG','Spd','wRAA','wOBA','wRC+']

    # Get the game logs
    if need_game_logs:
        get_game_logs(2015,2023)

    # Add column_names to game_logs
    game_logs = pd.read_csv('game_logs.csv')
    game_logs = game_logs.reindex(columns=[*game_logs.columns.tolist(), *column_names])
    game_logs.to_csv('game_logs.csv', index=False)

    # Scrape fangraphs
    for index, row in game_logs.iterrows():
        start_date, end_date = get_start_date(str(row['Date']))
        print(start_date, end_date)
        # create a dataframe to store the lineup data
        lineup_df = pd.DataFrame(columns=column_names)
        if index == 1:
            break
        # time the loop
        start_time = time.time()
        for i in range(1,9):
            home_player = get_player_info(row[[f'P{i} ID (Home)']])
            print(home_player.fg_name)
            player_df = scrape_fangraphs(get_player_advanced(home_player.fg_name, home_player.fg_id, start_date, end_date), stats_type = 'advanced')
            # append the dataframe to lineup dataframe
            lineup_df = pd.concat([lineup_df, player_df], ignore_index=True)

        lineup_df.to_csv('lineup.csv', index=False)
        # Average all the stats in the lineup dataframe into one row
        lineup_df = lineup_df.mean(axis=0)
        # lineup stats to game_logs
        game_logs.loc[index, column_names] = lineup_df

    game_logs.to_csv('game_logs.csv', index=False)
    print("--- %s seconds ---" % (time.time() - start_time))