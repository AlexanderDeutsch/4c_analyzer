import requests
import pandas as pd
import config
import datetime
from datetime import datetime
import json
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.box import ROUNDED
from datetime import datetime
import time
from rich.prompt import Prompt

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle NumPy and pandas-specific types"""
    def default(self, obj):
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)
    

#Function to grab the auth token
def grab_auth_token():

    #First we need to login to the API to get the auth token
    login_url = "https://api.4casters.io/user/login"
    payload = {
        "username": config.USERNAME,
        "password": config.PASSWORD 
    }
    headers = {}
    response = requests.request("POST", login_url, headers=headers, data=payload)

    #Save Auth token as a variable
    auth_token = response.json()['data']['user']['auth']
    return auth_token
#Function to grab the raw orderbook
def scrape_raw_orderbook(auth_token):
    url = "https://api.4casters.io/exchange/v2/getOrderbook?league=nba"

    payload = ""
    headers = {
        'Authorization':auth_token
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    game_data = response.json()
    game_data = game_data['data']['games']

    # Extract game details
    games_list = []
    for game in game_data:
        games_list.append({
            "Game ID": game["id"],
            "Matchup": f"{game['participants'][0]['longName']} vs {game['participants'][1]['longName']}",
            "Start Time": game["start"],
            "League": game["league"],
        })
    return game_data


# Add these helper functions at the top of your file

def calculate_best_odds(odds_series):
    """
    Find best odds considering American odds format.
    For negative odds, less negative is better (-110 is better than -120)
    For positive odds, higher is better (+120 is better than +110)
    """
    if odds_series.empty:
        return None
    if all(odds_series < 0):
        return odds_series.max()  # Least negative
    elif all(odds_series > 0):
        return odds_series.max()  # Highest positive
    else:
        # Mixed positive and negative odds - find best in each category
        neg_best = odds_series[odds_series < 0].max() if any(odds_series < 0) else None
        pos_best = odds_series[odds_series > 0].max() if any(odds_series > 0) else None
        
        # Return the overall best (challenging to compare +/- directly)
        if neg_best is None:
            return pos_best
        if pos_best is None:
            return neg_best
        # If both exist, return the one with better implied probability
        # (would require conversion to decimal odds for true comparison)
        return pos_best  # Simplified; typically positive odds offer better value
        
def get_odds_range(odds_series):
    """
    Returns the range of odds [best, worst] respecting American odds format
    Best odds come first, worst odds second
    """
    if odds_series.empty:
        return [None, None]
        
    if all(odds_series < 0):
        # For negative odds, best is least negative, worst is most negative
        return [odds_series.max(), odds_series.min()]
    elif all(odds_series > 0):
        # For positive odds, best is highest, worst is lowest
        return [odds_series.max(), odds_series.min()]
    else:
        # Mixed positive and negative - need to determine which is best
        # Typically positive odds are better than negative odds
        pos_odds = odds_series[odds_series > 0]
        neg_odds = odds_series[odds_series < 0]
        
        if not pos_odds.empty and not neg_odds.empty:
            # If we have both, determine which is better based on implied probability
            return [pos_odds.max(), neg_odds.min()]
        elif not pos_odds.empty:
            return [pos_odds.max(), pos_odds.min()]
        else:
            return [neg_odds.max(), neg_odds.min()]
        

# Add this new function at the top of your file, alongside your other helper functions
def get_odds_list(odds_series):
    """
    Returns a sorted list of all distinct odds, from most competitive to least competitive
    """
    if odds_series.empty:
        return []
        
    if all(odds_series < 0):
        # For negative odds, sort from least negative to most negative
        return sorted(odds_series.unique(), reverse=True)
    elif all(odds_series > 0):
        # For positive odds, sort from highest to lowest
        return sorted(odds_series.unique(), reverse=True)
    else:
        # Mixed positive and negative - sort each separately then combine
        pos_odds = sorted(odds_series[odds_series > 0].unique(), reverse=True)
        neg_odds = sorted(odds_series[odds_series < 0].unique(), reverse=True)
        
        # Return positive odds first (typically better value), then negative odds
        return pos_odds + neg_odds
    
def get_top_competitive_odds(odds_series, top_n=6):
    """
    Returns the top N most competitive distinct odds, regardless of liquidity
    """
    if odds_series.empty:
        return []
        
    if all(odds_series < 0):
        # For negative odds, get least negative (best) first
        return sorted(odds_series.unique(), reverse=True)[:top_n]
    elif all(odds_series > 0):
        # For positive odds, get highest (best) first
        return sorted(odds_series.unique(), reverse=True)[:top_n]
    else:
        # Mixed positive and negative
        pos_odds = sorted(odds_series[odds_series > 0].unique(), reverse=True)
        neg_odds = sorted(odds_series[odds_series < 0].unique(), reverse=True)
        
        # Combine and take top N
        combined = pos_odds + neg_odds
        return combined[:top_n]

 
def calculate_imbalance(numerator, denominator, default_value=0):
    """
    Calculate ratio between two values with proper handling of zero denominator
    """
    if denominator > 0:
        return numerator / denominator
    return default_value  # Instead of infinity, use a default value


#Function to verify there isn't some sort of error or issue with collection of orderbook data 
def check_analysis_consistency(ml_analysis, spread_analysis):
    """Check for consistency between moneyline and spread analyses"""
    
    # Skip check if either analysis lacks sufficient data
    if not ml_analysis['has_significant_data'] or not spread_analysis['has_significant_data']:
        return {
            'consistent': None,
            'reason': "Insufficient data for consistency check",
            'details': {}
        }
    
    # Compare favorite/underdog designations
    ml_favorite = ml_analysis['favorite_team']
    spread_favorite = spread_analysis['favorite_team']
    
    if ml_favorite == spread_favorite:
        return {
            'consistent': True,
            'reason': "Consistent favorite across markets",
            'details': {
                'favorite': ml_favorite
            }
        }
    else:
        return {
            'consistent': False,
            'reason': "Inconsistent favorite designations",
            'details': {
                'moneyline_favorite': ml_favorite,
                'spread_favorite': spread_favorite,
                'moneyline_favorite_odds': ml_analysis['favorite_best_odds'],
                'spread_favorite_odds': spread_analysis['favorite_odds_range']
            }
        }
    
    #Function to verify there isn't some sort of error or issue with collection of orderbook data 
def check_analysis_consistency(ml_analysis, spread_analysis):
    """Check for consistency between moneyline and spread analyses"""
    
    # Skip check if either analysis lacks sufficient data
    if not ml_analysis['has_significant_data'] or not spread_analysis['has_significant_data']:
        return {
            'consistent': None,
            'reason': "Insufficient data for consistency check",
            'details': {}
        }
    
    # Compare favorite/underdog designations
    ml_favorite = ml_analysis['favorite_team']
    spread_favorite = spread_analysis['favorite_team']
    
    if ml_favorite == spread_favorite:
        return {
            'consistent': True,
            'reason': "Consistent favorite across markets",
            'details': {
                'favorite': ml_favorite
            }
        }
    else:
        return {
            'consistent': False,
            'reason': "Inconsistent favorite designations",
            'details': {
                'moneyline_favorite': ml_favorite,
                'spread_favorite': spread_favorite,
                'moneyline_favorite_odds': ml_analysis['favorite_best_odds'],
                'spread_favorite_odds': spread_analysis['favorite_odds_range']
            }
        }

def identify_matched_liquidity(df, threshold=50):
    """Identify matched liquidity between opposite sides of the same market, processing each game separately"""
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Initialize the matched liquidity column
    df_copy['matched_liquidity'] = False
    
    # Process each game separately
    for game_id in df_copy['GameID'].unique():
        game_df = df_copy[df_copy['GameID'] == game_id]
        
        # Process spread markets
        spread_df = game_df[game_df['Market'] == 'Spread']
        
        # Group by spread value
        spreads = spread_df['Spread/Total'].unique()
        
        for spread_value in spreads:
            if pd.isna(spread_value) or spread_value == "N/A":
                continue
                
            # Get opposite spread
            opposite_spread = -1 * float(spread_value)
            
            # Get bets on both sides of this spread
            current_spread = spread_df[abs(spread_df['Spread/Total'] - float(spread_value)) < 0.5]
            opposite_spread_df = spread_df[abs(spread_df['Spread/Total'] - opposite_spread) < 0.5]
            
            # Skip if either side is empty
            if current_spread.empty or opposite_spread_df.empty:
                continue
            
            # Check for matching bet amounts within threshold
            for idx1, current_row in current_spread.iterrows():
                for idx2, opposite_row in opposite_spread_df.iterrows():
                    if abs(current_row['Bet Amount'] - opposite_row['Bet Amount']) <= threshold:
                        # Mark both rows as matched liquidity
                        df_copy.loc[idx1, 'matched_liquidity'] = True
                        df_copy.loc[idx2, 'matched_liquidity'] = True

    
        
        #Process moneyline markets 
        ml_df = game_df[game_df['Market'] == 'Moneyline']

        if not ml_df.empty:

            away_ml = ml_df[ml_df['Side'] == 'Away']
            home_ml = ml_df[ml_df['Side'] == 'Home']
            # We need to determine favorite and underdog based on odds - that will tell us which column to use for comparison
            if not away_ml.empty and not home_ml.empty:
                # Calculate average odds for each side
                away_avg_odds = away_ml['Odds'].mean()
                home_avg_odds = home_ml['Odds'].mean()
                
                #negative odds = favorite, positive odds = underdog
                if away_avg_odds < 0 and home_avg_odds > 0:
                    # Away is favorite, Home is underdog
                    favorite_ml = away_ml
                    underdog_ml = home_ml
                else:
                    # Home is favorite, Away is underdog
                    favorite_ml = home_ml
                    underdog_ml = away_ml
                
                # Check for matching values: Compare favorite's Bet Amount with underdog's Sum Untaken
                for idx_fav, fav_row in favorite_ml.iterrows():
                    for idx_dog, dog_row in underdog_ml.iterrows():
                        if abs(fav_row['Bet Amount'] - dog_row['Sum Untaken']) <= threshold:
                            # Mark both rows as matched liquidity
                            df_copy.loc[idx_fav, 'matched_liquidity'] = True
                            df_copy.loc[idx_dog, 'matched_liquidity'] = True
    
    return df_copy


def process_moneyline(ml_df):
    """
    Analyze moneyline orderbook data - focus on correct classification and filtering
    """
    # Create empty summary in case of empty dataframe
    if ml_df.empty:
        return {
            'favorite_side': None,
            'underdog_side': None,
            'favorite_team': None,
            'underdog_team': None,
            'favorite_best_odds': None,
            'underdog_best_odds': None,
            'underdog_untaken_sum': 0,
            'favorite_bet_amount': 0,
            'imbalance': 0,
            'has_significant_data': False
        }
    
    # Add matched liquidity identification
    ml_df_with_matches = identify_matched_liquidity(ml_df)
    # Use non-matched liquidity for sharp signal detection
    ml_df_sharp = ml_df_with_matches[~ml_df_with_matches['matched_liquidity']]
    matched_percentage = (ml_df_with_matches['matched_liquidity'].sum() / len(ml_df_with_matches)) * 100 if not ml_df_with_matches.empty else 0
    
    # Split by side
    away_ml = ml_df[ml_df['Side'] == 'Away']
    home_ml = ml_df[ml_df['Side'] == 'Home']
    
    # Get team names if available
    away_team = away_ml['Team'].iloc[0] if not away_ml.empty else "Away"
    home_team = home_ml['Team'].iloc[0] if not home_ml.empty else "Home"
    
    # Determine favorite and underdog based on average odds
    away_avg_odds = away_ml['Odds'].mean() if not away_ml.empty else 0
    home_avg_odds = home_ml['Odds'].mean() if not home_ml.empty else 0
    
    if away_avg_odds < 0 and home_avg_odds > 0:
        # Away is favorite, Home is underdog
        favorite_side = 'Away'
        underdog_side = 'Home'
        favorite_team = away_team
        underdog_team = home_team
        favorite = away_ml
        underdog = home_ml
    else:
        # Home is favorite (default), Away is underdog
        favorite_side = 'Home'
        underdog_side = 'Away'
        favorite_team = home_team
        underdog_team = away_team
        favorite = home_ml
        underdog = away_ml
    
    # Filter out insignificant wagers
    favorite_sig = favorite[favorite['Bet Amount'] >= 100]
    underdog_sig = underdog[underdog['Sum Untaken'] >= 100]
    
    has_significant_data = not favorite_sig.empty and not underdog_sig.empty
    
        # Find best odds using helper function
    favorite_best_odds = calculate_best_odds(favorite_sig['Odds']) if not favorite_sig.empty else None
    underdog_best_odds = calculate_best_odds(underdog_sig['Odds']) if not underdog_sig.empty else None
    
    # To this new approach:
    favorite_filtered = pd.DataFrame()
    underdog_filtered = pd.DataFrame()

    if not favorite_sig.empty:
        # Sort by odds competitiveness
        if all(favorite_sig['Odds'] < 0):
            # For negative odds, sort from least negative to most negative
            sorted_favorite = favorite_sig.sort_values('Odds', ascending=False)
        else:
            # For positive odds, sort from highest to lowest
            sorted_favorite = favorite_sig.sort_values('Odds', ascending=False)
        
        # Get up to 5 unique price points (might be fewer)
        unique_prices = sorted_favorite['Odds'].drop_duplicates().head(5)
        favorite_filtered = sorted_favorite[sorted_favorite['Odds'].isin(unique_prices)]

    if not underdog_sig.empty:
        # Sort by odds competitiveness
        if all(underdog_sig['Odds'] > 0):
            # For positive odds, sort from highest to lowest
            sorted_underdog = underdog_sig.sort_values('Odds', ascending=False)
        else:
            # For negative odds, sort from least negative to most negative
            sorted_underdog = underdog_sig.sort_values('Odds', ascending=False)
        
        # Get up to 5 unique price points
        unique_prices = sorted_underdog['Odds'].drop_duplicates().head(5)
        underdog_filtered = sorted_underdog[sorted_underdog['Odds'].isin(unique_prices)]
    # Calculate metrics for the filtered data
    underdog_untaken_sum = underdog_filtered['Sum Untaken'].sum() if not underdog_filtered.empty else 0
    favorite_bet_amount = favorite_filtered['Bet Amount'].sum() if not favorite_filtered.empty else 0
    imbalance = calculate_imbalance(underdog_untaken_sum, favorite_bet_amount)
    
    # Summary with all information but no signals
    summary = {
        'favorite_side': favorite_side,
        'underdog_side': underdog_side,
        'favorite_team': favorite_team,
        'underdog_team': underdog_team,
        'favorite_best_odds': favorite_best_odds,
        'underdog_best_odds': underdog_best_odds,
        'underdog_untaken_sum': underdog_untaken_sum,
        'favorite_bet_amount': favorite_bet_amount,
        'imbalance': imbalance,
        'has_significant_data': has_significant_data,
        'favorite_filtered_count': len(favorite_filtered) if not favorite_filtered.empty else 0,
        'underdog_filtered_count': len(underdog_filtered) if not underdog_filtered.empty else 0
    }
    
    # After filtering, add odds ranges and liquidity sums - CORRECTED
    if has_significant_data:
        # Add odds ranges with correct ordering
        if not favorite_filtered.empty:
            # Add odds ranges using helper function
            summary['favorite_filtered_odds_range'] = get_odds_range(favorite_filtered['Odds']) if not favorite_filtered.empty else [None, None]
            # Add complete odds list (new)
            summary['favorite_filtered_odds_list'] = get_odds_list(favorite_filtered['Odds']) if not favorite_filtered.empty else []
                        
        if not underdog_filtered.empty:
            # Add odds ranges using helper function
            summary['underdog_filtered_odds_range'] = get_odds_range(underdog_filtered['Odds']) if not underdog_filtered.empty else [None, None]
            # Add complete odds list (new)
            summary['underdog_filtered_odds_list'] = get_odds_list(underdog_filtered['Odds']) if not underdog_filtered.empty else []

        summary['favorite_raw_odds_list'] = get_top_competitive_odds(favorite_sig['Odds'], top_n=6) if not favorite_sig.empty else []
        summary['underdog_raw_odds_list'] = get_top_competitive_odds(underdog_sig['Odds'], top_n=6) if not underdog_sig.empty else []

    
        # Add complete liquidity information (kept separate)
        summary['favorite_untaken_sum'] = favorite_filtered['Sum Untaken'].sum() if not favorite_filtered.empty else 0
        summary['underdog_bet_amount'] = underdog_filtered['Bet Amount'].sum() if not underdog_filtered.empty else 0
        
        # Add the other side liquidity measures (not used for imbalance but informative)
        summary['favorite_all_untaken_sum'] = favorite_sig['Sum Untaken'].sum() if not favorite_sig.empty else 0
        summary['favorite_all_bet_amount'] = favorite_sig['Bet Amount'].sum() if not favorite_sig.empty else 0
        summary['underdog_all_untaken_sum'] = underdog_sig['Sum Untaken'].sum() if not underdog_sig.empty else 0
        summary['underdog_all_bet_amount'] = underdog_sig['Bet Amount'].sum() if not underdog_sig.empty else 0
    
    return summary

def process_spread(spread_df):
    """
    Analyze spread orderbook data - focus on correct classification and filtering
    """
    # Create empty summary in case of empty dataframe
    if spread_df.empty:
        return {
            'main_spread': None,
            'away_main_spread': None,
            'home_main_spread': None,
            'favorite_side': None,
            'underdog_side': None,
            'favorite_team': None,
            'underdog_team': None,
            'has_significant_data': False
        }
    
    # Convert spread to numeric if not already
    spread_df['Spread/Total'] = pd.to_numeric(spread_df['Spread/Total'], errors='coerce')
    
    # Split by side
    away_spread = spread_df[spread_df['Side'] == 'Away']
    home_spread = spread_df[spread_df['Side'] == 'Home']
    
    # Get team names if available
    away_team = away_spread['Team'].iloc[0] if not away_spread.empty else "Away"
    home_team = home_spread['Team'].iloc[0] if not home_spread.empty else "Home"
    
    # Filter out small wagers
    away_spread_sig = away_spread[away_spread['Sum Untaken'] + away_spread['Bet Amount'] >= 100]
    home_spread_sig = home_spread[home_spread['Sum Untaken'] + home_spread['Bet Amount'] >= 100]
    
    has_significant_data = not away_spread_sig.empty and not home_spread_sig.empty
    
    #Counts how many bets are at each spread value, the spread with the most is the "main spread"
    #We don't want to count the amt of open interest because sometimes there is massive open interest at unreasonably prices
    away_spread_counts = away_spread_sig.groupby('Spread/Total').size() if not away_spread_sig.empty else pd.Series()
    away_main_spread = away_spread_counts.idxmax() if not away_spread_counts.empty else None
    
    home_spread_counts = home_spread_sig.groupby('Spread/Total').size() if not home_spread_sig.empty else pd.Series()
    home_main_spread = home_spread_counts.idxmax() if not home_spread_counts.empty else None
    
    #Initialize empty dataframes
    # Replace with this new filtering approach:
    away_main_filtered = pd.DataFrame()
    home_main_filtered = pd.DataFrame()

    if not away_spread_sig.empty and away_main_spread is not None:
        # Get only the away bets at the main spread
        away_main_bets = away_spread_sig[away_spread_sig['Spread/Total'] == away_main_spread]
        
        if not away_main_bets.empty:
            # Sort by odds competitiveness
            if all(away_main_bets['Odds'] < 0):
                # For negative odds, sort from least negative to most negative
                sorted_away = away_main_bets.sort_values('Odds', ascending=False)
            else:
                # For positive odds, sort from highest to lowest
                sorted_away = away_main_bets.sort_values('Odds', ascending=False)
            
            # Get up to 5 unique price points
            unique_prices = sorted_away['Odds'].drop_duplicates().head(5)
            away_main_filtered = sorted_away[sorted_away['Odds'].isin(unique_prices)]

    if not home_spread_sig.empty and home_main_spread is not None:
        # Get only the home bets at the main spread
        home_main_bets = home_spread_sig[home_spread_sig['Spread/Total'] == home_main_spread]
        
        if not home_main_bets.empty:
            # Sort by odds competitiveness
            if all(home_main_bets['Odds'] < 0):
                # For negative odds, sort from least negative to most negative
                sorted_home = home_main_bets.sort_values('Odds', ascending=False)
            else:
                # For positive odds, sort from highest to lowest
                sorted_home = home_main_bets.sort_values('Odds', ascending=False)
            
            # Get up to 5 unique price points
            unique_prices = sorted_home['Odds'].drop_duplicates().head(5)
            home_main_filtered = sorted_home[sorted_home['Odds'].isin(unique_prices)]
    
    # Determine overall main spread
    main_spread = None
    
    if away_main_spread is not None and home_main_spread is not None:
        # Check if they're close to being opposites (allowing for small differences)
        if abs(abs(away_main_spread) - abs(home_main_spread)) < 1:
            # They're approximately opposites, use the one with more activity
            if away_spread_counts.get(away_main_spread, 0) > home_spread_counts.get(home_main_spread, 0):
                main_spread = away_main_spread
            else:
                main_spread = home_main_spread
        else:
            # They're not opposites - use the one with more activity
            if away_spread_counts.get(away_main_spread, 0) > home_spread_counts.get(home_main_spread, 0):
                main_spread = away_main_spread
            else:
                main_spread = home_main_spread
    elif away_main_spread is not None:
        main_spread = away_main_spread
    elif home_main_spread is not None:
        main_spread = home_main_spread
    
    # Get all bets at the main spread line, accounting for signs
    if main_spread is not None:
        away_main_bets = away_spread[abs(abs(away_spread['Spread/Total']) - abs(main_spread)) < 0.5]
        home_main_bets = home_spread[abs(abs(home_spread['Spread/Total']) - abs(main_spread)) < 0.5]
        main_spread_df = pd.concat([away_main_bets, home_main_bets])
    else:
        away_main_bets = pd.DataFrame()
        home_main_bets = pd.DataFrame()
        main_spread_df = pd.DataFrame()

    # Now use away_main_bets and home_main_bets directly
    away_main = away_main_bets
    home_main = home_main_bets
    #Don't use the odds number, we need the actual handicap number to determine the side that is favorite/dog
    away_handicap = away_main['Spread/Total'].median() if not away_main.empty else 0
    home_handicap = home_main['Spread/Total'].median() if not home_main.empty else 0
    print(away_handicap, home_handicap)
    
    if away_handicap < 0 and home_handicap > 0:
        # Away is favorite, Home is underdog
        favorite_side = 'Away'
        underdog_side = 'Home'
        favorite_team = away_team
        underdog_team = home_team
        favorite_spread = away_main_filtered
        underdog_spread = home_main_filtered
    else:
        # Home is favorite, Away is underdog
        favorite_side = 'Home'
        underdog_side = 'Away'
        favorite_team = home_team
        underdog_team = away_team
        favorite_spread = home_main_filtered
        underdog_spread = away_main_filtered
    
    # Calculate metrics
    underdog_untaken = underdog_spread['Sum Untaken'].sum() if not underdog_spread.empty else 0
    favorite_bet = favorite_spread['Bet Amount'].sum() if not favorite_spread.empty else 0
    imbalance = calculate_imbalance(underdog_untaken, favorite_bet, default_value=float('inf'))
    
    # Summary without signals
    summary = {
        'main_spread': main_spread,
        'away_main_spread': away_main_spread,
        'home_main_spread': home_main_spread,
        'favorite_side': favorite_side,
        'underdog_side': underdog_side,
        'favorite_team': favorite_team,
        'underdog_team': underdog_team,
        'underdog_untaken': underdog_untaken,
        'favorite_bet': favorite_bet,
        'imbalance': imbalance,
        'has_significant_data': has_significant_data
    }
    
    # Add details for debugging - CORRECTED odds ranges
    if has_significant_data:
        # Add odds ranges with correct ordering
        if not favorite_spread.empty:
            # Add odds ranges using helper function
            summary['favorite_filtered_odds_range'] = get_odds_range(favorite_spread['Odds']) if not favorite_spread.empty else [None, None]
            # Add complete odds list (new)
            summary['favorite_filtered_odds_list'] = get_odds_list(favorite_spread['Odds']) if not favorite_spread.empty else []
                        
        if not underdog_spread.empty:
            # Add odds ranges using helper function
            summary['underdog_filtered_odds_range'] = get_odds_range(underdog_spread['Odds']) if not underdog_spread.empty else [None, None]
            # Add complete odds list (new)
            summary['underdog_filtered_odds_list'] = get_odds_list(underdog_spread['Odds']) if not underdog_spread.empty else []
            # Add raw odds info (new) - using away_main_bets and home_main_bets (before filtering)
        if not away_main_bets.empty and favorite_side == 'Away':
            summary['favorite_raw_odds_list'] = get_top_competitive_odds(away_main_bets['Odds'], top_n=6)
        elif not home_main_bets.empty and favorite_side == 'Home':
            summary['favorite_raw_odds_list'] = get_top_competitive_odds(home_main_bets['Odds'], top_n=6)
        else:
            summary['favorite_raw_odds_list'] = []
            
        if not home_main_bets.empty and underdog_side == 'Home':
            summary['underdog_raw_odds_list'] = get_top_competitive_odds(home_main_bets['Odds'], top_n=6)
        elif not away_main_bets.empty and underdog_side == 'Away':
            summary['underdog_raw_odds_list'] = get_top_competitive_odds(away_main_bets['Odds'], top_n=6)
        else:
            summary['underdog_raw_odds_list'] = []

            
        # For the filtered data at main spread
        summary['favorite_untaken'] = favorite_spread['Sum Untaken'].sum() if not favorite_spread.empty else 0
        summary['favorite_bet'] = favorite_spread['Bet Amount'].sum() if not favorite_spread.empty else 0
        summary['underdog_untaken'] = underdog_spread['Sum Untaken'].sum() if not underdog_spread.empty else 0
        summary['underdog_bet'] = underdog_spread['Bet Amount'].sum() if not underdog_spread.empty else 0
        
        # For all significant data
        if not away_spread_sig.empty and not home_spread_sig.empty:
            if favorite_side == 'Away':
                summary['favorite_all_untaken'] = away_spread_sig['Sum Untaken'].sum()
                summary['favorite_all_bet'] = away_spread_sig['Bet Amount'].sum()
                summary['underdog_all_untaken'] = home_spread_sig['Sum Untaken'].sum()
                summary['underdog_all_bet'] = home_spread_sig['Bet Amount'].sum()
            else:
                summary['favorite_all_untaken'] = home_spread_sig['Sum Untaken'].sum()
                summary['favorite_all_bet'] = home_spread_sig['Bet Amount'].sum()
                summary['underdog_all_untaken'] = away_spread_sig['Sum Untaken'].sum()
                summary['underdog_all_bet'] = away_spread_sig['Bet Amount'].sum()
    
    return summary




def detect_sharp_signals(ml_analysis, spread_analysis):
    """Detect sharp signals based on moneyline and spread analysis"""
    signals = []
    
    # Moneyline signals
    if ml_analysis['has_significant_data']:
        if ml_analysis['imbalance'] > 1.5 and ml_analysis['underdog_untaken_sum'] > 1500:
            signals.append({
                'market': 'Moneyline',
                'signal': ml_analysis['favorite_team'],
                'strength': 'Strong' if ml_analysis['imbalance'] > 5 else 'Moderate',
                'reason': f"Large untaken amount on {ml_analysis['underdog_team']} (${ml_analysis['underdog_untaken_sum']:.2f})"
            })
        elif ml_analysis['imbalance'] < 0.5 and ml_analysis['favorite_bet_amount'] > 1500:
            signals.append({
                'market': 'Moneyline',
                'signal': ml_analysis['underdog_team'],
                'strength': 'Strong' if ml_analysis['imbalance'] < 0.2 else 'Moderate',
                'reason': f"Large bet amount on {ml_analysis['favorite_team']} (${ml_analysis['favorite_bet_amount']:.2f})"
            })
    
    # Spread signals
    if spread_analysis['has_significant_data'] and spread_analysis['main_spread'] is not None:
        if spread_analysis['imbalance'] > 1.5 and spread_analysis['underdog_untaken'] > 1500:
            signals.append({
                'market': 'Spread',
                'signal': spread_analysis['favorite_team'],
                'line': spread_analysis['main_spread'],
                'strength': 'Strong' if spread_analysis['imbalance'] > 5 else 'Moderate',
                'reason': f"Large untaken amount on {spread_analysis['underdog_team']} (${spread_analysis['underdog_untaken']:.2f})"
            })
        elif spread_analysis['imbalance'] < 0.5 and spread_analysis['favorite_bet'] > 1500:
            signals.append({
                'market': 'Spread',
                'signal': spread_analysis['underdog_team'],
                'line': spread_analysis['main_spread'],
                'strength': 'Strong' if spread_analysis['imbalance'] < 0.2 else 'Moderate',
                'reason': f"Large bet amount on {spread_analysis['favorite_team']} (${spread_analysis['favorite_bet']:.2f})"
            })
    
    return signals


def analyze_unmatched_liquidity(game_data, game_df):
    """
    Identify significant unmatched liquidity at competitive prices
    
    Parameters:
    game_data (dict): Parsed game data containing moneyline and spread information
    game_df (DataFrame): Raw game data with matched_liquidity column
    
    Returns:
    dict: Analysis of unmatched liquidity with significance ratings
    """
    ml_data = game_data['moneyline']
    spread_data = game_data['spread']
    results = {
        'moneyline_signals': [],
        'spread_signals': [],
        'total_signals': 0
    }
    
    # Get only unmatched liquidity rows
    unmatched_df = game_df[game_df['matched_liquidity'] == False]
    
    # Analyze moneyline unmatched liquidity
    if ml_data['has_significant_data']:
        # Split by market and side
        ml_unmatched = unmatched_df[unmatched_df['Market'] == 'Moneyline']
        fav_unmatched = ml_unmatched[ml_unmatched['Side'] == ml_data['favorite_side']]
        dog_unmatched = ml_unmatched[ml_unmatched['Side'] == ml_data['underdog_side']]
        
        # Check favorite side
        fav_raw_odds = ml_data.get('favorite_raw_odds_list', [])
        
        if fav_raw_odds and len(fav_raw_odds) > 0:
            # Consider top 6 competitive prices
            top_n = min(6, len(fav_raw_odds))
            competitive_prices = fav_raw_odds[:top_n]
            
            # Filter unmatched liquidity to only include these competitive prices
            #Use bet amount instead of sum untaken for favorites, we want high amts of to win, not risk
            fav_competitive_unmatched = fav_unmatched[fav_unmatched['Odds'].isin(competitive_prices)]
            fav_bet_amount = fav_competitive_unmatched['Bet Amount'].sum() if not fav_competitive_unmatched.empty else 0
            
            if fav_bet_amount > 1000:
                # Create detailed breakdown by price point
                price_breakdown = []
                for price in competitive_prices:
                    price_rows = fav_unmatched[fav_unmatched['Odds'] == price]
                    price_sum = price_rows['Bet Amount'].sum() if not price_rows.empty else 0
                    if price_sum > 0:
                        price_breakdown.append({
                            'price': price,
                            'amount': price_sum
                        })
                
                results['moneyline_signals'].append({
                    'side': 'Favorite',
                    'team': ml_data['favorite_team'],
                    'untaken_amount': fav_bet_amount,
                    'competitive_prices': competitive_prices,
                    'price_breakdown': price_breakdown,
                    'significance': 'High' if fav_bet_amount > 1000 else 'Medium'
                })
        
        # Check underdog side with same approach
        dog_raw_odds = ml_data.get('underdog_raw_odds_list', [])
        
        if dog_raw_odds and len(dog_raw_odds) > 0:
            top_n = min(6, len(dog_raw_odds))
            competitive_prices = dog_raw_odds[:top_n]
            
            dog_competitive_unmatched = dog_unmatched[dog_unmatched['Odds'].isin(competitive_prices)]
            dog_untaken_sum = dog_competitive_unmatched['Sum Untaken'].sum() if not dog_competitive_unmatched.empty else 0
            
            if dog_untaken_sum > 1000:
                price_breakdown = []
                for price in competitive_prices:
                    price_rows = dog_unmatched[dog_unmatched['Odds'] == price]
                    price_sum = price_rows['Sum Untaken'].sum() if not price_rows.empty else 0
                    if price_sum > 0:
                        price_breakdown.append({
                            'price': price,
                            'amount': price_sum
                        })
                
                results['moneyline_signals'].append({
                    'side': 'Underdog',
                    'team': ml_data['underdog_team'],
                    'untaken_amount': dog_untaken_sum,
                    'competitive_prices': competitive_prices,
                    'price_breakdown': price_breakdown,
                    'significance': 'High' if dog_untaken_sum > 1000 else 'Medium'
                })
    
    # Analyze spread unmatched liquidity with the same approach
    if spread_data['has_significant_data'] and spread_data.get('main_spread') is not None:
        # Split by market and side
        spread_unmatched = unmatched_df[unmatched_df['Market'] == 'Spread']
        
        # Only get spreads at the main spread line
        main_spread = spread_data.get('main_spread')
        spread_unmatched = spread_unmatched[abs(abs(spread_unmatched['Spread/Total']) - abs(main_spread)) < 0.5]
        
        fav_unmatched = spread_unmatched[spread_unmatched['Side'] == spread_data['favorite_side']]
        dog_unmatched = spread_unmatched[spread_unmatched['Side'] == spread_data['underdog_side']]
        
        # Favorite side
        fav_raw_odds = spread_data.get('favorite_raw_odds_list', [])
        
        if fav_raw_odds and len(fav_raw_odds) > 0:
            top_n = min(6, len(fav_raw_odds))
            competitive_prices = fav_raw_odds[:top_n]
            
            fav_competitive_unmatched = fav_unmatched[fav_unmatched['Odds'].isin(competitive_prices)]
            fav_untaken_sum = fav_competitive_unmatched['Sum Untaken'].sum() if not fav_competitive_unmatched.empty else 0
            
            if fav_untaken_sum > 1000:
                price_breakdown = []
                for price in competitive_prices:
                    price_rows = fav_unmatched[fav_unmatched['Odds'] == price]
                    price_sum = price_rows['Sum Untaken'].sum() if not price_rows.empty else 0
                    if price_sum > 0:
                        price_breakdown.append({
                            'price': price,
                            'amount': price_sum
                        })
                
                results['spread_signals'].append({
                    'side': 'Favorite',
                    'team': spread_data['favorite_team'],
                    'spread': main_spread,
                    'untaken_amount': fav_untaken_sum,
                    'competitive_prices': competitive_prices,
                    'price_breakdown': price_breakdown,
                    'significance': 'High' if fav_untaken_sum > 1000 else 'Medium'
                })
        
        # Underdog side
        dog_raw_odds = spread_data.get('underdog_raw_odds_list', [])
        
        if dog_raw_odds and len(dog_raw_odds) > 0:
            top_n = min(6, len(dog_raw_odds))
            competitive_prices = dog_raw_odds[:top_n]
            
            dog_competitive_unmatched = dog_unmatched[dog_unmatched['Odds'].isin(competitive_prices)]
            dog_untaken_sum = dog_competitive_unmatched['Sum Untaken'].sum() if not dog_competitive_unmatched.empty else 0
            
            if dog_untaken_sum > 1000:
                price_breakdown = []
                for price in competitive_prices:
                    price_rows = dog_unmatched[dog_unmatched['Odds'] == price]
                    price_sum = price_rows['Sum Untaken'].sum() if not price_rows.empty else 0
                    if price_sum > 0:
                        price_breakdown.append({
                            'price': price,
                            'amount': price_sum
                        })
                
                results['spread_signals'].append({
                    'side': 'Underdog',
                    'team': spread_data['underdog_team'],
                    'spread': main_spread,
                    'untaken_amount': dog_untaken_sum,
                    'competitive_prices': competitive_prices,
                    'price_breakdown': price_breakdown,
                    'significance': 'High' if dog_untaken_sum > 1000 else 'Medium'
                })
    
    # Calculate total signals
    results['total_signals'] = len(results['moneyline_signals']) + len(results['spread_signals'])
    
    return results


def detect_top_book_imbalance(game_data, game_df):
    """
    Detect significant imbalances at the top of the orderbook between corresponding sides
    """
    results = {
        'moneyline_signals': [],
        'spread_signals': [],
        'total_signals': 0,
        'moneyline_summary': {
            'has_data': False,
            'favorite_top_volume': 0,
            'favorite_breakdown': [],
            'underdog_top_volume': 0,
            'underdog_breakdown': []
        },
        'spread_summary': {
            'has_data': False,
            'favorite_top_volume': 0,
            'favorite_breakdown': [],
            'underdog_top_volume': 0,
            'underdog_breakdown': []
        }
    }
    
    # Analyze spread market
    spread_data = game_data['spread']
    
    if spread_data['has_significant_data'] and spread_data.get('main_spread') is not None:
        # Create a filtered dataframe just for spread market
        spread_df = game_df[game_df['Market'] == 'Spread'].copy()
        
        # Get favorite and underdog data
        fav_df = spread_df[spread_df['Side'] == spread_data['favorite_side']]
        dog_df = spread_df[spread_df['Side'] == spread_data['underdog_side']]
        
        # Get the top prices from each side
        fav_raw_odds = spread_data.get('favorite_raw_odds_list', [])
        dog_raw_odds = spread_data.get('underdog_raw_odds_list', [])
        
        if fav_raw_odds and dog_raw_odds and not fav_df.empty and not dog_df.empty:
            # Use top 3 price levels (or fewer if not available)
            top_n = min(3, len(fav_raw_odds), len(dog_raw_odds))
            
            # Calculate volume for top N price levels for favorite
            fav_top_volume = 0
            fav_top_breakdown = []
            
            for i in range(top_n):
                if i < len(fav_raw_odds):
                    price = fav_raw_odds[i]
                    # Get all bets at this price
                    price_bets = fav_df[fav_df['Odds'] == price]
                    
                    #Sum of bet amount
                    price_volume = price_bets['Bet Amount'].sum()
                    fav_top_volume += price_volume
                    
                    fav_top_breakdown.append({
                        'price': price,
                        'volume': price_volume
                    })
            
            # Calculate volume for top N price levels for underdog
            dog_top_volume = 0
            dog_top_breakdown = []
            
            for i in range(top_n):
                if i < len(dog_raw_odds):
                    price = dog_raw_odds[i]
                    # Get all bets at this price
                    price_bets = dog_df[dog_df['Odds'] == price]
                    
                    # Sum of bet amount 
                    price_volume = price_bets['Bet Amount'].sum()
                    dog_top_volume += price_volume
                    
                    dog_top_breakdown.append({
                        'price': price,
                        'volume': price_volume
                    })
                       # Store summary data regardless of imbalance
            results['spread_summary'] = {
                'has_data': True,
                'main_spread': spread_data['main_spread'],
                'favorite_team': spread_data['favorite_team'],
                'underdog_team': spread_data['underdog_team'],
                'favorite_top_volume': fav_top_volume,
                'favorite_breakdown': fav_top_breakdown,
                'underdog_top_volume': dog_top_volume,
                'underdog_breakdown': dog_top_breakdown
            }
            # Calculate imbalance ratio - larger side divided by smaller side
            if fav_top_volume > dog_top_volume and dog_top_volume > 0:
                larger_side = 'Favorite'
                larger_team = spread_data['favorite_team']
                smaller_team = spread_data['underdog_team']
                ratio = fav_top_volume / dog_top_volume
            elif dog_top_volume > fav_top_volume and fav_top_volume > 0:
                larger_side = 'Underdog'
                larger_team = spread_data['underdog_team']
                smaller_team = spread_data['favorite_team']
                ratio = dog_top_volume / fav_top_volume
            else:
                # Equal or one is zero
                larger_side = None
                ratio = 0
            
            # Signal if there's a significant imbalance AND the larger side has meaningful volume
            min_volume_threshold = 1000
            ratio_threshold = 1.5
            
            if larger_side and ratio > ratio_threshold and (fav_top_volume > min_volume_threshold or dog_top_volume > min_volume_threshold):
                results['spread_signals'].append({
                    'market': 'Spread',
                    'signal': smaller_team, #The side that has the most open interest is the side that the SHARP bettors are betting against, so the smaller team is the signal
                    'larger_side': larger_side,
                    'main_spread': spread_data['main_spread'],
                    'favorite_top_volume': fav_top_volume,
                    'favorite_breakdown': fav_top_breakdown,
                    'underdog_top_volume': dog_top_volume,
                    'underdog_breakdown': dog_top_breakdown,
                    'imbalance_ratio': ratio,
                    'significance': 'High' if ratio > 7 else 'Medium'
                })
    
    # Analyze moneyline market (no Spread/Total issue here)
    ml_data = game_data['moneyline']
    
    if ml_data['has_significant_data']:
        # Get the top prices from each side
        fav_raw_odds = ml_data.get('favorite_raw_odds_list', [])
        dog_raw_odds = ml_data.get('underdog_raw_odds_list', [])
        
        if fav_raw_odds and dog_raw_odds:
            # Use top 3 price levels (or fewer if not available)
            top_n = min(3, len(fav_raw_odds), len(dog_raw_odds))
            
            # Calculate volume for top N price levels for favorite
            fav_top_volume = 0
            fav_top_breakdown = []
            
            for i in range(top_n):
                price = fav_raw_odds[i]
                price_bets = game_df[(game_df['Market'] == 'Moneyline') & 
                                   (game_df['Side'] == ml_data['favorite_side']) &
                                   (game_df['Odds'] == price)]
                #Use bet amount for fav(to win amount)
                price_volume = price_bets['Bet Amount'].sum()
                fav_top_volume += price_volume
                
                fav_top_breakdown.append({
                    'price': price,
                    'volume': price_volume
                })
            
            # Calculate volume for top N price levels for underdog
            dog_top_volume = 0
            dog_top_breakdown = []
            
            for i in range(top_n):
                price = dog_raw_odds[i]
                price_bets = game_df[(game_df['Market'] == 'Moneyline') & 
                                   (game_df['Side'] == ml_data['underdog_side']) &
                                   (game_df['Odds'] == price)]
                #Use sum untaken for dog(risk amount)
                price_volume = price_bets['Sum Untaken'].sum()
                dog_top_volume += price_volume
                
                dog_top_breakdown.append({
                    'price': price,
                    'volume': price_volume
                })
                        # Store summary data regardless of imbalance
            results['moneyline_summary'] = {
                'has_data': True,
                'favorite_team': ml_data['favorite_team'],
                'underdog_team': ml_data['underdog_team'],
                'favorite_top_volume': fav_top_volume,
                'favorite_breakdown': fav_top_breakdown,
                'underdog_top_volume': dog_top_volume,
                'underdog_breakdown': dog_top_breakdown
            }
            # Calculate imbalance ratio
            if fav_top_volume > dog_top_volume:
                larger_side = 'Favorite'
                larger_team = ml_data['favorite_team']
                smaller_team = ml_data['underdog_team']
                ratio = fav_top_volume / dog_top_volume if dog_top_volume > 0 else float('inf')
            else:
                larger_side = 'Underdog'
                larger_team = ml_data['underdog_team']
                smaller_team = ml_data['favorite_team']
                ratio = dog_top_volume / fav_top_volume if fav_top_volume > 0 else float('inf')
            
            # Moneyline markets often have more natural imbalance, so use higher threshold
            min_volume_threshold = 500
            ratio_threshold = 1.5  # Higher for moneyline than spread
            
            if ratio > ratio_threshold and (fav_top_volume > min_volume_threshold or dog_top_volume > min_volume_threshold):
                results['moneyline_signals'].append({
                    'market': 'Moneyline',
                    'signal': smaller_team, #The side that has the most open interest is the side that the SHARP bettors are betting against, so the smaller team is the signal
                    'larger_side': larger_side,
                    'favorite_top_volume': fav_top_volume,
                    'favorite_breakdown': fav_top_breakdown,
                    'underdog_top_volume': dog_top_volume,
                    'underdog_breakdown': dog_top_breakdown,
                    'imbalance_ratio': ratio,
                    'significance': 'High' if ratio > 8 else 'Medium'
                })
    
    # Calculate total signals
    results['total_signals'] = len(results['moneyline_signals']) + len(results['spread_signals'])
    
    return results






console = Console()


#New function for menu navigation
def navigate_menus(current_menu_func, *args, **kwargs):
    """
    Navigation wrapper function that supports going back
    Returns True if user wants to go back, False if they want to exit
    """
    while True:
        # Call the current menu function
        result = current_menu_func(*args, **kwargs)
        
        if result == "back":
            return True  # Go back to previous menu
        elif result == "exit":
            return False  # Exit completely
        elif result == "main":
            return "main"  # Return to main menu
        

def create_dashboard(analyses):
    """Create a rich dashboard layout with comprehensive orderbook analysis data"""
    # Create layout
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main")
    )
    layout["main"].split_row(
        Layout(name="signals", ratio=2),
        Layout(name="details", ratio=2)
    )
    
    # Create header
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = Panel(f"Orderbook Analysis Dashboard - Last Updated: {current_time}", style="bold blue")
    layout["header"].update(header)
    
    # Create signals table
    signals_table = Table(title="Detected Signals", box=ROUNDED, show_lines=True)
    signals_table.add_column("Game", style="cyan", no_wrap=True)
    signals_table.add_column("Signal", style="green")
    signals_table.add_column("Market", style="magenta")
    signals_table.add_column("Strength", style="yellow")
    signals_table.add_column("Ratio", style="red")
    
    # Add rows for each signal
    signal_count = 0
    previous_game = None
    
    for game_id, analysis in analyses.items():
        game_name = f"{analysis['game_info']['away_team']} @ {analysis['game_info']['home_team']}"
        
        # Track signals by game for grouping
        game_signals = []
        
        # Collect sharp signals
        for signal in analysis.get('sharp_signals', []):
            game_signals.append({
                'game': game_name,
                'signal': signal['signal'],
                'market': signal['market'],
                'strength': signal['strength'],
                'ratio': "N/A"
            })
            signal_count += 1
        
        # Collect top book imbalances
        if 'top_book_imbalance' in analysis:
            for signal in analysis['top_book_imbalance'].get('moneyline_signals', []):
                game_signals.append({
                    'game': game_name,
                    'signal': signal['signal'],
                    'market': "ML Imbalance",
                    'strength': signal['significance'],
                    'ratio': f"{signal['imbalance_ratio']:.2f}x"
                })
                signal_count += 1
            
            for signal in analysis['top_book_imbalance'].get('spread_signals', []):
                game_signals.append({
                    'game': game_name,
                    'signal': signal['signal'],
                    'market': f"Spread {signal.get('main_spread', '')}",
                    'strength': signal['significance'],
                    'ratio': f"{signal['imbalance_ratio']:.2f}x"
                })
                signal_count += 1
        
        # Collect unmatched liquidity signals
        if 'unmatched_liquidity' in analysis:
            for signal in analysis['unmatched_liquidity'].get('moneyline_signals', []):
                game_signals.append({
                    'game': game_name,
                    'signal': signal.get('team', 'Unknown'),
                    'market': "ML Unmatched",
                    'strength': signal.get('significance', 'Unknown'),
                    'ratio': "N/A"
                })
                signal_count += 1
            
            for signal in analysis['unmatched_liquidity'].get('spread_signals', []):
                game_signals.append({
                    'game': game_name,
                    'signal': signal.get('team', 'Unknown'),
                    'market': f"Spread {signal.get('spread', '')} Unmatched",
                    'strength': signal.get('significance', 'Unknown'),
                    'ratio': "N/A"
                })
                signal_count += 1
        
        # Add all signals for this game
        for i, signal in enumerate(game_signals):
            signals_table.add_row(
                signal['game'] if i == 0 else "",  # Only show game name once per game
                signal['signal'],
                signal['market'],
                signal['strength'],
                signal['ratio']
            )
    
    if signal_count == 0:
        signals_table.add_row("No signals detected", "", "", "", "")
    
    # Create details table with raw orderbook data - WITH HORIZONTAL LINES
    details = Table(title="Orderbook Details", box=ROUNDED, show_lines=True)
    details.add_column("Game", style="cyan", no_wrap=True)
    details.add_column("Market", style="yellow")
    details.add_column("Teams", style="green")
    details.add_column("Top Prices", style="magenta")
    details.add_column("Volume by Team", style="blue", justify="right")
    
    previous_game = None
    
    for game_id, analysis in analyses.items():
        game_name = f"{analysis['game_info']['away_team']} @ {analysis['game_info']['home_team']}"
        
        # Add moneyline details
        ml_data = analysis.get('moneyline', {})
        if ml_data and 'favorite_team' in ml_data:
            fav_team = ml_data['favorite_team']
            dog_team = ml_data['underdog_team']
            
            # Get top book summary if available
            if 'top_book_imbalance' in analysis and analysis['top_book_imbalance']['moneyline_summary']['has_data']:
                ml_summary = analysis['top_book_imbalance']['moneyline_summary']
                
                # Format top prices
                fav_prices = []
                for price in ml_summary['favorite_breakdown'][:3]:
                    fav_prices.append(f"{price['price']}:{price['volume']:.0f}")
                
                dog_prices = []
                for price in ml_summary['underdog_breakdown'][:3]:
                    dog_prices.append(f"{price['price']}:{price['volume']:.0f}")
                
                fav_prices_str = ", ".join(fav_prices)
                dog_prices_str = ", ".join(dog_prices)
                
                prices_display = f"{fav_team}: {fav_prices_str}\n{dog_team}: {dog_prices_str}"
                
                # Format volumes with team names
                volumes_display = f"{fav_team}: ${ml_summary['favorite_top_volume']:.0f}\n{dog_team}: ${ml_summary['underdog_top_volume']:.0f}"
                
                details.add_row(
                    game_name,
                    "Moneyline",
                    f"{fav_team} (Fav) vs\n{dog_team} (Dog)",
                    prices_display,
                    volumes_display
                )
        
        # Add spread details
        spread_data = analysis.get('spread', {})
        if spread_data and 'favorite_team' in spread_data:
            fav_team = spread_data['favorite_team']
            dog_team = spread_data['underdog_team']
            
            # Get top book summary if available
            if 'top_book_imbalance' in analysis and analysis['top_book_imbalance']['spread_summary']['has_data']:
                spread_summary = analysis['top_book_imbalance']['spread_summary']
                
                # Format top prices
                fav_prices = []
                for price in spread_summary['favorite_breakdown'][:3]:
                    fav_prices.append(f"{price['price']}:{price['volume']:.0f}")
                
                dog_prices = []
                for price in spread_summary['underdog_breakdown'][:3]:
                    dog_prices.append(f"{price['price']}:{price['volume']:.0f}")
                
                fav_prices_str = ", ".join(fav_prices)
                dog_prices_str = ", ".join(dog_prices)
                
                prices_display = f"{fav_team}: {fav_prices_str}\n{dog_team}: {dog_prices_str}"
                
                # Format volumes with team names
                volumes_display = f"{fav_team}: ${spread_summary['favorite_top_volume']:.0f}\n{dog_team}: ${spread_summary['underdog_top_volume']:.0f}"
                
                details.add_row(
                    "",  # Leave game name blank to group with the moneyline row
                    f"Spread {spread_data.get('main_spread', '')}",
                    f"{fav_team} (Fav) vs\n{dog_team} (Dog)",
                    prices_display,
                    volumes_display
                )
    
    # Update the layout
    layout["signals"].update(signals_table)
    layout["details"].update(details)
    
    return layout

def display_dashboard(analyses):
    """Display the dashboard once"""
    dashboard = create_dashboard(analyses)
    console.print(dashboard)

def live_dashboard(analyze_func, interval_seconds=300):
    """Run a live updating dashboard"""
    with console.screen() as screen:
        while True:
            try:
                # Run analysis
                analyses, _ = analyze_func()
                
                # Update dashboard
                dashboard = create_dashboard(analyses)
                screen.update(dashboard)
                
                # Wait for next update
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")
                time.sleep(30)  # Wait a bit before retrying




import datetime

def process_full_orderbook(orderbook_data):
    """Process the full orderbook data for all games at once"""
    all_games_bet_data = []
    all_games_info = {}
    
    # Iterate through each game in the response
    for game in orderbook_data:
        game_id = game['id']
        event_name = game['eventName']
        
        # Extract team information
        home_team = next((p['shortName'] for p in game['participants'] if p['homeAway'] == 'home'), None)
        away_team = next((p['shortName'] for p in game['participants'] if p['homeAway'] == 'away'), None)
        
        # Store game info
        all_games_info[game_id] = {
            "event_name": event_name,
            "league": game['league'],
            "start_time": game['start'],
            "home_team": home_team,
            "away_team": away_team
        }
        
        # Process moneylines
        for bet in game.get('awayMoneylines', []):
            all_games_bet_data.append({
                "GameID": game_id,
                "Event": event_name,
                "Market": "Moneyline",
                "Side": "Away",
                "Team": away_team,
                "Odds": bet["odds"],
                "Spread/Total": "N/A",
                "Sum Untaken": bet["sumUntaken"],
                "Bet Amount": bet["bet"]
            })
            
        for bet in game.get('homeMoneylines', []):
            all_games_bet_data.append({
                "GameID": game_id,
                "Event": event_name,
                "Market": "Moneyline",
                "Side": "Home",
                "Team": home_team,
                "Odds": bet["odds"],
                "Spread/Total": "N/A",
                "Sum Untaken": bet["sumUntaken"],
                "Bet Amount": bet["bet"]
            })
        
        # Process spreads - these are in list format in your JSON
        for bet in game.get('awaySpreads', []):
            all_games_bet_data.append({
                "GameID": game_id,
                "Event": event_name,
                "Market": "Spread",
                "Side": "Away",
                "Team": away_team,
                "Odds": bet["odds"],
                "Spread/Total": bet["spread"],
                "Sum Untaken": bet["sumUntaken"],
                "Bet Amount": bet["bet"]
            })
                
        for bet in game.get('homeSpreads', []):
            all_games_bet_data.append({
                "GameID": game_id,
                "Event": event_name, 
                "Market": "Spread",
                "Side": "Home",
                "Team": home_team,
                "Odds": bet["odds"],
                "Spread/Total": bet["spread"],
                "Sum Untaken": bet["sumUntaken"],
                "Bet Amount": bet["bet"]
            })
        
        # Process totals
        for bet in game.get('over', []):
            all_games_bet_data.append({
                "GameID": game_id,
                "Event": event_name,
                "Market": "Total",
                "Side": "Over",
                "Team": "N/A",
                "Odds": bet["odds"],
                "Spread/Total": bet["total"],
                "Sum Untaken": bet["sumUntaken"],
                "Bet Amount": bet["bet"]
            })
            
        for bet in game.get('under', []):
            all_games_bet_data.append({
                "GameID": game_id,
                "Event": event_name,
                "Market": "Total",
                "Side": "Under", 
                "Team": "N/A",
                "Odds": bet["odds"],
                "Spread/Total": bet["total"],
                "Sum Untaken": bet["sumUntaken"],
                "Bet Amount": bet["bet"]
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_games_bet_data)
    return df, all_games_info



import json
import sqlite3
import os
def ensure_database_initialized(db_path='orderbook_analyzer.db'):
    """Check if database exists and is properly initialized"""
    if not os.path.exists(db_path):
        # Database doesn't exist, create it
        initialize_database(db_path)
        return
    
    # Check if tables exist
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Check for key tables
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_results'")
    tables_exist = c.fetchone() is not None
    
    conn.close()
    
    if not tables_exist:
        # Tables don't exist, initialize database
        initialize_database(db_path)

def initialize_database(db_path='orderbook_analyzer.db'):
    """Create database tables based on the actual data structure"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Table for raw orderbook snapshots
    c.execute('''CREATE TABLE IF NOT EXISTS orderbook_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        raw_data TEXT NOT NULL
    )''')
    
    # Table for game metadata
    c.execute('''CREATE TABLE IF NOT EXISTS games (
        game_id TEXT PRIMARY KEY,
        event_name TEXT NOT NULL,
        league TEXT,
        start_time TEXT,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        first_seen TEXT NOT NULL,
        last_updated TEXT NOT NULL
    )''')
    
    # Table for orderbook details
    c.execute('''CREATE TABLE IF NOT EXISTS orderbook_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_id INTEGER NOT NULL,
        game_id TEXT NOT NULL,
        market TEXT NOT NULL,
        side TEXT NOT NULL,
        team TEXT NOT NULL,
        odds REAL NOT NULL,
        spread_total TEXT,
        sum_untaken REAL NOT NULL,
        bet_amount REAL NOT NULL,
        matched_liquidity INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (snapshot_id) REFERENCES orderbook_snapshots(id),
        FOREIGN KEY (game_id) REFERENCES games(game_id)
    )''')
    
    # Table for analysis results with relevant fields
    c.execute('''CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        game_id TEXT NOT NULL,
        
        /* Key metrics */
        ml_favorite TEXT,
        ml_underdog TEXT,
        ml_favorite_best_odds REAL,
        ml_underdog_best_odds REAL,
        ml_imbalance REAL,
        
        spread_value REAL,
        spread_favorite TEXT,
        spread_underdog TEXT,
        spread_imbalance REAL,
        
        matched_liquidity_pct REAL,
        matched_liquidity_count INTEGER,
        
        /* Signal indicators */
        sharp_signal_count INTEGER,
        unmatched_signal_count INTEGER,
        top_book_signal_count INTEGER,
        
        /* Store full JSON data */
        analysis_data TEXT NOT NULL,
        
        FOREIGN KEY (game_id) REFERENCES games(game_id)
    )''')
    
    # Table for individual signals
    c.execute('''CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER NOT NULL,
        game_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        
        signal_type TEXT NOT NULL,  /* 'sharp', 'unmatched_liquidity', 'top_book_imbalance' */
        market TEXT NOT NULL,       /* 'Moneyline', 'Spread' */
        team TEXT NOT NULL,
        strength TEXT,
        imbalance_ratio REAL,
        signal_data TEXT NOT NULL,
        
        FOREIGN KEY (analysis_id) REFERENCES analysis_results(id),
        FOREIGN KEY (game_id) REFERENCES games(game_id)
    )''')
    
    # Create indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team, away_team)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON analysis_results(timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_analysis_game ON analysis_results(game_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_signals_game ON signals(game_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type)')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

def save_orderbook_snapshot(orderbook_data, db_path='orderbook_analyzer.db'):
    """Save a raw orderbook snapshot and return its ID"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    raw_data_json = json.dumps(orderbook_data)
    
    c.execute('INSERT INTO orderbook_snapshots (timestamp, raw_data) VALUES (?, ?)',
              (timestamp, raw_data_json))
    
    snapshot_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return snapshot_id, timestamp

def save_game_data(games_info, db_path='orderbook_analyzer.db'):
    """Save or update game metadata"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    for game_id, game_info in games_info.items():
        # Check if game already exists
        c.execute('SELECT game_id FROM games WHERE game_id = ?', (game_id,))
        exists = c.fetchone()
        
        if exists:
            # Update existing game
            c.execute('''
            UPDATE games SET 
                event_name = ?, 
                league = ?, 
                start_time = ?, 
                home_team = ?, 
                away_team = ?, 
                last_updated = ?
            WHERE game_id = ?
            ''', (
                game_info['event_name'],
                game_info.get('league', ''),
                game_info.get('start_time', ''),
                game_info['home_team'],
                game_info['away_team'],
                timestamp,
                game_id
            ))
        else:
            # Insert new game
            c.execute('''
            INSERT INTO games 
            (game_id, event_name, league, start_time, home_team, away_team, first_seen, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                game_id,
                game_info['event_name'],
                game_info.get('league', ''),
                game_info.get('start_time', ''),
                game_info['home_team'],
                game_info['away_team'],
                timestamp,
                timestamp
            ))
    
    conn.commit()
    conn.close()

def save_orderbook_details(snapshot_id, timestamp, all_games_df, db_path='orderbook_analyzer.db'):
    """Save detailed orderbook data for each row"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Use a transaction for efficiency
    c.execute('BEGIN TRANSACTION')
    
    try:
        for _, row in all_games_df.iterrows():
            c.execute('''
            INSERT INTO orderbook_details 
            (snapshot_id, game_id, market, side, team, odds, spread_total, sum_untaken, bet_amount, matched_liquidity, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot_id,
                row['GameID'],
                row['Market'],
                row['Side'],
                row['Team'],
                float(row['Odds']),
                str(row['Spread/Total']),
                float(row['Sum Untaken']),
                float(row['Bet Amount']),
                bool(row['matched_liquidity']),
                timestamp
            ))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error saving orderbook details: {e}")
        raise
    finally:
        conn.close()

def save_analysis_results(timestamp, all_analyses, db_path='orderbook_analyzer.db'):
    
    if not timestamp:
        timestamp = datetime.now().isoformat()
        
    print(f"Saving analysis results to database. Timestamp: {timestamp}")
    print(f"Number of analyses to save: {len(all_analyses)}")
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    try:
        c.execute("BEGIN TRANSACTION")
        saved_count = 0
        signals_count = 0

        for game_id, analysis in all_analyses.items():
            # Extract key data
            ml_data = analysis.get('moneyline', {})
            spread_data = analysis.get('spread', {})
            
            print(f"Processing game_id: {game_id}")
            print(f"ML favorite: {ml_data.get('favorite_team')}")
            print(f"Spread value: {spread_data.get('main_spread')}")
            
            # Ensure we have valid types for all fields that might be None
            ml_favorite = ml_data.get('favorite_team') or None
            ml_underdog = ml_data.get('underdog_team') or None
            ml_favorite_odds = ml_data.get('favorite_best_odds') or None
            ml_underdog_odds = ml_data.get('underdog_best_odds') or None
            ml_imbalance = ml_data.get('imbalance') or 0.0
            
            spread_value = spread_data.get('main_spread') or None
            spread_favorite = spread_data.get('favorite_team') or None
            spread_underdog = spread_data.get('underdog_team') or None
            spread_imbalance = spread_data.get('imbalance') or 0.0
            
            matched_pct = analysis.get('matched_liquidity_percentage') or 0.0
            matched_count = analysis.get('matched_liquidity_count') or 0
            
            sharp_count = len(analysis.get('sharp_signals', []))
            unmatched_count = analysis.get('unmatched_liquidity', {}).get('total_signals', 0)
            top_book_count = analysis.get('top_book_imbalance', {}).get('total_signals', 0)
            
            # Convert entire analysis to JSON for storage - USE THE CUSTOM ENCODER
            analysis_json = json.dumps(analysis, cls=NumpyEncoder)
            
            # Insert analysis result
            c.execute('''
            INSERT INTO analysis_results (
                timestamp, game_id, 
                ml_favorite, ml_underdog, ml_favorite_best_odds, ml_underdog_best_odds, ml_imbalance,
                spread_value, spread_favorite, spread_underdog, spread_imbalance,
                matched_liquidity_pct, matched_liquidity_count,
                sharp_signal_count, unmatched_signal_count, top_book_signal_count,
                analysis_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                game_id,
                ml_favorite,
                ml_underdog,
                ml_favorite_odds,
                ml_underdog_odds,
                ml_imbalance,
                spread_value,
                spread_favorite,
                spread_underdog,
                spread_imbalance,
                matched_pct,
                matched_count,
                sharp_count,
                unmatched_count,
                top_book_count,
                analysis_json
            ))
            
            analysis_id = c.lastrowid
            saved_count += 1
            
            if analysis_id:
                print(f"Saved analysis ID: {analysis_id}")
                
                # Save each individual signal - USING THE CUSTOM ENCODER
                # Sharp signals
                for signal in analysis.get('sharp_signals', []):
                    c.execute('''
                    INSERT INTO signals (
                        analysis_id, game_id, timestamp, signal_type, market, team, 
                        strength, imbalance_ratio, signal_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        analysis_id,
                        game_id,
                        timestamp,
                        'sharp',
                        signal.get('market', ''),
                        signal.get('signal', ''),
                        signal.get('strength', ''),
                        0.0,  # Sharp signals don't have ratio
                        json.dumps(signal, cls=NumpyEncoder)
                    ))
                    signals_count += 1
                
                # Unmatched liquidity signals
                for market in ['moneyline_signals', 'spread_signals']:
                    for signal in analysis.get('unmatched_liquidity', {}).get(market, []):
                        c.execute('''
                        INSERT INTO signals (
                            analysis_id, game_id, timestamp, signal_type, market, team, 
                            strength, imbalance_ratio, signal_data
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            analysis_id,
                            game_id,
                            timestamp,
                            'unmatched_liquidity',
                            'Moneyline' if market == 'moneyline_signals' else 'Spread',
                            signal.get('team', ''),
                            signal.get('significance', ''),
                            0.0,  # Unmatched doesn't have ratio
                            json.dumps(signal, cls=NumpyEncoder)
                        ))
                        signals_count += 1
                
                # Top book imbalance signals
                for market in ['moneyline_signals', 'spread_signals']:
                    for signal in analysis.get('top_book_imbalance', {}).get(market, []):
                        c.execute('''
                        INSERT INTO signals (
                            analysis_id, game_id, timestamp, signal_type, market, team, 
                            strength, imbalance_ratio, signal_data
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            analysis_id,
                            game_id,
                            timestamp,
                            'top_book_imbalance',
                            'Moneyline' if market == 'moneyline_signals' else 'Spread',
                            signal.get('signal', ''),
                            signal.get('significance', ''),
                            signal.get('imbalance_ratio', 0.0),
                            json.dumps(signal, cls=NumpyEncoder)
                        ))
                        signals_count += 1
        
        c.execute("COMMIT")
        print(f"Successfully saved {saved_count} analyses and {signals_count} signals to database")
    
    except Exception as e:
        c.execute("ROLLBACK")
        print(f"Error saving analysis results: {str(e)}")
        import traceback
        traceback.print_exc()  # Print the full stack trace
    
    finally:
        conn.close()
# Debugging function to check database before saving
def check_db_before_save(db_path='orderbook_analyzer.db'):
    """Check if database tables are properly set up"""
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    try:
        # Check if analysis_results table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_results'")
        has_analysis_table = c.fetchone() is not None
        
        # Check if signals table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'")
        has_signals_table = c.fetchone() is not None
        
        # Get column info if tables exist
        analysis_cols = []
        signals_cols = []
        
        if has_analysis_table:
            c.execute("PRAGMA table_info(analysis_results)")
            analysis_cols = [row[1] for row in c.fetchall()]
            
        if has_signals_table:
            c.execute("PRAGMA table_info(signals)")
            signals_cols = [row[1] for row in c.fetchall()]
        
        print("Database status:")
        print(f"- analysis_results table exists: {has_analysis_table}")
        print(f"- signals table exists: {has_signals_table}")
        print(f"- analysis_results columns: {', '.join(analysis_cols)}")
        print(f"- signals columns: {', '.join(signals_cols)}")
        
        return has_analysis_table and has_signals_table
    
    except Exception as e:
        print(f"Error checking database: {str(e)}")
        return False
    
    finally:
        conn.close()


def get_recent_signals(hours=24, signal_type=None, market=None, db_path='orderbook_analyzer.db'):
    """Get recent signals of specified type and market"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    query = '''
    SELECT g.event_name, g.home_team, g.away_team, g.start_time,
           s.timestamp, s.signal_type, s.market, s.team, s.strength, s.imbalance_ratio
    FROM signals s
    JOIN games g ON s.game_id = g.game_id
    WHERE s.timestamp > datetime('now', ?)
    '''
    params = [f'-{hours} hours']
    
    if signal_type:
        query += ' AND s.signal_type = ?'
        params.append(signal_type)
    
    if market:
        query += ' AND s.market = ?'
        params.append(market)
    
    query += ' ORDER BY s.timestamp DESC'
    
    c.execute(query, params)
    results = [dict(row) for row in c.fetchall()]
    conn.close()
    
    return results


def live_dashboard(analyze_func, interval_seconds=300):
    """Run a live updating dashboard"""
    with console.screen() as screen:
        while True:
            try:
                # Run analysis
                analyses, _ = analyze_func()
                
                # Update dashboard
                dashboard = create_dashboard(analyses)
                screen.update(dashboard)
                
                # Wait for next update
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")
                time.sleep(30)  # Wait a bit before retrying



def analyze_all_games(orderbook_data, db_path='orderbook_analyzer.db'):
    """Analyze all games in the orderbook and detect sharp signals"""
    from datetime import datetime
    
    # Process the full orderbook
    ensure_database_initialized(db_path)
    
    # Verify database tables exist
    check_db_before_save(db_path)
    
    snapshot_id, timestamp = save_orderbook_snapshot(orderbook_data, db_path)
    all_games_df, all_games_info = process_full_orderbook(orderbook_data)
    save_game_data(all_games_info, db_path)
    
    # Add a column for matched liquidity to the full dataset
    all_games_df['matched_liquidity'] = False

    # For each game, identify matched liquidity for each bet type
    for game_id in all_games_df['GameID'].unique():
        game_df = all_games_df[all_games_df['GameID'] == game_id]
        matched_liquidity = identify_matched_liquidity(game_df, threshold=50)
        all_games_df.loc[all_games_df['GameID'] == game_id, 'matched_liquidity'] = matched_liquidity['matched_liquidity']

    # Save orderbook details
    save_orderbook_details(snapshot_id, timestamp, all_games_df, db_path)
    
    # Save the full dataset with matched liquidity identified
    now = datetime.now().strftime('%Y_%m_%d_%I%M%p')
    all_games_df.to_csv(f'all_games_orderbook_with_matches_{now}.csv', index=False)
    
    # Dictionary to store all analyses
    all_analyses = {}
    
    # Process each game individually
    for game_id, game_info in all_games_info.items():
        # Get game data (full data, not filtered)
        game_df = all_games_df[all_games_df['GameID'] == game_id]
        
        # Calculate matched liquidity percentage
        matched_percentage = (game_df['matched_liquidity'].sum() / len(game_df)) * 100 if not game_df.empty else 0
        
        # Split by market type
        ml_df = game_df[game_df['Market'] == 'Moneyline']
        spread_df = game_df[game_df['Market'] == 'Spread']
        
        # Analyze each market using full data
        ml_analysis = process_moneyline(ml_df)
        spread_analysis = process_spread(spread_df)
        
        # Combine analyses
        game_analysis = {
            'game_info': game_info,
            'moneyline': ml_analysis,
            'spread': spread_analysis,
            'matched_liquidity_percentage': matched_percentage,
            'matched_liquidity_count': game_df['matched_liquidity'].sum()
        }
        
        # Detect sharp signals
        game_analysis['sharp_signals'] = detect_sharp_signals(ml_analysis, spread_analysis)
        game_analysis['signal_count'] = len(game_analysis['sharp_signals'])

        # Add other analysis
        game_analysis['unmatched_liquidity'] = analyze_unmatched_liquidity(game_analysis, game_df)
        game_analysis['top_book_imbalance'] = detect_top_book_imbalance(game_analysis, game_df)
        
        # Store analysis
        all_analyses[game_id] = game_analysis
        
        # Print results
        print(f"\n{game_info['event_name']} ({game_info['away_team']} @ {game_info['home_team']})")
        print("-------------------------------------------")
    
    # Save analysis results to database with detailed debug logging
    try:
        print("Saving analysis results to database...")
        save_analysis_results(timestamp, all_analyses, db_path)
        print("Analysis results saved successfully")
    except Exception as e:
        print(f"Error saving analysis results: {str(e)}")
    try:
        track_signals(all_analyses, db_path)
    except Exception as e:
        print(f"Error tracking signals: {str(e)}")
    
    display_dashboard(all_analyses) 
    
    return all_analyses, all_games_df


from datetime import datetime, timedelta

def continuous_monitoring(interval_minutes=5, max_hours=24, db_path='orderbook_analyzer.db'):
    """Continuously monitor the orderbook at specified intervals"""
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=max_hours)
    
    print(f"Starting continuous monitoring at {start_time}")
    print(f"Will run until {end_time}")
    print(f"Data will be saved to {db_path}")
    
    while datetime.now() < end_time:
        try:
            # Get current time for logging
            current_time = datetime.now()
            print(f"\n--- Scraping orderbook at {current_time} ---")
            
            # Get auth token and scrape data
            auth_token = grab_auth_token()
            orderbook_data = scrape_raw_orderbook(auth_token)
            
            # Analyze the data and save to database
            analyses, _ = analyze_all_games(orderbook_data, db_path)
            
            # Check for signals
            signal_count = 0
            for game_id, analysis in analyses.items():
                total_signals = (
                    len(analysis.get('sharp_signals', [])) + 
                    analysis.get('unmatched_liquidity', {}).get('total_signals', 0) + 
                    analysis.get('top_book_imbalance', {}).get('total_signals', 0)
                )
                
                if total_signals > 0:
                    signal_count += 1
                    print(f"*** SIGNALS DETECTED for {analysis['game_info']['event_name']} ***")
            
            print(f"Completed analysis: {len(analyses)} games, {signal_count} with signals")
            print(f"All data saved to {db_path}")
            
            # Wait for the next interval
            next_check = datetime.now() + timedelta(minutes=interval_minutes)
            print(f"Next check scheduled for {next_check}")
            
            sleep_time = interval_minutes * 60
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"Error in monitoring loop: {str(e)}")
            print("Waiting 2 minutes before retrying...")
            time.sleep(120)
    
    print(f"Monitoring complete. Ran from {start_time} to {datetime.now()}")
    print(f"All data saved to {db_path}")

def inspect_database(db_path='orderbook_analyzer.db'):
    """List all tables in the database and show their structure"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    table_info = Table(title=f"Database Structure: {db_path}")
    table_info.add_column("Table Name", style="cyan")
    table_info.add_column("Column Name", style="yellow")
    table_info.add_column("Type", style="green")
    table_info.add_column("Not Null", style="magenta")
    table_info.add_column("Default", style="blue")
    table_info.add_column("Primary Key", style="red")
    
    for table_name in tables:
        table_name = table_name[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        for i, col in enumerate(columns):
            table_info.add_row(
                table_name if i == 0 else "",
                col[1],         # Column name
                col[2],         # Data type
                "Yes" if col[3] else "No",  # Not Null
                str(col[4]) if col[4] is not None else "",  # Default value
                "Yes" if col[5] else "No"   # Primary Key
            )
    
    console.print(table_info)
    conn.close()


def count_table_rows(db_path='orderbook_analyzer.db'):
    """Count the number of rows in each table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    count_table = Table(title="Row Counts by Table")
    count_table.add_column("Table Name", style="cyan")
    count_table.add_column("Row Count", style="yellow", justify="right")
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        count_table.add_row(table_name, str(row_count))
    
    console.print(count_table)
    conn.close()

def sample_table_data(table_name, limit=5, db_path='orderbook_analyzer.db'):
    """Print a sample of rows from a specific table"""
    conn = sqlite3.connect(db_path)
    
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Read data into DataFrame for easy display
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
        
        if len(df) == 0:
            console.print(f"[yellow]No data found in table: {table_name}[/yellow]")
            return
            
        # Format table for display
        sample_table = Table(title=f"Sample Data from {table_name} ({len(df)} of {total_rows} rows)")
        
        # Add columns with appropriate width constraints
        for col in df.columns:
            if "data" in col.lower() or col.lower().endswith("_json"):
                # JSON columns get truncated display
                sample_table.add_column(col, max_width=40)
            else:
                sample_table.add_column(col)
                
        # Add rows
        for _, row in df.iterrows():
            values = []
            for col in df.columns:
                val = row[col]
                # Format JSON columns nicely
                if isinstance(val, str) and val.startswith('{') and val.endswith('}'):
                    try:
                        val = json.dumps(json.loads(val), indent=2)
                        if len(val) > 40:
                            val = val[:37] + "..."
                    except:
                        pass
                values.append(str(val))
            sample_table.add_row(*values)
            
        console.print(sample_table)
        
        # For JSON columns, offer to show full content for a specific row
        json_cols = [col for col in df.columns if "data" in col.lower() or col.lower().endswith("_json")]
        if json_cols and len(df) > 0:
            console.print("\n[bold]JSON columns detected. View full content?[/bold]")
            view_json = Prompt.ask("View JSON for row #", choices=[str(i) for i in range(1, len(df)+1)] + ["n"], default="n")
            
            if view_json != "n":
                row_idx = int(view_json) - 1
                col = Prompt.ask("Select column", choices=json_cols)
                
                try:
                    json_data = json.loads(df.iloc[row_idx][col])
                    console.print(f"\n[bold]Full content of {col} for row {row_idx+1}:[/bold]")
                    console.print(json.dumps(json_data, indent=2))
                except Exception as e:
                    console.print(f"[bold red]Error parsing JSON: {str(e)}[/bold red]")
    
    except Exception as e:
        console.print(f"[bold red]Error inspecting table {table_name}: {str(e)}[/bold red]")
    
    finally:
        conn.close()


def query_signals(hours=24, db_path='orderbook_analyzer.db'):
    """Query recent signals from the database"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
    SELECT s.id, s.game_id, s.timestamp, s.signal_type, s.market, s.team, 
           s.strength, s.imbalance_ratio, 
           g.home_team, g.away_team, g.event_name
    FROM signals s
    JOIN games g ON s.game_id = g.game_id
    WHERE s.timestamp > datetime('now', ?)
    ORDER BY s.timestamp DESC
    """
    
    cursor.execute(query, [f'-{hours} hours'])
    rows = cursor.fetchall()
    
    signals_table = Table(title=f"Recent Signals (Last {hours} hours)")
    signals_table.add_column("ID", style="dim")
    signals_table.add_column("Game", style="cyan")
    signals_table.add_column("Signal Type", style="yellow")
    signals_table.add_column("Market", style="magenta")
    signals_table.add_column("Team", style="green")
    signals_table.add_column("Strength", style="blue")
    signals_table.add_column("Ratio", style="red")
    signals_table.add_column("Timestamp", style="dim")
    
    for row in rows:
        signals_table.add_row(
            str(row['id']),
            f"{row['away_team']} @ {row['home_team']}",
            row['signal_type'],
            row['market'],
            row['team'],
            row['strength'] or "N/A",
            f"{row['imbalance_ratio']:.2f}x" if row['imbalance_ratio'] > 0 else "N/A",
            row['timestamp'].split('T')[0] + " " + row['timestamp'].split('T')[1][:8]
        )
    
    console.print(signals_table)
    conn.close()

def query_signals_by_type(signal_type, db_path='orderbook_analyzer.db'):
    """Query signals by type"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
    SELECT s.id, s.game_id, s.timestamp, s.signal_type, s.market, s.team, 
           s.strength, s.imbalance_ratio, 
           g.home_team, g.away_team, g.event_name
    FROM signals s
    JOIN games g ON s.game_id = g.game_id
    WHERE s.signal_type = ?
    ORDER BY s.timestamp DESC
    LIMIT 50
    """
    
    cursor.execute(query, [signal_type])
    rows = cursor.fetchall()
    
    signals_table = Table(title=f"Signals by Type: {signal_type}")
    signals_table.add_column("ID", style="dim")
    signals_table.add_column("Game", style="cyan")
    signals_table.add_column("Market", style="magenta")
    signals_table.add_column("Team", style="green")
    signals_table.add_column("Strength", style="blue")
    signals_table.add_column("Ratio", style="red")
    signals_table.add_column("Timestamp", style="dim")
    
    for row in rows:
        signals_table.add_row(
            str(row['id']),
            f"{row['away_team']} @ {row['home_team']}",
            row['market'],
            row['team'],
            row['strength'] or "N/A",
            f"{row['imbalance_ratio']:.2f}x" if row['imbalance_ratio'] > 0 else "N/A",
            row['timestamp'].split('T')[0] + " " + row['timestamp'].split('T')[1][:8]
        )
    
    console.print(signals_table)
    conn.close()


def query_signals_by_market(market, db_path='orderbook_analyzer.db'):
    """Query signals by market"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
    SELECT s.id, s.game_id, s.timestamp, s.signal_type, s.market, s.team, 
           s.strength, s.imbalance_ratio, 
           g.home_team, g.away_team, g.event_name
    FROM signals s
    JOIN games g ON s.game_id = g.game_id
    WHERE s.market LIKE ?
    ORDER BY s.timestamp DESC
    LIMIT 50
    """
    
    cursor.execute(query, [f"%{market}%"])
    rows = cursor.fetchall()
    
    signals_table = Table(title=f"Signals by Market: {market}")
    signals_table.add_column("ID", style="dim")
    signals_table.add_column("Game", style="cyan")
    signals_table.add_column("Signal Type", style="yellow")
    signals_table.add_column("Team", style="green")
    signals_table.add_column("Strength", style="blue")
    signals_table.add_column("Ratio", style="red")
    signals_table.add_column("Timestamp", style="dim")
    
    for row in rows:
        signals_table.add_row(
            str(row['id']),
            f"{row['away_team']} @ {row['home_team']}",
            row['signal_type'],
            row['team'],
            row['strength'] or "N/A",
            f"{row['imbalance_ratio']:.2f}x" if row['imbalance_ratio'] > 0 else "N/A",
            row['timestamp'].split('T')[0] + " " + row['timestamp'].split('T')[1][:8]
        )
    
    console.print(signals_table)
    conn.close()
def query_signals_by_team(team, db_path='orderbook_analyzer.db'):
    """Query signals for a specific team"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
    SELECT s.id, s.game_id, s.timestamp, s.signal_type, s.market, s.team, 
           s.strength, s.imbalance_ratio, 
           g.home_team, g.away_team, g.event_name
    FROM signals s
    JOIN games g ON s.game_id = g.game_id
    WHERE s.team LIKE ? OR g.home_team LIKE ? OR g.away_team LIKE ?
    ORDER BY s.timestamp DESC
    LIMIT 50
    """
    
    cursor.execute(query, [f"%{team}%", f"%{team}%", f"%{team}%"])
    rows = cursor.fetchall()
    
    signals_table = Table(title=f"Signals for Team: {team}")
    signals_table.add_column("ID", style="dim")
    signals_table.add_column("Game", style="cyan")
    signals_table.add_column("Signal Type", style="yellow")
    signals_table.add_column("Market", style="magenta")
    signals_table.add_column("Team", style="green", style_rule=lambda x: "bold green" if team.upper() in x.upper() else "")
    signals_table.add_column("Strength", style="blue")
    signals_table.add_column("Timestamp", style="dim")
    
    for row in rows:
        signals_table.add_row(
            str(row['id']),
            f"{row['away_team']} @ {row['home_team']}",
            row['signal_type'],
            row['market'],
            row['team'],
            row['strength'] or "N/A",
            row['timestamp'].split('T')[0] + " " + row['timestamp'].split('T')[1][:8]
        )
    
    console.print(signals_table)
    conn.close()

def query_team_history(team, db_path='orderbook_analyzer.db'):
    """Query historical analysis for a specific team"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
    SELECT g.game_id, g.event_name, g.home_team, g.away_team, g.start_time,
           a.timestamp, a.ml_favorite, a.ml_underdog, a.ml_imbalance,
           a.spread_value, a.spread_favorite, a.spread_underdog, a.spread_imbalance
    FROM games g
    JOIN analysis_results a ON g.game_id = a.game_id
    WHERE g.home_team LIKE ? OR g.away_team LIKE ?
    ORDER BY a.timestamp DESC
    """
    
    cursor.execute(query, [f'%{team}%', f'%{team}%'])
    rows = cursor.fetchall()
    
    if not rows:
        console.print(f"[yellow]No games found for team: {team}[/yellow]")
        return
    
    history_table = Table(title=f"Analysis History for {team}")
    history_table.add_column("Game", style="cyan")
    history_table.add_column("Time", style="dim")
    history_table.add_column("ML Favorite", style="green")
    history_table.add_column("ML Imbalance", style="yellow")
    history_table.add_column("Spread", style="magenta")
    history_table.add_column("Spread Favorite", style="blue")
    history_table.add_column("Spread Imbalance", style="red")
    
    for row in rows:
        game_name = f"{row['away_team']} @ {row['home_team']}"
        start_time = row['start_time'].split('T')[0] if row['start_time'] else "N/A"
        analysis_time = row['timestamp'].split('T')[0] + " " + row['timestamp'].split('T')[1][:8]
        
        history_table.add_row(
            game_name,
            f"{start_time}\n{analysis_time}",
            str(row['ml_favorite'] or "N/A"),
            f"{row['ml_imbalance']:.2f}" if row['ml_imbalance'] is not None else "N/A",
            str(row['spread_value'] or "N/A"),
            str(row['spread_favorite'] or "N/A"),
            f"{row['spread_imbalance']:.2f}" if row['spread_imbalance'] is not None else "N/A"
        )
    
    console.print(history_table)
    conn.close()

def query_recent_analysis(days=7, db_path='orderbook_analyzer.db'):
    """Query recent analysis results"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
    SELECT a.id, a.timestamp, a.game_id, 
           a.ml_favorite, a.ml_underdog, a.ml_imbalance,
           a.spread_value, a.spread_favorite, a.spread_underdog, a.spread_imbalance,
           a.sharp_signal_count, a.unmatched_signal_count, a.top_book_signal_count,
           g.home_team, g.away_team, g.event_name
    FROM analysis_results a
    JOIN games g ON a.game_id = g.game_id
    WHERE a.timestamp > datetime('now', ?)
    ORDER BY a.timestamp DESC
    """
    
    cursor.execute(query, [f'-{days} days'])
    rows = cursor.fetchall()
    
    analysis_table = Table(title=f"Recent Analysis Results (Last {days} days)")
    analysis_table.add_column("ID", style="dim")
    analysis_table.add_column("Game", style="cyan")
    analysis_table.add_column("Time", style="dim")
    analysis_table.add_column("ML Favorite", style="green")
    analysis_table.add_column("Spread", style="magenta")
    analysis_table.add_column("Signal Count", style="yellow")
    
    for row in rows:
        game_name = f"{row['away_team']} @ {row['home_team']}"
        analysis_time = row['timestamp'].split('T')[0] + " " + row['timestamp'].split('T')[1][:8]
        signal_count = (row['sharp_signal_count'] or 0) + (row['unmatched_signal_count'] or 0) + (row['top_book_signal_count'] or 0)
        
        analysis_table.add_row(
            str(row['id']),
            game_name,
            analysis_time,
            str(row['ml_favorite'] or "N/A"),
            f"{row['spread_value'] or 'N/A'} ({row['spread_favorite'] or 'N/A'})",
            str(signal_count)
        )
    
    console.print(analysis_table)
    conn.close()

def query_most_signaled_teams(db_path='orderbook_analyzer.db', today_only=True):
    """
    Query teams with the most signals, with option to filter for today only
    
    Parameters:
    db_path (str): Path to database file
    today_only (bool): If True, only include signals from today
    """
    from rich.console import Console
    from rich.table import Table
    import sqlite3
    from datetime import date
    
    console = Console()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Base query
    query = """
    SELECT team, COUNT(*) as signal_count,
           SUM(CASE WHEN signal_type = 'sharp' THEN 1 ELSE 0 END) as sharp_count,
           SUM(CASE WHEN signal_type = 'unmatched_liquidity' THEN 1 ELSE 0 END) as unmatched_count,
           SUM(CASE WHEN signal_type = 'top_book_imbalance' THEN 1 ELSE 0 END) as imbalance_count
    FROM signals
    """
    
    # Add today filter if requested
    if today_only:
        today = date.today().isoformat()
        query += f" WHERE date(timestamp) = '{today}'"
    
    # Complete the query
    query += """
    GROUP BY team
    ORDER BY signal_count DESC
    LIMIT 20
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # Create title with date information
    title = "Teams with Most Signals"
    if today_only:
        title += f" (Today: {date.today().strftime('%Y-%m-%d')})"
    
    team_table = Table(title=title)
    team_table.add_column("Team", style="green")
    team_table.add_column("Total Signals", style="cyan", justify="right")
    team_table.add_column("Sharp", style="yellow", justify="right")
    team_table.add_column("Unmatched", style="magenta", justify="right")
    team_table.add_column("Imbalance", style="blue", justify="right")
    
    for row in rows:
        team_table.add_row(
            row['team'],
            str(row['signal_count']),
            str(row['sharp_count']),
            str(row['unmatched_count']),
            str(row['imbalance_count'])
        )
    
    console.print(team_table)
    conn.close()

    
def query_games_with_most_signals(db_path='orderbook_analyzer.db', today_only=True):
    """
    Query games with the most signals, with option to filter for today only
    
    Parameters:
    db_path (str): Path to database file
    today_only (bool): If True, only include signals from today
    """
    from rich.console import Console
    from rich.table import Table
    import sqlite3
    from datetime import date
    
    console = Console()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Base query
    query = """
    SELECT s.game_id, g.home_team, g.away_team, g.start_time, g.event_name,
           COUNT(*) as signal_count,
           GROUP_CONCAT(DISTINCT s.team) as signaled_teams
    FROM signals s
    JOIN games g ON s.game_id = g.game_id
    """
    
    # Add today filter if requested
    if today_only:
        today = date.today().isoformat()
        query += f" WHERE date(s.timestamp) = '{today}'"
    
    # Complete the query
    query += """
    GROUP BY s.game_id
    ORDER BY signal_count DESC
    LIMIT 15
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # Create title with date information
    title = "Games with Most Signals"
    if today_only:
        title += f" (Today: {date.today().strftime('%Y-%m-%d')})"
    
    games_table = Table(title=title)
    games_table.add_column("Game", style="cyan")
    games_table.add_column("Date", style="dim")
    games_table.add_column("Signal Count", style="yellow", justify="right")
    games_table.add_column("Signaled Teams", style="green")
    
    for row in rows:
        game_name = f"{row['away_team']} @ {row['home_team']}"
        game_date = row['start_time'].split('T')[0] if row['start_time'] else "N/A"
        
        games_table.add_row(
            game_name,
            game_date,
            str(row['signal_count']),
            row['signaled_teams']
        )
    
    console.print(games_table)
    
    # If we're doing today only, also get the team breakdown
    if today_only and rows:
        # For each game, get signal counts by team
        for row in rows:  # Only do the top 5 to avoid too much output
            game_id = row['game_id']
            game_name = f"{row['away_team']} @ {row['home_team']}"
            
            team_query = """
            SELECT team, COUNT(*) as team_signals,
                   SUM(CASE WHEN signal_type = 'sharp' THEN 1 ELSE 0 END) as sharp,
                   SUM(CASE WHEN signal_type = 'unmatched_liquidity' THEN 1 ELSE 0 END) as unmatched,
                   SUM(CASE WHEN signal_type = 'top_book_imbalance' THEN 1 ELSE 0 END) as imbalance
            FROM signals
            WHERE game_id = ? AND date(timestamp) = ?
            GROUP BY team
            ORDER BY team_signals DESC
            """
            
            cursor.execute(team_query, (game_id, today))
            team_rows = cursor.fetchall()
            
            if team_rows:
                team_table = Table(title=f"Signal Breakdown for {game_name}")
                team_table.add_column("Team", style="green")
                team_table.add_column("Total", style="cyan", justify="right")
                team_table.add_column("Sharp", style="yellow", justify="right")
                team_table.add_column("Unmatched", style="magenta", justify="right")
                team_table.add_column("Imbalance", style="blue", justify="right")
                
                for team_row in team_rows:
                    team_table.add_row(
                        team_row['team'],
                        str(team_row['team_signals']),
                        str(team_row['sharp']),
                        str(team_row['unmatched']),
                        str(team_row['imbalance'])
                    )
                
                console.print(team_table)
                console.print()
    
    conn.close()


def query_market_activity_by_day(db_path='orderbook_analyzer.db'):
    """Query market activity by day"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
    SELECT date(timestamp) as day,
           COUNT(DISTINCT game_id) as games_count,
           COUNT(*) as total_analyses,
           SUM(sharp_signal_count) as sharp_signals,
           SUM(unmatched_signal_count) as unmatched_signals,
           SUM(top_book_signal_count) as topbook_signals
    FROM analysis_results
    GROUP BY day
    ORDER BY day DESC
    LIMIT 30
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    activity_table = Table(title="Market Activity by Day")
    activity_table.add_column("Date", style="cyan")
    activity_table.add_column("Games", style="yellow", justify="right")
    activity_table.add_column("Analyses", style="green", justify="right")
    activity_table.add_column("Sharp Signals", style="magenta", justify="right")
    activity_table.add_column("Unmatched", style="blue", justify="right")
    activity_table.add_column("Top Book", style="red", justify="right")
    
    for row in rows:
        activity_table.add_row(
            row['day'],
            str(row['games_count']),
            str(row['total_analyses']),
            str(row['sharp_signals'] or 0),
            str(row['unmatched_signals'] or 0),
            str(row['topbook_signals'] or 0)
        )
    
    console.print(activity_table)
    conn.close()

def query_signal_trends(db_path='orderbook_analyzer.db'):
    """Query signal trends over time"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
    SELECT date(timestamp) as day,
           signal_type,
           COUNT(*) as signal_count,
           COUNT(DISTINCT game_id) as games_count
    FROM signals
    GROUP BY day, signal_type
    ORDER BY day DESC, signal_type
    LIMIT 50
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    trend_table = Table(title="Signal Trends Over Time")
    trend_table.add_column("Date", style="cyan")
    trend_table.add_column("Signal Type", style="yellow")
    trend_table.add_column("Signal Count", style="green", justify="right")
    trend_table.add_column("Games", style="magenta", justify="right")
    
    for row in rows:
        trend_table.add_row(
            row['day'],
            row['signal_type'],
            str(row['signal_count']),
            str(row['games_count'])
        )
    
    console.print(trend_table)
    conn.close()




def query_game_history(team, db_path='orderbook_analyzer.db'):
    """Query historical analysis for a specific team"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
    SELECT g.game_id, g.event_name, g.home_team, g.away_team, g.start_time,
           a.timestamp, a.ml_favorite, a.ml_underdog, a.ml_imbalance,
           a.spread_value, a.spread_favorite, a.spread_underdog, a.spread_imbalance
    FROM games g
    JOIN analysis_results a ON g.game_id = a.game_id
    WHERE g.home_team LIKE ? OR g.away_team LIKE ?
    ORDER BY a.timestamp DESC
    """
    
    cursor.execute(query, [f'%{team}%', f'%{team}%'])
    rows = cursor.fetchall()
    
    if not rows:
        console.print(f"[yellow]No games found for team: {team}[/yellow]")
        return
    
    history_table = Table(title=f"Analysis History for {team}")
    history_table.add_column("Game", style="cyan")
    history_table.add_column("Time", style="dim")
    history_table.add_column("ML Favorite", style="green")
    history_table.add_column("ML Imbalance", style="yellow")
    history_table.add_column("Spread", style="magenta")
    history_table.add_column("Spread Favorite", style="blue")
    history_table.add_column("Spread Imbalance", style="red")
    
    for row in rows:
        game_name = f"{row['away_team']} @ {row['home_team']}"
        start_time = row['start_time'].split('T')[0] if row['start_time'] else "N/A"
        analysis_time = row['timestamp'].split('T')[0] + " " + row['timestamp'].split('T')[1][:8]
        
        history_table.add_row(
            game_name,
            f"{start_time}\n{analysis_time}",
            str(row['ml_favorite'] or "N/A"),
            f"{row['ml_imbalance']:.2f}" if row['ml_imbalance'] is not None else "N/A",
            str(row['spread_value'] or "N/A"),
            str(row['spread_favorite'] or "N/A"),
            f"{row['spread_imbalance']:.2f}" if row['spread_imbalance'] is not None else "N/A"
        )
    
    console.print(history_table)
    conn.close()
def view_signal_instances(db_path='orderbook_analyzer.db'):
    """Display records from the signal_instances table"""
    from rich.console import Console
    from rich.table import Table
    import sqlite3
    
    console = Console()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Query signal_instances table
    cursor.execute("""
    SELECT id, game_id, team, signal_type, market, first_seen, last_seen, 
           occurrence_count, strength, is_active
    FROM signal_instances
    ORDER BY id DESC
    LIMIT 20
    """)
    
    rows = cursor.fetchall()
    
    if not rows:
        console.print("[yellow]No records found in signal_instances table[/yellow]")
        conn.close()
        return
    
    table = Table(title="Records from signal_instances Table")
    table.add_column("ID", style="dim")
    table.add_column("Game ID", style="dim", max_width=15)
    table.add_column("Team", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Market", style="magenta")
    table.add_column("Count", style="cyan", justify="right")
    table.add_column("First Seen", style="dim")
    table.add_column("Last Seen", style="dim")
    table.add_column("Active", style="blue")
    
    for row in rows:
        table.add_row(
            str(row['id']),
            row['game_id'][:12] + "...",
            row['team'],
            row['signal_type'],
            row['market'] or "None",
            str(row['occurrence_count']),
            row['first_seen'].split('T')[1][:8] if row['first_seen'] else "N/A",
            row['last_seen'].split('T')[1][:8] if row['last_seen'] else "N/A",
            "Yes" if row['is_active'] else "No"
        )
    
    console.print(table)
    
    # Count by market
    cursor.execute("""
    SELECT market, COUNT(*) as count
    FROM signal_instances
    GROUP BY market
    ORDER BY count DESC
    """)
    
    market_counts = cursor.fetchall()
    
    market_table = Table(title="Market Distribution")
    market_table.add_column("Market", style="magenta")
    market_table.add_column("Count", style="cyan", justify="right")
    
    for row in market_counts:
        market_table.add_row(
            row['market'] or "None",
            str(row['count'])
        )
    
    console.print(market_table)
    
    conn.close()




def sample_data_menu(db_path):
    """Menu for viewing sample data from different tables"""
    console.clear()
    console.print(Panel("Sample Data Menu", style="bold green"))
    console.print("1. games table")
    console.print("2. analysis_results table")
    console.print("3. signals table")
    console.print("4. orderbook_snapshots table")
    console.print("5. orderbook_details table")
    console.print("6. Back to database inspection menu")
    console.print("8. Signal Instances")
    console.print("0. Exit")
    
    choice = Prompt.ask("Enter choice", choices=["0", "1", "2", "3", "4", "5", "6","8"], default="6")
    
    if choice == "1":
        sample_table_data("games", db_path=db_path)
    elif choice == "2":
        sample_table_data("analysis_results", db_path=db_path)
    elif choice == "3":
        sample_table_data("signals", db_path=db_path)
    elif choice == "4":
        sample_table_data("orderbook_snapshots", db_path=db_path)
    elif choice == "5":
        sample_table_data("orderbook_details", db_path=db_path)
    elif choice == "6":
        return "back"
    elif choice == "8":
        view_signal_instances(db_path)
    elif choice == "0":
        return "exit"
    
    console.print()
    input("Press Enter to continue...")
    return None  # Stay in current menu
def signals_query_menu(db_path):
    """Menu for various signal queries"""
    console.clear()
    console.print(Panel("Signals Query Menu", style="bold magenta"))
    console.print("1. Recent signals (last 24 hours)")
    console.print("2. Signals by time period")
    console.print("3. Signals by type")
    console.print("4. Signals by market")
    console.print("5. Signals by team")
    console.print("6. Back to database inspection menu")
    console.print("0. Exit")
    
    choice = Prompt.ask("Enter choice", choices=["0", "1", "2", "3", "4", "5", "6"], default="6")
    
    if choice == "1":
        query_signals(hours=24, db_path=db_path)
    elif choice == "2":
        hours = int(Prompt.ask("Hours to look back", default="48"))
        query_signals(hours=hours, db_path=db_path)
    elif choice == "3":
        signal_type = Prompt.ask("Enter signal type (sharp, unmatched_liquidity, top_book_imbalance)", 
                        choices=["sharp", "unmatched_liquidity", "top_book_imbalance"])
        query_signals_by_type(signal_type, db_path=db_path)
    elif choice == "4":
        market = Prompt.ask("Enter market (Moneyline, Spread)", 
                   choices=["Moneyline", "Spread"])
        query_signals_by_market(market, db_path=db_path)
    elif choice == "5":
        team = Prompt.ask("Enter team abbreviation (e.g., NYK, LAL)")
        query_signals_by_team(team, db_path=db_path)
    elif choice == "6":
        return "back"
    elif choice == "0":
        return "exit"
    
    console.print()
    input("Press Enter to continue...")
    return None

def advanced_query_menu(db_path):
    """Menu for advanced database queries"""
    console.clear()
    console.print(Panel("Advanced Query Menu", style="bold yellow"))
    console.print("1. Most signaled teams(today only)")
    console.print("2. Signal success rate by team")
    console.print("3. Games with most signals(today only)")
    console.print("4. Market activity by day")
    console.print("5. Signal trends over time")
    console.print("6. Back to database inspection menu")
    console.print("0. Exit")
    
    choice = Prompt.ask("Enter choice", choices=["0", "1", "2", "3", "4", "5", "6"], default="6")
    
    if choice == "1":
        query_most_signaled_teams(db_path)
    elif choice == "2":
        # This would require adding outcome data to your database
        console.print("[yellow]Feature not implemented yet. Requires outcome data.[/yellow]")
    elif choice == "3":
        query_games_with_most_signals(db_path)
    elif choice == "4":
        query_market_activity_by_day(db_path)
    elif choice == "5":
        query_signal_trends(db_path)
    elif choice == "6":
        return "back"
    elif choice == "0":
        return "exit"
    
    console.print()
    input("Press Enter to continue...")
    return None

def fix_existing_unknown_markets(db_path='orderbook_analyzer.db'):
    """Fix existing Unknown markets in the signal_instances table"""
    from rich.console import Console
    console = Console()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get count of Unknown markets
        cursor.execute("SELECT COUNT(*) FROM signal_instances WHERE market = 'Unknown'")
        unknown_count = cursor.fetchone()[0]
        
        if unknown_count > 0:
            console.print(f"Found {unknown_count} records with Unknown markets")
            
            # Delete signals with Unknown markets (they're likely Totals)
            cursor.execute("DELETE FROM signal_instances WHERE market = 'Unknown'")
            
            conn.commit()
            console.print(f"Removed {unknown_count} Unknown market records (likely Totals)")
        else:
            console.print("No Unknown market records found")
            
    except Exception as e:
        conn.rollback()
        console.print(f"[bold red]Error fixing unknown markets: {str(e)}[/bold red]")
        
    finally:
        conn.close()

def db_inspection_menu(db_path='orderbook_analyzer.db'):
    """Enhanced database inspection menu with navigation"""
    console.clear()
    console.print(Panel("Database Inspection", style="bold blue"))
    console.print("1. View database structure")
    console.print("2. Count rows in tables")
    console.print("3. View sample data from tables")
    console.print("4. Query signals")
    console.print("5. Advanced queries")
    console.print("6. Search team history")
    console.print("7. View recent analysis results")
    console.print("8. View active signals with persistence")
    console.print("9. Return to main menu")
    console.print("0. Exit")
    console.print('11. Fix existing "Unknown" markets')
    
    choice = Prompt.ask("Enter choice", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8",'9','0','11'], default="8")
    
    if choice == "1":
        inspect_database(db_path)
        console.print()
        input("Press Enter to continue...")
        return None  # Stay in current menu
    elif choice == "2":
        count_table_rows(db_path)
        console.print()
        input("Press Enter to continue...")
        return None
    elif choice == "3":
        return navigate_menus(sample_data_menu, db_path)
    elif choice == "4":
        return navigate_menus(signals_query_menu, db_path)
    elif choice == "5":
        return navigate_menus(advanced_query_menu, db_path)
    elif choice == "6":
        team = Prompt.ask("Enter team abbreviation (e.g., NYK, LAL)")
        query_team_history(team, db_path)
        console.print()
        input("Press Enter to continue...")
        return None
    elif choice == "7":
        days = int(Prompt.ask("Enter number of days to look back", default="7"))
        query_recent_analysis(days, db_path)
        console.print()
        input("Press Enter to continue...")
        return None
    elif choice == "8":
        display_active_signals(db_path)
        console.print()
        input("Press Enter to continue...")
        return None
    elif choice == "9":
        return "main"
    elif choice == "0":
        return "exit"
    elif choice == '11':
        fix_existing_unknown_markets(db_path)
        console.print()
        input("Press Enter to continue...")
        return None

def initialize_signal_tracking(db_path='orderbook_analyzer.db'):
    """Add tables for tracking signal persistence over time"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create signal_instances table to store every occurrence
    c.execute('''CREATE TABLE IF NOT EXISTS signal_instances (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id TEXT NOT NULL,
        team TEXT NOT NULL,
        signal_type TEXT NOT NULL,
        market TEXT NOT NULL,
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        occurrence_count INTEGER NOT NULL DEFAULT 1,
        strength TEXT,
        details TEXT,
        is_active INTEGER NOT NULL DEFAULT 1,
        FOREIGN KEY (game_id) REFERENCES games(game_id)
    )''')
    
    # Create index for faster lookups
    c.execute('CREATE INDEX IF NOT EXISTS idx_signal_game_team ON signal_instances(game_id, team, signal_type, market)')
    
    conn.commit()
    conn.close()
    print("Signal tracking tables initialized")

def is_total_market(market, signal_data):
    """Determine if a signal is for a Total market"""
    # Check the market name
    if market and isinstance(market, str):
        if 'total' in market.lower() or 'over' in market.lower() or 'under' in market.lower():
            return True
    
    # Check signal data
    if isinstance(signal_data, dict):
        # Convert to string and check for total-related terms
        signal_str = str(signal_data).lower()
        if 'total' in signal_str and ('over' in signal_str or 'under' in signal_str):
            return True
        
        # Check specific fields that might indicate a total
        if signal_data.get('Market') == 'Total' or signal_data.get('market') == 'Total':
            return True
            
        # Look for total-specific fields
        if 'total' in signal_data or 'Total' in signal_data:
            return True
    
    return False


def process_signal(cursor, game_id, team, signal_type, market, strength, signal_data, current_time):
    """Process an individual signal instance"""
    global new_signals, updated_signals
    
    # Skip if missing essential data
    if not team:
        return
    
    # Skip Total markets
    if is_total_market(market, signal_data):
        return
    
    # Ensure we have a valid market
    if not market:
        if signal_type == 'unmatched_liquidity':
            market = 'Moneyline'  # Default for unmatched liquidity
        else:
            market = 'Unknown'
    
    # Check if this signal already exists
    cursor.execute('''
    SELECT id, occurrence_count FROM signal_instances 
    WHERE game_id = ? AND team = ? AND signal_type = ? AND market = ? AND is_active = 1
    ''', (game_id, team, signal_type, market))
    
    existing = cursor.fetchone()
    
    if existing:
        # Update existing signal
        signal_id, count = existing
        cursor.execute('''
        UPDATE signal_instances 
        SET last_seen = ?, occurrence_count = ?, strength = ?, details = ?
        WHERE id = ?
        ''', (
            current_time, 
            count + 1, 
            strength,
            json.dumps(signal_data, cls=NumpyEncoder),
            signal_id
        ))
        updated_signals += 1
    else:
        # Insert new signal
        cursor.execute('''
        INSERT INTO signal_instances 
        (game_id, team, signal_type, market, first_seen, last_seen, strength, details, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        ''', (
            game_id,
            team,
            signal_type,
            market,
            current_time,
            current_time,
            strength,
            json.dumps(signal_data, cls=NumpyEncoder)
        ))
        new_signals += 1


def track_signals(all_analyses, db_path='orderbook_analyzer.db'):
    """Track signals over time, excluding Total markets from signal tracking"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Initialize counters in a global scope for this function
    global new_signals, updated_signals
    new_signals = 0
    updated_signals = 0
    
    current_time = datetime.now().isoformat()
    
    try:
        for game_id, analysis in all_analyses.items():
            # Process sharp signals
            sharp_signals = analysis.get('sharp_signals', [])
            for signal in sharp_signals:
                team = signal.get('signal', '')
                market = signal.get('market', '')
                
                # Skip Total markets
                if is_total_market(market, signal):
                    continue
                
                strength = signal.get('strength', '')
                process_signal(c, game_id, team, 'sharp', market, strength, signal, current_time)
            
            # Process unmatched liquidity signals
            unmatched = analysis.get('unmatched_liquidity', {})
            
            # Process moneyline signals
            for signal in unmatched.get('moneyline_signals', []):
                team = signal.get('team', '')
                market = 'Moneyline'
                strength = signal.get('significance', '')
                
                process_signal(c, game_id, team, 'unmatched_liquidity', market, strength, signal, current_time)
            
            # Process spread signals
            for signal in unmatched.get('spread_signals', []):
                team = signal.get('team', '')
                spread_value = signal.get('spread', '')
                market = f"Spread {spread_value}" if spread_value else 'Spread'
                strength = signal.get('significance', '')
                
                process_signal(c, game_id, team, 'unmatched_liquidity', market, strength, signal, current_time)
            
            # Process top book imbalance signals
            top_book = analysis.get('top_book_imbalance', {})
            
            # Process moneyline signals
            for signal in top_book.get('moneyline_signals', []):
                team = signal.get('signal', '')
                market = 'Moneyline'
                strength = signal.get('significance', '')
                
                process_signal(c, game_id, team, 'top_book_imbalance', market, strength, signal, current_time)
            
            # Process spread signals
            for signal in top_book.get('spread_signals', []):
                team = signal.get('signal', '')
                spread_value = signal.get('main_spread', '')
                market = f"Spread {spread_value}" if spread_value else 'Spread'
                strength = signal.get('significance', '')
                
                process_signal(c, game_id, team, 'top_book_imbalance', market, strength, signal, current_time)
                
        # Mark old signals as inactive
        c.execute('''
        UPDATE signal_instances 
        SET is_active = 0 
        WHERE last_seen < datetime('now', '-30 minutes') 
        AND is_active = 1
        ''')
        
        inactive_count = c.rowcount
        
        conn.commit()
        print(f"Signal tracking complete: {new_signals} new, {updated_signals} updated, {inactive_count} marked inactive")
        
    except Exception as e:
        conn.rollback()
        print(f"Error tracking signals: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        conn.close()
def process_signal_group(cursor, game_id, signals, signal_type, current_time):
    """Process a group of signals of the same type"""
    new_signals = 0
    updated_signals = 0
    
    for signal in signals:
        team = signal.get('team') or signal.get('signal', '')
        
        # Fix for unmatched_liquidity market display
        if signal_type == 'unmatched_liquidity':
            # Try to extract market information from different sources
            if 'spread' in signal:
                market = f"Spread {signal.get('spread', '')}"
            elif signal.get('market') == 'Moneyline' or signal.get('side') == 'Moneyline':
                market = 'Moneyline'
            else:
                # Check if it's in a nested structure
                spread_value = signal.get('spread', '')
                if spread_value:
                    market = f"Spread {spread_value}"
                else:
                    # Try to infer from the signal data structure or name
                    signal_data_str = json.dumps(signal).lower()
                    if 'moneyline' in signal_data_str:
                        market = 'Moneyline'
                    elif 'spread' in signal_data_str:
                        market = 'Spread'
                    else:
                        market = 'Unknown'
                        print(f"WARNING: Inferred market '{market}' for {signal_type} signal for team {team}")
        else:
            # For other signal types, get market directly
            market = signal.get('market', '')
        
        strength = signal.get('strength') or signal.get('significance', '')
        
        # Check if this signal already exists
        cursor.execute('''
        SELECT id, occurrence_count FROM signal_instances 
        WHERE game_id = ? AND team = ? AND signal_type = ? AND market = ? AND is_active = 1
        ''', (game_id, team, signal_type, market))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing signal
            signal_id, count = existing
            cursor.execute('''
            UPDATE signal_instances 
            SET last_seen = ?, occurrence_count = ?, strength = ?, details = ?
            WHERE id = ?
            ''', (
                current_time, 
                count + 1, 
                strength,
                json.dumps(signal, cls=NumpyEncoder),  # Use NumpyEncoder to handle NumPy types
                signal_id
            ))
            updated_signals += 1
        else:
            # Insert new signal
            cursor.execute('''
            INSERT INTO signal_instances 
            (game_id, team, signal_type, market, first_seen, last_seen, strength, details, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            ''', (
                game_id,
                team,
                signal_type,
                market,
                current_time,
                current_time,
                strength,
                json.dumps(signal, cls=NumpyEncoder)  # Use NumpyEncoder to handle NumPy types
            ))
            new_signals += 1
    
    return new_signals, updated_signals



def query_active_signals(db_path='orderbook_analyzer.db'):
    """Query all currently active signals with persistence information"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    query = """
    SELECT si.id, si.game_id, si.team, si.signal_type, si.market, 
           si.first_seen, si.last_seen, si.occurrence_count, si.strength,
           g.home_team, g.away_team, g.event_name
    FROM signal_instances si
    JOIN games g ON si.game_id = g.game_id
    WHERE si.is_active = 1
    ORDER BY si.occurrence_count DESC, si.last_seen DESC
    """
    
    c.execute(query)
    results = [dict(row) for row in c.fetchall()]
    conn.close()
    
    return results

def display_active_signals(db_path='orderbook_analyzer.db'):
    """Display all active signals with persistence information"""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    signals = query_active_signals(db_path)
    
    if not signals:
        console.print("[yellow]No active signals found[/yellow]")
        return
    
    table = Table(title="Active Signals with Persistence")
    table.add_column("Game", style="cyan")
    table.add_column("Signal", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Market", style="magenta")
    table.add_column("Persist", justify="right", style="bold red")
    table.add_column("Strength", style="blue")
    table.add_column("First Seen", style="dim")
    table.add_column("Last Seen", style="dim")
    
    for signal in signals:
        # Calculate time difference in minutes
        first_seen = datetime.fromisoformat(signal['first_seen'])
        last_seen = datetime.fromisoformat(signal['last_seen'])
        duration_mins = round((last_seen - first_seen).total_seconds() / 60)
        
        # Format persistence display
        if duration_mins > 0:
            persistence = f"{signal['occurrence_count']}x ({duration_mins}m)"
        else:
            persistence = f"{signal['occurrence_count']}x"
        
        table.add_row(
            f"{signal['away_team']} @ {signal['home_team']}",
            signal['team'],
            signal['signal_type'],
            signal['market'],
            persistence,
            signal['strength'],
            signal['first_seen'].split('T')[1][:8],
            signal['last_seen'].split('T')[1][:8]
        )
    
    console.print(table)





def main():
    """Main execution function with improved navigation"""
    db_path = 'orderbook_analyzer.db'
    ensure_database_initialized(db_path)
    initialize_signal_tracking(db_path)
    
    while True:
        console.clear()
        console.print("=" * 60)
        console.print("ORDERBOOK ANALYZER".center(60))
        console.print("=" * 60)
        console.print("\n1. Run one-time analysis")
        console.print("2. Start continuous monitoring")
        console.print("3. View recent signals from database")
        console.print("4. Inspect database tables")
        console.print("5. Exit")
        
        choice = Prompt.ask("\nEnter your choice", choices=["1", "2", "3", "4", "5"], default="5")
        
        if choice == "1":
            console.print("\nRunning one-time analysis...")
            auth_token = grab_auth_token()
            orderbook_data = scrape_raw_orderbook(auth_token)
            analyses, _ = analyze_all_games(orderbook_data, db_path)
            input("\nPress Enter to continue...")
            
        elif choice == "2":
            console.print("\nStarting continuous monitoring...")
            interval = int(Prompt.ask("Enter check interval in minutes", default="5"))
            duration = int(Prompt.ask("Enter monitoring duration in hours", default="24"))
            continuous_monitoring(interval_minutes=interval, max_hours=duration, db_path=db_path)
            
        elif choice == "3":
            hours = int(Prompt.ask("How many hours back to look", default="24"))
            query_signals(hours=hours, db_path=db_path)
            input("\nPress Enter to continue...")
            
        elif choice == "4":
            # Use navigation system for database inspection
            navigate_menus(db_inspection_menu, db_path)
            
        elif choice == "5":
            console.print("\nExiting...")
            break





if __name__ == "__main__":
    main()