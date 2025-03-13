import requests
import pandas as pd
import config
import datetime
from datetime import datetime
import json
import numpy as np



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
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




from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.box import ROUNDED
from datetime import datetime
import time

console = Console()

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
    
    for table_name in tables:
        table_name = table_name[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        for i, col in enumerate(columns):
            table_info.add_row(
                table_name if i == 0 else "",
                col[1],  # Column name
                col[2]   # Data type
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
        # Read data into DataFrame for easy display
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
        
        if len(df) == 0:
            console.print(f"[yellow]No data found in table: {table_name}[/yellow]")
            return
        
        # Expand any JSON columns
        json_cols = []
        for col in df.columns:
            # Try to parse the first non-null value as JSON
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(sample, str) and sample.startswith('{') and sample.endswith('}'):
                try:
                    json.loads(sample)
                    json_cols.append(col)
                except:
                    pass
        
        # Print basic table info
        console.print(f"[bold cyan]Sample data from table: {table_name}[/bold cyan]")
        console.print(f"Columns: {', '.join(df.columns.tolist())}")
        console.print(f"Rows displayed: {len(df)} of {limit} requested")
        
        # Print DataFrame
        console.print("\n[bold]Data sample:[/bold]")
        console.print(df)
        
        # Pretty print any JSON columns
        for col in json_cols:
            console.print(f"\n[bold]Contents of JSON column: {col}[/bold]")
            sample_row = df.iloc[0]
            json_data = json.loads(sample_row[col])
            console.print(json.dumps(json_data, indent=2))
    
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


def main():
    """Main execution function"""
    db_path = 'orderbook_analyzer.db'
    
    print("=" * 60)
    print("ORDERBOOK ANALYZER".center(60))
    print("=" * 60)
    print("\n1. Run one-time analysis")
    print("2. Start continuous monitoring")
    print("3. View recent signals from database")
    print("4. Inspect database tables")  # New option
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        print("\nRunning one-time analysis...")
        auth_token = grab_auth_token()
        orderbook_data = scrape_raw_orderbook(auth_token)
        analyses, _ = analyze_all_games(orderbook_data, db_path)
        input("\nPress Enter to continue...")
        main()
        
    elif choice == '2':
        print("\nStarting continuous monitoring...")
        interval = int(input("Enter check interval in minutes (default: 5): ") or "5")
        duration = int(input("Enter monitoring duration in hours (default: 24): ") or "24")
        continuous_monitoring(interval_minutes=interval, max_hours=duration, db_path=db_path)
        main()
        
    elif choice == '3':
        print("\nViewing recent signals...")
        hours = int(input("How many hours back to look (default: 24): ") or "24")
        signals = get_recent_signals(hours=hours, db_path=db_path)
        
        if signals:
            signal_table = Table(title=f"Recent Signals (Last {hours} hours)")
            signal_table.add_column("Game", style="cyan")
            signal_table.add_column("Type", style="yellow")
            signal_table.add_column("Market", style="magenta")
            signal_table.add_column("Team", style="green")
            signal_table.add_column("Strength", style="blue")
            signal_table.add_column("Time", style="dim")
            
            for signal in signals:
                signal_table.add_row(
                    f"{signal['away_team']} @ {signal['home_team']}",
                    signal['signal_type'],
                    signal['market'],
                    signal['team'],
                    signal['strength'],
                    signal['timestamp'].split('T')[1].split('.')[0]  # Format time
                )
            
            console.print(signal_table)
        else:
            print("No signals found in the specified time period.")
        
        input("\nPress Enter to continue...")
        main()
        
    elif choice == '4':
        print("\nDatabase inspection:")
        print("1. View database structure")
        print("2. Count rows in tables")
        print("3. View sample data")
        print("4. Query signals")
        print("5. Search team history")
        print("6. Return to main menu")
        
        db_choice = input("\nEnter choice (1-6): ")
        
        if db_choice == '1':
            inspect_database(db_path)
        elif db_choice == '2':
            count_table_rows(db_path)
        elif db_choice == '3':
            table = input("Enter table name: ")
            sample_table_data(table, db_path=db_path)
        elif db_choice == '4':
            hours = int(input("Hours to look back: ") or "24")
            query_signals(hours, db_path)
        elif db_choice == '5':
            team = input("Enter team abbreviation: ")
            query_game_history(team, db_path)
        
        input("\nPress Enter to continue...")
        main()
    elif choice == '5':
        print("\nExiting program...")
        return
    
    else:
        print("\nInvalid choice. Please try again.")
        main()




if __name__ == "__main__":
    main()