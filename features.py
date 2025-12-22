import numpy as np
import pandas as pd
import pandas_ta as ta

def preprocess_basic(df):
    df = df.sort_index()
    df = df.dropna()
    df = df.drop_duplicates()

    df['feature_close'] = (df['close'] - df['close'].mean()) / df['close'].std()

    return df

def preprocess_basic_bis(df):
    df = df.sort_index()
    df = df.dropna()
    df = df.drop_duplicates()

    # Valeur des différentes features par rapport à la moyenne
    df['feature_close'] = (df['close'] - df['close'].mean()) / df['close'].std()
    df['feature_high'] = (df['high'] - df['high'].mean()) / df['high'].std()
    df['feature_low'] = (df['low'] - df['low'].mean()) / df['low'].std()

    #Valeur des ratios entre 2 des 4 niveaux (haut/bas ou fermeture/ouverture)
    df['feature_ratio_high_low'] = (df['high'] / df['low'])
    df['feature_ratio_close_open'] = (df['close'] / df['open'])
    df['feature_ratio_high_close'] = (df['high'] / df['close'])

    #Valeur des features par rapport à leur max
    df['feature_max_ratio_close'] = (df['close'] - df['close'].max()) / df['close'].max()
    df['feature_max_ratio_open'] = (df['open'] - df['open'].max()) / df['open'].max()
    df['feature_max_ratio_low'] = (df['low'] - df['low'].max()) / df['low'].max()
    df['feature_max_ratio_high'] = (df['high'] - df['high'].max()) / df['high'].max()

    #Valeur des ratios par rapport à leur max entre 2 des 4 niveaux (haut/bas ou fermeture/ouverture)
    df['feature_ratio_max_high_low'] = (df['feature_ratio_high_low'] - df['feature_ratio_high_low'].max()) / df['feature_ratio_high_low'].max()
    df['feature_ratio_max__close_open'] = (df['feature_ratio_close_open'] - df['feature_ratio_close_open'].max()) / df['feature_ratio_close_open'].max()
    df['feature_ratio_max_high_close'] = (df['feature_ratio_high_close'] - df['feature_ratio_high_close'].max()) / df['feature_ratio_high_close'].max()

    return df

def preprocess_finance(df):
    df = df.copy()
    

    df['feature_log_ret'] = np.log(df['close'] / df['close'].shift(1))

    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/14, adjust=False).mean()
    ma_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    
    df['feature_rsi'] = rsi / 100.0 
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    
    df['feature_macd_norm'] = macd_line / df['close']

    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    df['feature_atr_norm'] = atr / df['close']

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
        
    df['feature_hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['feature_hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    df = df.dropna()
    df = df.replace([np.inf, -np.inf], 0)

    return df

def preprocess_finance_other(df):
    df = df.copy()
    
    df['feature_log_ret'] = np.log(df['close'] / df['close'].shift(1))

    df['feature_rsi'] = ta.rsi(df['close'], length=14) / 100.0 
    
    macd = ta.macd(df['close'])
    df['feature_macd_norm'] = macd['MACD_12_26_9'] / df['close']
    for i in range(1, 6):
        df[f'feature_log_ret_lag_{i}'] = df['feature_log_ret'].shift(i)

    df['feature_atr_norm'] = ta.atr(df['high'], df['low'], df['close'], length=14) / df['close']
    
    df['feature_hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['feature_hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    df = df.dropna().replace([np.inf, -np.inf], 0)
    return df

def preprocess_finance_finale(df):
    df["log_ret"] = np.log(df["close"]).diff()

    # Tendance
    df.ta.macd(append=True)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    
    # Calcul de distance par rapport aux EMA (normalisation)
    df["dist_ema20"] = (df["close"] - df["EMA_20"]) / df["EMA_20"]
    
    # Oscillateurs / Momentum
    df.ta.rsi(length=14, append=True)
    df["RSI_14"] = df["RSI_14"] / 100.0 

    # Volatilité
    df.ta.atr(length=14, append=True)
    df["ATR_14_norm"] = df["ATRr_14"] / df["close"] 

    # Nettoyage
    df.dropna(inplace=True) 
    return df

def preprocess_dqn(df):
    df = df.copy()
    df = df.sort_index()
    
    # Price Position
    # Bollinger band (price according to min and max)
    sma = df["close"].rolling(window=20).mean()
    std = df["close"].rolling(window=20).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    df["feature_bb_pos"] = (df["close"] - lower) / (upper - lower) # Position (0 = low, 1 = high, >1 = higher than before)
    df["feature_bb_width"] = (upper - lower) / df["close"]
    
    # Trend
    # Distance to Moving Average 20
    df["ma_20"] = df["close"].rolling(window=20).mean()
    df["feature_dist_ma_20"] = df["close"] / df["ma_20"] - 1
    # Distance to Moving Average 50
    df["ma_50"] = df["close"].rolling(window=50).mean()
    df["feature_dist_ma_50"] = df["close"] / df["ma_50"] - 1
    # Distance to Moving Average 200
    df["ma_200"] = df["close"].rolling(window=200).mean()
    df["feature_dist_ma_200"] = df["close"] / df["ma_200"] - 1
    
    
    # Trend speed
    df["feature_log_ret"] = np.log(df["close"] / df["close"].shift(1))
    
    # Trend Strength
    # ADX Simplified
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(14).mean() + 1e-10
    # Directional Movement
    up_move = df["high"] - df["high"].shift()
    down_move = df["low"].shift() - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr)
    # DX et ADX
    sum_di = plus_di + minus_di + 1e-10
    dx = 100 * np.abs(plus_di - minus_di) / sum_di
    df["feature_adx"] = dx.rolling(14).mean() / 100.0

    # Volatility
    df["feature_std_deviation"] = df["close"].rolling(window=20).std() / df["close"]
    df["feature_candle_range"] = (df["high"] - df["low"]) / df["close"]

    # Momentum
    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["feature_rsi_norm"] = df["rsi"] / 100.0
    
    # Volume
    # Test if Volume == 0
    if "volume" not in df.columns or (df["volume"] == 0).all():
        df["volume"] = 1.0 # Value with no impact to prevent crash
        has_volume = False
    else:
        df["volume"] = df["volume"].replace(0, 1e-5) # Replace 0 with low values
        has_volume = True
    # Calculations
    if has_volume:
        # Volume Relative Strength
        df["vol_ma_20"] = df["volume"].rolling(window=20).mean()
        df["feature_vol_rel"] = (df["volume"] / df["vol_ma_20"] - 1).fillna(0)
        # OBV Slope
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        df["feature_obv_slope"] = df["obv"].pct_change(5).fillna(0)
    else:
        df["feature_vol_rel"] = 0.0
        df["feature_obv_slope"] = 0.0
        
    # Clean up
    df.dropna(inplace=True) # Remove NaN
    df.replace([np.inf, -np.inf], 0, inplace=True) # Remove infinite values due to calculations
    
    return df

