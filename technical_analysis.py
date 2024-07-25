from scipy.signal import find_peaks
import pandas as pd

class Technical:
    def __init__(self, data):
        self.data = data
    
    """
        Calculating RSI.
        formula: RSI = 100 - (100/1+RS) Where RS = Average Gain / Average Loss
    """
    def calculate_rsi(self, window):
        delta = self.data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    """
        Calculating Stochastic.
        The stochastic oscillator is calculated by subtracting
        the low for the period from the current closing price,
        dividing by the total range for the period, and multiplying by 100
    """
    def calculate_stochastic(self, k_window=14, d_window=3):
        low_min = self.data['Low'].rolling(window=k_window).min()
        high_max = self.data['High'].rolling(window=k_window).max()
        k = 100 * ((self.data['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_window).mean()
        return k, d
    
    """
        Calculate Moving Average.
        adding up all the data points during a specific
        period and dividing the sum by the number of time periods
    """
    def calculate_ma(self, window):
        return self.data["Close"].rolling(window=window).mean()
    
    """
        Calculate ADX.
        ADX = 100 times the smoothed moving average of the absolute value of (+DI − -DI) divided by (+DI + -DI)
    """
    def calculate_adx(self, window):
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        plus_dm = high.diff()
        minus_dm = low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1, join='inner').max(axis=1)

        atr = tr.rolling(window=window).mean()
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = abs(100 * (minus_dm.rolling(window=window).mean() / atr))

        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=window).mean()
        return adx, atr

    """
        Calcualte Moving Average Convergence and Divergence.
        Subtracting the long-term EMA (26 periods) from the short-term EMA (12 periods)
    """
    def calculate_macd(self, short_window=12, long_window=26, signal_window=9):
        short_ema = self.data['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = self.data['Close'].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    """
        Calculate Momentum
        the difference between the latest closing price (C)
        and the closing price “n” days ago (Cn) or as the ratio
        of the latest closing price (C) to the closing price “n”
        days ago (Cn) multiplied by 100
    """
    def calculate_momentum(self, window):
        return self.data['Close'] - self.data['Close'].shift(window)

    """
        Calculate Squeeze Momentm

        Calculate the Bollinger Bands:
            Middle Band (MB): 20-period simple moving average (SMA) of the closing prices.
            Upper Band (UB): MB + (2 * standard deviation of the last 20 closing prices).
            Lower Band (LB): MB - (2 * standard deviation of the last 20 closing prices).

        Calculate the Keltner Channels:
            Middle Line (ML): 20-period exponential moving average (EMA) of the closing prices.
            Upper Channel (UC): ML + (1.5 * Average True Range (ATR) of the last 20 periods).
            Lower Channel (LC): ML - (1.5 * ATR of the last 20 periods).
    """
    def calculate_squeeze_momentum(self, window=20, bb_mult=2, kc_mult=1.5):
        # Calculate Bollinger Bands
        sma = self.data['Close'].rolling(window=window).mean()
        std = self.data['Close'].rolling(window=window).std()
        bb_upper = sma + bb_mult * std
        bb_lower = sma - bb_mult * std

        # Calculate Keltner Channels
        tr = pd.concat([self.data['High'] - self.data['Low'],
                        abs(self.data['High'] - self.data['Close'].shift()),
                        abs(self.data['Low'] - self.data['Close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        kc_upper = sma + kc_mult * atr
        kc_lower = sma - kc_mult * atr

        # Calculate Squeeze
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        squeeze_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)

        # Calculate Momentum
        momentum = self.data['Close'] - sma
        return squeeze_on.astype(int) - squeeze_off.astype(int), momentum

    """
        Detecting Double Top.
    """
    def detect_double_top(self, window=14):
        double_top_signals = []
        for i in range(window, len(self.data) - window):
            window_data = self.data['Close'][i-window:i+window]
            peaks, _ = find_peaks(window_data)
            if len(peaks) >= 2:
                peaks = peaks[-2:]
                first_peak = window_data.iloc[peaks[0]]
                second_peak = window_data.iloc[peaks[1]]
                trough = window_data.iloc[peaks[0]:peaks[1]].min()
                if abs(first_peak - second_peak) < 0.05 * first_peak and (first_peak - trough) > 0.1 * first_peak:
                    double_top_signals.append(1)
                else:
                    double_top_signals.append(0)
            else:
                double_top_signals.append(0)
        double_top_signals = [0] * window + double_top_signals + [0] * window
        return pd.Series(double_top_signals, index=self.data.index)

    """
        Detecting Double Bottom.
    """
    def detect_double_bottom(self, window=14):
        double_bottom_signals = []
        for i in range(window, len(self.data) - window):
            window_data = self.data['Close'][i-window:i+window]
            troughs, _ = find_peaks(-window_data)
            if len(troughs) >= 2:
                troughs = troughs[-2:]
                first_trough = window_data.iloc[troughs[0]]
                second_trough = window_data.iloc[troughs[1]]
                peak = window_data.iloc[troughs[0]:troughs[1]].max()
                if abs(first_trough - second_trough) < 0.05 * first_trough and (peak - first_trough) > 0.1 * peak:
                    double_bottom_signals.append(1)
                else:
                    double_bottom_signals.append(0)
            else:
                double_bottom_signals.append(0)
        double_bottom_signals = [0] * window + double_bottom_signals + [0] * window
        return pd.Series(double_bottom_signals, index=self.data.index)

    """
        Detect Wyckoff Accumulation
    """
    def detect_wyckoff_accumulation(self, window=100):
        accumulation_signals = []
        for i in range(window, len(self.data) - window):
            window_data = self.data['Close'][i-window:i+window]
            troughs, _ = find_peaks(-window_data)
            peaks, _ = find_peaks(window_data)
            if len(troughs) >= 2 and len(peaks) >= 1:
                first_trough = window_data.iloc[troughs[0]]
                second_trough = window_data.iloc[troughs[1]]
                peak = window_data.iloc[troughs[0]:troughs[1]].max()
                if abs(first_trough - second_trough) < 0.05 * first_trough and (peak - first_trough) > 0.1 * peak:
                    accumulation_signals.append(1)
                else:
                    accumulation_signals.append(0)
            else:
                accumulation_signals.append(0)
        accumulation_signals = [0] * window + accumulation_signals + [0] * window
        return pd.Series(accumulation_signals, index=self.data.index)

    """
        Detect Wyckoff Distribution.
    """
    def detect_wyckoff_distribution(self, window=100):
        distribution_signals = []
        for i in range(window, len(self.data) - window):
            window_data = self.data['Close'][i-window:i+window]
            peaks, _ = find_peaks(window_data)
            troughs, _ = find_peaks(-window_data)
            if len(peaks) >= 2 and len(troughs) >= 1:
                first_peak = window_data.iloc[peaks[0]]
                second_peak = window_data.iloc[peaks[1]]
                trough = window_data.iloc[peaks[0]:peaks[1]].min()
                if abs(first_peak - second_peak) < 0.05 * first_peak and (first_peak - trough) > 0.1 * first_peak:
                    distribution_signals.append(1)
                else:
                    distribution_signals.append(0)
            else:
                distribution_signals.append(0)
        distribution_signals = [0] * window + distribution_signals + [0] * window
        return pd.Series(distribution_signals, index=self.data.index)

