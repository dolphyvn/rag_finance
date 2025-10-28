import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGDataConverter:
    def __init__(self):
        """Initialize the RAG data converter"""
        self.setup_patterns = {
            'BREAKOUT': self._identify_breakout_setup,
            'REVERSAL': self._identify_reversal_setup,
            'TREND_CONTINUATION': self._identify_trend_continuation_setup,
            'RANGE_BOUND': self._identify_range_bound_setup,
            'SUPPORT_TEST': self._identify_support_test_setup,
            'RESISTANCE_TEST': self._identify_resistance_test_setup
        }

    def _calculate_trade_outcome(self, df, entry_index, direction, profit_target_points=5.0, stop_loss_points=3.0):
        """
        Calculate the outcome of a potential trade

        Args:
            df: DataFrame with price data
            entry_index: Index where trade would enter
            direction: 'LONG' or 'SHORT'
            profit_target_points: Target profit in points
            stop_loss_points: Stop loss in points

        Returns:
            dict with outcome details
        """
        if entry_index >= len(df) - 10:  # Need future data
            return None

        entry_price = df.iloc[entry_index]['close']
        entry_time = df.iloc[entry_index]['timestamp']

        if direction == 'LONG':
            profit_target = entry_price + profit_target_points
            stop_loss = entry_price - stop_loss_points
        else:  # SHORT
            profit_target = entry_price - profit_target_points
            stop_loss = entry_price + stop_loss_points

        # Check future candles for outcome
        future_df = df.iloc[entry_index + 1:entry_index + 51]  # Next 50 candles max

        if len(future_df) == 0:
            return None

        # Find which level was hit first
        for i, (_, candle) in enumerate(future_df.iterrows()):
            high = candle['high']
            low = candle['low']

            if direction == 'LONG':
                if high >= profit_target:
                    # Profit target hit
                    exit_price = profit_target
                    profit = profit_target_points
                    exit_time = candle['timestamp']
                    duration_minutes = i + 1
                    return {
                        'outcome': 'PROFIT_TARGET',
                        'profit': profit,
                        'exit_price': exit_price,
                        'exit_time': exit_time,
                        'duration_minutes': duration_minutes,
                        'success': True
                    }
                elif low <= stop_loss:
                    # Stop loss hit
                    exit_price = stop_loss
                    profit = -stop_loss_points
                    exit_time = candle['timestamp']
                    duration_minutes = i + 1
                    return {
                        'outcome': 'STOP_LOSS',
                        'profit': profit,
                        'exit_price': exit_price,
                        'exit_time': exit_time,
                        'duration_minutes': duration_minutes,
                        'success': False
                    }
            else:  # SHORT
                if low <= profit_target:
                    # Profit target hit
                    exit_price = profit_target
                    profit = profit_target_points
                    exit_time = candle['timestamp']
                    duration_minutes = i + 1
                    return {
                        'outcome': 'PROFIT_TARGET',
                        'profit': profit,
                        'exit_price': exit_price,
                        'exit_time': exit_time,
                        'duration_minutes': duration_minutes,
                        'success': True
                    }
                elif high >= stop_loss:
                    # Stop loss hit
                    exit_price = stop_loss
                    profit = -stop_loss_points
                    exit_time = candle['timestamp']
                    duration_minutes = i + 1
                    return {
                        'outcome': 'STOP_LOSS',
                        'profit': profit,
                        'exit_price': exit_price,
                        'exit_time': exit_time,
                        'duration_minutes': duration_minutes,
                        'success': False
                    }

        # Neither hit within 50 candles
        return None

    def _identify_breakout_setup(self, df, index):
        """Identify breakout setups"""
        if index < 20 or index >= len(df) - 50:
            return None

        current = df.iloc[index]
        previous = df.iloc[index-20:index]

        # Volume spike
        avg_volume = previous['volume'].mean()
        volume_spike = current['volume'] > avg_volume * 2.0

        # Break above resistance
        resistance = previous['high'].max()
        breakout = current['close'] > resistance

        if volume_spike and breakout and current['trend'] == 'BULLISH':
            return {
                'direction': 'LONG',
                'confidence': 0.8,
                'entry': current['close'],
                'stop_loss': current['close'] - (2 * current['atr_14']),
                'take_profit': current['close'] + (3 * current['atr_14']),
                'reasoning': f"Volume spike ({current['volume']:.0f} vs {avg_volume:.0f} avg) and break above resistance ({resistance:.2f})"
            }

        # Break below support
        support = previous['low'].min()
        breakdown = current['close'] < support

        if volume_spike and breakdown and current['trend'] == 'BEARISH':
            return {
                'direction': 'SHORT',
                'confidence': 0.8,
                'entry': current['close'],
                'stop_loss': current['close'] + (2 * current['atr_14']),
                'take_profit': current['close'] - (3 * current['atr_14']),
                'reasoning': f"Volume spike ({current['volume']:.0f} vs {avg_volume:.0f} avg) and break below support ({support:.2f})"
            }

        return None

    def _identify_reversal_setup(self, df, index):
        """Identify reversal setups"""
        if index < 20 or index >= len(df) - 50:
            return None

        current = df.iloc[index]
        previous = df.iloc[index-10:index]

        # RSI divergence
        rsi_trend = 'rising' if previous['rsi_14'].is_monotonic_increasing else 'falling' if previous['rsi_14'].is_monotonic_decreasing else 'neutral'
        price_trend = 'rising' if previous['close'].is_monotonic_increasing else 'falling' if previous['close'].is_monotonic_decreasing else 'neutral'

        # Bullish divergence: price falling, RSI rising
        if price_trend == 'falling' and rsi_trend == 'rising' and current['rsi_14'] < 30:
            return {
                'direction': 'LONG',
                'confidence': 0.7,
                'entry': current['close'],
                'stop_loss': current['close'] - (1.5 * current['atr_14']),
                'take_profit': current['close'] + (2.5 * current['atr_14']),
                'reasoning': f"Bullish RSI divergence ({current['rsi_14']:.1f}), oversold conditions"
            }

        # Bearish divergence: price rising, RSI falling
        if price_trend == 'rising' and rsi_trend == 'falling' and current['rsi_14'] > 70:
            return {
                'direction': 'SHORT',
                'confidence': 0.7,
                'entry': current['close'],
                'stop_loss': current['close'] + (1.5 * current['atr_14']),
                'take_profit': current['close'] - (2.5 * current['atr_14']),
                'reasoning': f"Bearish RSI divergence ({current['rsi_14']:.1f}), overbought conditions"
            }

        return None

    def _identify_trend_continuation_setup(self, df, index):
        """Identify trend continuation setups"""
        if index < 20 or index >= len(df) - 50:
            return None

        current = df.iloc[index]
        previous = df.iloc[index-20:index]

        # Strong trend in previous 20 candles
        prev_trend = previous['trend'].mode().iloc[0] if not previous['trend'].mode().empty else 'NEUTRAL'

        if prev_trend == 'BULLISH' and current['trend'] == 'BULLISH':
            # Pullback to EMA20
            if current['close'] > current['ema_20'] and current['low'] <= current['ema_20']:
                return {
                    'direction': 'LONG',
                    'confidence': 0.75,
                    'entry': current['close'],
                    'stop_loss': current['ema_20'] - (current['atr_14']),
                    'take_profit': current['close'] + (2 * current['atr_14']),
                    'reasoning': f"Bullish trend continuation, pullback to EMA20 ({current['ema_20']:.2f})"
                }

        elif prev_trend == 'BEARISH' and current['trend'] == 'BEARISH':
            # Pullback to EMA20
            if current['close'] < current['ema_20'] and current['high'] >= current['ema_20']:
                return {
                    'direction': 'SHORT',
                    'confidence': 0.75,
                    'entry': current['close'],
                    'stop_loss': current['ema_20'] + (current['atr_14']),
                    'take_profit': current['close'] - (2 * current['atr_14']),
                    'reasoning': f"Bearish trend continuation, pullback to EMA20 ({current['ema_20']:.2f})"
                }

        return None

    def _identify_range_bound_setup(self, df, index):
        """Identify range-bound setups"""
        if index < 20 or index >= len(df) - 50:
            return None

        current = df.iloc[index]
        previous = df.iloc[index-20:index]

        # Check if in range (low volatility)
        range_size = previous['high'].max() - previous['low'].min()
        atr_avg = previous['atr_14'].mean()

        if range_size < 3 * atr_avg:  # Tight range
            upper_bound = previous['high'].max()
            lower_bound = previous['low'].min()

            # At bottom of range, go long
            if current['close'] <= lower_bound + (0.1 * range_size):
                return {
                    'direction': 'LONG',
                    'confidence': 0.6,
                    'entry': current['close'],
                    'stop_loss': lower_bound - current['atr_14'],
                    'take_profit': upper_bound - current['atr_14'],
                    'reasoning': f"Range bound setup, buying at bottom of range ({lower_bound:.2f} - {upper_bound:.2f})"
                }

            # At top of range, go short
            elif current['close'] >= upper_bound - (0.1 * range_size):
                return {
                    'direction': 'SHORT',
                    'confidence': 0.6,
                    'entry': current['close'],
                    'stop_loss': upper_bound + current['atr_14'],
                    'take_profit': lower_bound + current['atr_14'],
                    'reasoning': f"Range bound setup, selling at top of range ({lower_bound:.2f} - {upper_bound:.2f})"
                }

        return None

    def _identify_support_test_setup(self, df, index):
        """Identify support test setups"""
        if index < 20 or index >= len(df) - 50:
            return None

        current = df.iloc[index]

        if current['support_1'] and current['low'] <= current['support_1'] + current['atr_14']:
            # Price testing support level
            if current['close'] > current['support_1']:  # Bounced off support
                return {
                    'direction': 'LONG',
                    'confidence': 0.7,
                    'entry': current['close'],
                    'stop_loss': current['support_1'] - current['atr_14'],
                    'take_profit': current['close'] + (2 * current['atr_14']),
                    'reasoning': f"Support test at {current['support_1']:.2f}, price bounced"
                }

        return None

    def _identify_resistance_test_setup(self, df, index):
        """Identify resistance test setups"""
        if index < 20 or index >= len(df) - 50:
            return None

        current = df.iloc[index]

        if current['resistance_1'] and current['high'] >= current['resistance_1'] - current['atr_14']:
            # Price testing resistance level
            if current['close'] < current['resistance_1']:  # Rejected from resistance
                return {
                    'direction': 'SHORT',
                    'confidence': 0.7,
                    'entry': current['close'],
                    'stop_loss': current['resistance_1'] + current['atr_14'],
                    'take_profit': current['close'] - (2 * current['atr_14']),
                    'reasoning': f"Resistance test at {current['resistance_1']:.2f}, price rejected"
                }

        return None

    def create_market_context(self, df, index):
        """Create rich market context for RAG"""
        current = df.iloc[index]
        previous = df.iloc[max(0, index-20):index]

        # Volume analysis
        volume_ratio = current['volume'] / previous['volume'].mean() if len(previous) > 0 and previous['volume'].mean() > 0 else 1.0

        # Volatility
        volatility = current['atr_14'] / current['close'] * 100  # ATR as percentage

        # Price position relative to EMAs
        price_vs_ema20 = (current['close'] - current['ema_20']) / current['atr_14']
        price_vs_ema50 = (current['close'] - current['ema_50']) / current['atr_14']

        context = f"""
**Market Snapshot - {current['timestamp']}**
Symbol: XAUUSD | Price: {current['close']:.2f} | Session: {current['session']}

**Price Action:**
- OHLC: O:{current['open']:.2f} H:{current['high']:.2f} L:{current['low']:.2f} C:{current['close']:.2f}
- Range: {current['high'] - current['low']:.2f} points
- Change: {current['close'] - current['open']:+.2f} ({((current['close'] - current['open']) / current['open'] * 100):+.1f}%)

**Technical Indicators:**
- RSI(14): {current['rsi_14']:.1f} ({'Overbought' if current['rsi_14'] > 70 else 'Oversold' if current['rsi_14'] < 30 else 'Neutral'})
- MACD: {current['macd']:.4f} | Signal: {current['macd_signal']:.4f} | Histogram: {current['macd_hist']:.4f}
- EMAs: 20:{current['ema_20']:.2f} 50:{current['ema_50']:.2f} 200:{current['ema_200']:.2f}
- ATR(14): {current['atr_14']:.2f} (Volatility: {volatility:.1f}%)
- Bollinger Bands: U:{current['bb_upper']:.2f} M:{current['bb_middle']:.2f} L:{current['bb_lower']:.2f}

**Volume Analysis:**
- Current: {current['volume']:,} | Average: {current['volume_avg']:,}
- Volume Ratio: {volume_ratio:.1f}x ({'High' if volume_ratio > 2.0 else 'Normal' if volume_ratio > 0.5 else 'Low'})

**Market Structure:**
- Trend: {current['trend']}
- Support: {current['support_1']:.2f if current['support_1'] else 'N/A'}
- Resistance: {current['resistance_1']:.2f if current['resistance_1'] else 'N/A'}
- Price vs EMA20: {price_vs_ema20:+.1f} ATR
- Price vs EMA50: {price_vs_ema50:+.1f} ATR

**Session Context:**
- Time: {current['timestamp'].strftime('%H:%M')} | Day: {current['day_of_week']}
- Session: {current['session']}
"""

        return context.strip()

    def convert_csv_to_rag(self, csv_file, output_file, min_profit_points=5.0):
        """
        Convert CSV data to RAG training format

        Args:
            csv_file: Input CSV file path
            output_file: Output JSON file path
            min_profit_points: Minimum profit to consider as successful trade
        """
        logger.info(f"Converting {csv_file} to RAG format...")

        # Load CSV data
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        logger.info(f"Loaded {len(df)} rows from {csv_file}")

        rag_examples = []
        processed_count = 0

        # Process each candle
        for i in range(200, len(df) - 50):  # Need history and future data
            try:
                # Try to identify different setup types
                setup_found = False
                for setup_type, setup_func in self.setup_patterns.items():
                    setup = setup_func(df, i)
                    if setup:
                        # Calculate actual outcome
                        outcome = self._calculate_trade_outcome(
                            df, i, setup['direction'],
                            profit_target_points=min_profit_points,
                            stop_loss_points=3.0
                        )

                        if outcome and outcome['success']:
                            # Create rich market context
                            context = self.create_market_context(df, i)

                            # Calculate DCA levels
                            atr = df.iloc[i]['atr_14']
                            if setup['direction'] == 'LONG':
                                dca_levels = [
                                    setup['entry'] - (0.5 * atr),
                                    setup['entry'] - (1.0 * atr),
                                    setup['entry'] - (1.5 * atr)
                                ]
                                tp_levels = [
                                    setup['entry'] + atr,
                                    setup['entry'] + (2 * atr),
                                    setup['entry'] + (3 * atr)
                                ]
                            else:  # SHORT
                                dca_levels = [
                                    setup['entry'] + (0.5 * atr),
                                    setup['entry'] + (1.0 * atr),
                                    setup['entry'] + (1.5 * atr)
                                ]
                                tp_levels = [
                                    setup['entry'] - atr,
                                    setup['entry'] - (2 * atr),
                                    setup['entry'] - (3 * atr)
                                ]

                            # Create RAG example
                            rag_example = {
                                'id': f"trade_{setup_type.lower()}_{i}_{int(datetime.now().timestamp())}",
                                'timestamp': df.iloc[i]['timestamp'].isoformat(),
                                'symbol': 'XAUUSD',
                                'price': float(setup['entry']),
                                'setup_type': setup_type,
                                'direction': setup['direction'],
                                'confidence': setup['confidence'],
                                'context': context,
                                'entry_reason': setup['reasoning'],
                                'entry': float(setup['entry']),
                                'dca_levels': [float(level) for level in dca_levels],
                                'stop_loss': float(setup['stop_loss']),
                                'take_profit': [float(level) for level in tp_levels],
                                'actual_outcome': outcome['outcome'],
                                'actual_profit': float(outcome['profit']),
                                'success': True,
                                'duration_minutes': outcome['duration_minutes'],
                                'risk_reward_ratio': abs(outcome['profit']) / 3.0,  # Assuming 3 point stop loss
                                'tags': [setup_type.lower(), setup['direction'].lower(), df.iloc[i]['session'].lower(), 'successful']
                            }

                            rag_examples.append(rag_example)
                            setup_found = True
                            break

                if setup_found:
                    processed_count += 1
                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count} successful trades...")

            except Exception as e:
                logger.warning(f"Error processing index {i}: {e}")
                continue

        # Save RAG examples
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(rag_examples, f, indent=2)

        logger.info(f"Successfully created {len(rag_examples)} RAG examples from {csv_file}")
        logger.info(f"Output saved to {output_file}")

        return rag_examples

    def create_sample_trades(self, csv_file, output_file, num_samples=50):
        """
        Create sample RAG examples for testing (without outcome verification)

        Args:
            csv_file: Input CSV file path
            output_file: Output JSON file path
            num_samples: Number of samples to create
        """
        logger.info(f"Creating {num_samples} sample RAG examples from {csv_file}...")

        # Load CSV data
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        rag_examples = []
        sample_indices = np.random.choice(range(200, len(df) - 50), size=min(num_samples, len(df) - 250), replace=False)

        for i in sample_indices:
            try:
                # Create context
                context = self.create_market_context(df, i)

                current = df.iloc[i]
                atr = current['atr_14']

                # Randomly choose setup type
                setup_types = ['BREAKOUT', 'REVERSAL', 'TREND_CONTINUATION', 'SUPPORT_TEST', 'RESISTANCE_TEST']
                setup_type = np.random.choice(setup_types)

                # Determine direction based on trend
                direction = 'LONG' if current['trend'] == 'BULLISH' else 'SHORT' if current['trend'] == 'BEARISH' else np.random.choice(['LONG', 'SHORT'])

                # Calculate levels
                if direction == 'LONG':
                    entry = current['close']
                    dca_levels = [entry - (0.5 * atr), entry - (1.0 * atr), entry - (1.5 * atr)]
                    stop_loss = entry - (2 * atr)
                    tp_levels = [entry + atr, entry + (2 * atr), entry + (3 * atr)]
                else:
                    entry = current['close']
                    dca_levels = [entry + (0.5 * atr), entry + (1.0 * atr), entry + (1.5 * atr)]
                    stop_loss = entry + (2 * atr)
                    tp_levels = [entry - atr, entry - (2 * atr), entry - (3 * atr)]

                rag_example = {
                    'id': f"sample_{setup_type.lower()}_{i}_{int(datetime.now().timestamp())}",
                    'timestamp': current['timestamp'].isoformat(),
                    'symbol': 'XAUUSD',
                    'price': float(entry),
                    'setup_type': setup_type,
                    'direction': direction,
                    'confidence': round(np.random.uniform(0.6, 0.9), 2),
                    'context': context,
                    'entry_reason': f"Sample {setup_type} setup for testing",
                    'entry': float(entry),
                    'dca_levels': [float(level) for level in dca_levels],
                    'stop_loss': float(stop_loss),
                    'take_profit': [float(level) for level in tp_levels],
                    'actual_outcome': 'UNKNOWN',
                    'actual_profit': None,
                    'success': None,
                    'duration_minutes': None,
                    'risk_reward_ratio': None,
                    'tags': [setup_type.lower(), direction.lower(), current['session'].lower(), 'sample']
                }

                rag_examples.append(rag_example)

            except Exception as e:
                logger.warning(f"Error creating sample at index {i}: {e}")
                continue

        # Save samples
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(rag_examples, f, indent=2)

        logger.info(f"Successfully created {len(rag_examples)} sample RAG examples")
        logger.info(f"Output saved to {output_file}")

        return rag_examples

def main():
    """Main function"""
    # Initialize converter
    converter = RAGDataConverter()

    # Input and output directories
    data_dir = 'data'
    output_dir = 'data'

    # Process M5 data (highest granularity)
    m5_file = os.path.join(data_dir, 'xauusd_training_data_m5.csv')

    if os.path.exists(m5_file):
        print(f"Processing {m5_file}...")

        # Create verified trades (with actual outcomes)
        verified_output = os.path.join(output_dir, 'xauusd_rag_verified.json')
        verified_trades = converter.convert_csv_to_rag(m5_file, verified_output, min_profit_points=5.0)

        # Create sample trades (for testing, no outcome verification needed)
        sample_output = os.path.join(output_dir, 'xauusd_rag_samples.json')
        sample_trades = converter.create_sample_trades(m5_file, sample_output, num_samples=100)

        print(f"\nSummary:")
        print(f"Verified successful trades: {len(verified_trades)}")
        print(f"Sample trades: {len(sample_trades)}")
        print(f"Output files:")
        print(f"  - Verified: {verified_output}")
        print(f"  - Samples: {sample_output}")

    else:
        print(f"Error: {m5_file} not found. Please run mt5_remote_client.py first.")

if __name__ == "__main__":
    main()