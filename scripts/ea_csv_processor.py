import pandas as pd
import numpy as np
import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EA_CSV_Processor:
    def __init__(self, ea_export_folder="C:\\RAG_Data\\", local_data_folder="data"):
        """
        Initialize EA CSV Processor

        Args:
            ea_export_folder: Folder where EA exports CSV files (Windows path)
            local_data_folder: Local data folder for processed files
        """
        self.ea_export_folder = ea_export_folder
        self.local_data_folder = local_data_folder
        self.processed_files_log = os.path.join(local_data_folder, "processed_files.log")

        # Create local data folder
        os.makedirs(local_data_folder, exist_ok=True)
        os.makedirs(os.path.join(local_data_folder, "raw_ea"), exist_ok=True)
        os.makedirs(os.path.join(local_data_folder, "processed"), exist_ok=True)

        # Load processed files log
        self.processed_files = self._load_processed_files_log()

    def _load_processed_files_log(self):
        """Load log of processed files to avoid reprocessing"""
        if os.path.exists(self.processed_files_log):
            try:
                with open(self.processed_files_log, 'r') as f:
                    return set(line.strip() for line in f)
            except Exception as e:
                logger.warning(f"Could not load processed files log: {e}")
        return set()

    def _save_processed_files_log(self):
        """Save log of processed files"""
        try:
            with open(self.processed_files_log, 'w') as f:
                for filename in self.processed_files:
                    f.write(f"{filename}\n")
        except Exception as e:
            logger.error(f"Could not save processed files log: {e}")

    def _calculate_additional_indicators(self, df):
        """Calculate additional technical indicators not provided by EA"""
        try:
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

            # Volume average
            df['volume_avg'] = df['volume'].rolling(window=20).mean()

            # Support and Resistance (basic implementation)
            df['support_1'] = df['low'].rolling(window=20, center=True).min()
            df['resistance_1'] = df['high'].rolling(window=20, center=True).max()

            # Session information
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            df['session'] = df['hour'].apply(lambda x:
                'ASIAN_SESSION' if 0 <= x < 8 else
                'LONDON_SESSION' if 8 <= x < 13 else
                'US_SESSION' if 13 <= x < 20 else
                'AFTER_HOURS'
            )

            # Clean up temporary columns
            df = df.drop(['bb_std'], axis=1, errors='ignore')

            return df

        except Exception as e:
            logger.error(f"Error calculating additional indicators: {e}")
            return df

    def _find_support_resistance_advanced(self, df, window=20):
        """Advanced support/resistance identification"""
        supports = []
        resistances = []

        for i in range(window, len(df) - window):
            # Resistance (local high)
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
                resistances.append(df['high'].iloc[i])

            # Support (local low)
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
                supports.append(df['low'].iloc[i])

        return {
            'support_1': sorted(supports, reverse=True)[0] if supports else None,
            'support_2': sorted(supports, reverse=True)[1] if len(supports) > 1 else None,
            'support_3': sorted(supports, reverse=True)[2] if len(supports) > 2 else None,
            'resistance_1': sorted(resistances)[0] if resistances else None,
            'resistance_2': sorted(resistances)[1] if len(resistances) > 1 else None,
            'resistance_3': sorted(resistances)[2] if len(resistances) > 2 else None,
        }

    def process_enhanced_csv(self, file_path):
        """
        Process enhanced CSV file from EA with additional calculations

        Args:
            file_path: Path to enhanced CSV file
        """
        try:
            logger.info(f"Processing enhanced CSV: {file_path}")

            # Load CSV
            df = pd.read_csv(file_path)

            # Validate required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                              'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                              'ema_20', 'ema_50', 'ema_200', 'atr_14', 'trend']

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None

            # Calculate additional indicators
            df = self._calculate_additional_indicators(df)

            # Enhanced support/resistance calculation
            sr_levels = self._find_support_resistance_advanced(df)

            # Update support/resistance columns with better calculations
            for i in range(40, len(df)):
                df.loc[df.index[i], 'support_1'] = sr_levels['support_1']
                df.loc[df.index[i], 'resistance_1'] = sr_levels['resistance_1']

            # Remove rows with NaN values
            initial_count = len(df)
            df = df.dropna()
            final_count = len(df)

            logger.info(f"Removed {initial_count - final_count} rows with NaN values")

            # Save processed data
            filename = os.path.basename(file_path)
            processed_path = os.path.join(self.local_data_folder, f"processed_{filename}")
            df.to_csv(processed_path, index=False)

            logger.info(f"Processed {final_count} rows, saved to {processed_path}")
            return processed_path

        except Exception as e:
            logger.error(f"Error processing enhanced CSV {file_path}: {e}")
            return None

    def process_basic_csv(self, file_path):
        """
        Process basic CSV file from EA (OHLCV only)

        Args:
            file_path: Path to basic CSV file
        """
        try:
            logger.info(f"Processing basic CSV: {file_path}")

            # Load CSV
            df = pd.read_csv(file_path)

            # Basic validation
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Calculate all indicators (like our original MT5 export)
            df = self._calculate_all_indicators(df)

            # Save enhanced data
            filename = os.path.basename(file_path)
            enhanced_filename = filename.replace('.csv', '_enhanced.csv')
            enhanced_path = os.path.join(self.local_data_folder, enhanced_filename)
            df.to_csv(enhanced_path, index=False)

            logger.info(f"Enhanced basic CSV with indicators, saved to {enhanced_path}")
            return enhanced_path

        except Exception as e:
            logger.error(f"Error processing basic CSV {file_path}: {e}")
            return None

    def _calculate_all_indicators(self, df):
        """Calculate all technical indicators for basic CSV"""
        try:
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_200'] = df['close'].ewm(span=200).mean()

            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # ATR
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift())
            df['low_close'] = abs(df['low'] - df['close'].shift())
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['atr_14'] = df['tr'].rolling(window=14).mean()

            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

            # Volume average
            df['volume_avg'] = df['volume'].rolling(window=20).mean()

            # Trend determination
            df['trend'] = np.where(
                (df['close'] > df['ema_20']) &
                (df['ema_20'] > df['ema_50']) &
                (df['ema_50'] > df['ema_200']),
                'BULLISH',
                np.where(
                    (df['close'] < df['ema_20']) &
                    (df['ema_20'] < df['ema_50']) &
                    (df['ema_50'] < df['ema_200']),
                    'BEARISH',
                    'NEUTRAL'
                )
            )

            # Session information
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            df['session'] = df['hour'].apply(lambda x:
                'ASIAN_SESSION' if 0 <= x < 8 else
                'LONDON_SESSION' if 8 <= x < 13 else
                'US_SESSION' if 13 <= x < 20 else
                'AFTER_HOURS'
            )

            # Support/resistance levels
            df['support_1'] = np.nan
            df['resistance_1'] = np.nan

            for i in range(40, len(df)):
                sr = self._find_support_resistance_advanced(df.iloc[i-40:i])
                df.loc[df.index[i], 'support_1'] = sr['support_1']
                df.loc[df.index[i], 'resistance_1'] = sr['resistance_1']

            # Clean up temporary columns
            df = df.drop(['high_low', 'high_close', 'low_close', 'tr', 'bb_std'], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

    def process_current_market_json(self, file_path):
        """
        Process current market JSON file from EA

        Args:
            file_path: Path to current market JSON file
        """
        try:
            logger.info(f"Processing current market JSON: {file_path}")

            # Load JSON
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Enhance with additional data
            data['data_source'] = 'MT5_EA'
            data['processed_timestamp'] = datetime.now().isoformat()

            # Save enhanced JSON
            filename = os.path.basename(file_path)
            enhanced_path = os.path.join(self.local_data_folder, f"enhanced_{filename}")

            with open(enhanced_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Enhanced current market JSON, saved to {enhanced_path}")
            return enhanced_path

        except Exception as e:
            logger.error(f"Error processing current market JSON {file_path}: {e}")
            return None

    def scan_and_process_files(self):
        """Scan EA export folder and process new files"""
        if not os.path.exists(self.ea_export_folder):
            logger.warning(f"EA export folder does not exist: {self.ea_export_folder}")
            return False

        logger.info(f"Scanning EA export folder: {self.ea_export_folder}")

        processed_count = 0
        total_files = 0

        # Scan for files
        for filename in os.listdir(self.ea_export_folder):
            file_path = os.path.join(self.ea_export_folder, filename)

            if not os.path.isfile(file_path):
                continue

            total_files += 1

            # Skip if already processed
            if filename in self.processed_files:
                continue

            try:
                # Copy file to local raw folder
                raw_local_path = os.path.join(self.local_data_folder, "raw_ea", filename)
                shutil.copy2(file_path, raw_local_path)

                # Process based on file type
                if filename.endswith('.csv'):
                    if 'enhanced' in filename:
                        processed_path = self.process_enhanced_csv(raw_local_path)
                    else:
                        processed_path = self.process_basic_csv(raw_local_path)

                elif filename.endswith('.json') and 'current_market' in filename:
                    processed_path = self.process_current_market_json(raw_local_path)
                else:
                    logger.info(f"Skipping unsupported file: {filename}")
                    continue

                if processed_path:
                    processed_count += 1
                    logger.info(f"Successfully processed: {filename}")

                # Mark as processed
                self.processed_files.add(filename)

            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")

        # Save processed files log
        self._save_processed_files_log()

        logger.info(f"Processing complete: {processed_count}/{total_files} files processed")
        return processed_count > 0

    def get_latest_processed_data(self):
        """Get information about latest processed data"""
        processed_files = []

        for filename in os.listdir(self.local_data_folder):
            if filename.startswith('processed_') and filename.endswith('.csv'):
                file_path = os.path.join(self.local_data_folder, filename)
                file_time = os.path.getmtime(file_path)
                processed_files.append((filename, file_time, file_path))

        if not processed_files:
            return None

        # Sort by modification time
        processed_files.sort(key=lambda x: x[1], reverse=True)
        latest_file = processed_files[0]

        return {
            'filename': latest_file[0],
            'path': latest_file[2],
            'modified_time': datetime.fromtimestamp(latest_file[1])
        }

def main():
    """Main function for manual processing"""
    print("=== EA CSV Processor ===")

    # Initialize processor (adjust paths as needed)
    processor = EA_CSV_Processor(
        ea_export_folder="C:\\RAG_Data\\",  # Update with your EA export path
        local_data_folder="data"
    )

    # Process files
    print("Scanning for new files...")
    success = processor.scan_and_process_files()

    if success:
        print("Files processed successfully!")

        # Show latest data
        latest = processor.get_latest_processed_data()
        if latest:
            print(f"Latest processed file: {latest['filename']}")
            print(f"Modified: {latest['modified_time']}")
    else:
        print("No new files found or processing failed.")

if __name__ == "__main__":
    main()