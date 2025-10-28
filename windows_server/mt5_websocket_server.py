import asyncio
import websockets
import json
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MT5WebSocketServer:
    def __init__(self):
        """Initialize MT5 connection"""
        logger.info("Initializing MT5...")
        if not mt5.initialize():
            raise Exception("MT5 initialization failed. Make sure MT5 terminal is running.")

        self.connected_clients = set()
        logger.info("MT5 WebSocket Server initialized successfully")

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
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
        df['volume_avg'] = df['tick_volume'].rolling(window=20).mean()

        return df

    def find_support_resistance(self, df, window=20):
        """Find support and resistance levels"""
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

    async def export_historical_data(self, symbol, timeframe_str, start_date_str, end_date_str):
        """Export historical data with indicators"""
        try:
            # Convert string parameters
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }

            timeframe = timeframe_map.get(timeframe_str)
            if not timeframe:
                raise ValueError(f"Invalid timeframe: {timeframe_str}")

            start_date = datetime.fromisoformat(start_date_str)
            end_date = datetime.fromisoformat(end_date_str)

            logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")

            # Get historical data
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

            if rates is None or len(rates) == 0:
                raise ValueError(f"No data retrieved for {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={'time': 'timestamp', 'tick_volume': 'volume'})

            # Calculate indicators
            df = self.calculate_indicators(df)

            # Determine trend direction
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

            # Add session information
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            df['session'] = df['hour'].apply(lambda x:
                'ASIAN_SESSION' if 0 <= x < 8 else
                'LONDON_SESSION' if 8 <= x < 13 else
                'US_SESSION' if 13 <= x < 20 else
                'AFTER_HOURS'
            )

            # Add support/resistance levels
            df['support_1'] = np.nan
            df['resistance_1'] = np.nan

            for i in range(40, len(df)):
                sr = self.find_support_resistance(df.iloc[i-40:i])
                df.loc[df.index[i], 'support_1'] = sr['support_1']
                df.loc[df.index[i], 'resistance_1'] = sr['resistance_1']

            # Select relevant columns and convert to dict
            columns_to_export = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'ema_20', 'ema_50', 'ema_200',
                'atr_14', 'bb_upper', 'bb_middle', 'bb_lower',
                'volume_avg', 'support_1', 'resistance_1',
                'trend', 'session', 'day_of_week', 'hour'
            ]

            df_export = df[columns_to_export].copy()
            df_export = df_export.dropna()

            # Convert to list of dicts for JSON serialization
            result = df_export.to_dict('records')

            logger.info(f"Successfully exported {len(result)} records")
            return {"status": "success", "data": result, "count": len(result)}

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return {"status": "error", "message": str(e)}

    async def get_current_market_data(self, symbol):
        """Get current multi-timeframe market data"""
        try:
            timeframes = {
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }

            data = {
                'query_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'multi_timeframe': {}
            }

            for tf_name, tf_value in timeframes.items():
                # Get last 200 candles
                rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 200)

                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df = df.rename(columns={'tick_volume': 'volume'})

                    # Calculate indicators
                    df = self.calculate_indicators(df)
                    latest = df.iloc[-1]

                    # Determine trend
                    trend = 'BULLISH' if (latest['close'] > latest['ema_20'] and
                                        latest['ema_20'] > latest['ema_50'] and
                                        latest['ema_50'] > latest['ema_200']) else \
                           'BEARISH' if (latest['close'] < latest['ema_20'] and
                                        latest['ema_20'] < latest['ema_50'] and
                                        latest['ema_50'] < latest['ema_200']) else 'NEUTRAL'

                    data['multi_timeframe'][tf_name] = {
                        'open': float(latest['open']),
                        'high': float(latest['high']),
                        'low': float(latest['low']),
                        'close': float(latest['close']),
                        'volume': int(latest['volume']),
                        'rsi_14': float(latest['rsi_14']),
                        'macd': float(latest['macd']),
                        'macd_signal': float(latest['macd_signal']),
                        'ema_20': float(latest['ema_20']),
                        'ema_50': float(latest['ema_50']),
                        'ema_200': float(latest['ema_200']),
                        'atr_14': float(latest['atr_14']),
                        'trend': trend
                    }

            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                data['current_price'] = float(tick.bid)

            logger.info(f"Successfully retrieved current market data for {symbol}")
            return {"status": "success", "data": data}

        except Exception as e:
            logger.error(f"Error getting current market data: {e}")
            return {"status": "error", "message": str(e)}

    async def handle_client_message(self, message):
        """Handle incoming client messages"""
        try:
            request = json.loads(message)
            request_type = request.get('type')

            if request_type == 'export_historical':
                return await self.export_historical_data(
                    symbol=request['symbol'],
                    timeframe_str=request['timeframe'],
                    start_date_str=request['start_date'],
                    end_date_str=request['end_date']
                )

            elif request_type == 'get_current_market':
                return await self.get_current_market_data(
                    symbol=request['symbol']
                )

            elif request_type == 'ping':
                return {"status": "success", "message": "pong"}

            else:
                return {"status": "error", "message": f"Unknown request type: {request_type}"}

        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            return {"status": "error", "message": str(e)}

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        client_address = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_address}")

        self.connected_clients.add(websocket)

        try:
            async for message in websocket:
                logger.info(f"Received message from {client_address}: {message[:100]}...")

                # Process the request
                response = await self.handle_client_message(message)

                # Send response
                await websocket.send(json.dumps(response))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_address}")
        except Exception as e:
            logger.error(f"Error with client {client_address}: {e}")
        finally:
            self.connected_clients.discard(websocket)

    async def broadcast_message(self, message):
        """Broadcast message to all connected clients"""
        if self.connected_clients:
            await asyncio.gather(
                *[client.send(message) for client in self.connected_clients],
                return_exceptions=True
            )

    async def start_server(self, host="0.0.0.0", port=8765):
        """Start the WebSocket server"""
        logger.info(f"Starting MT5 WebSocket server on {host}:{port}")

        async with websockets.serve(self.handle_client, host, port):
            logger.info("Server started successfully. Waiting for connections...")
            await asyncio.Future()  # Run forever

    def cleanup(self):
        """Cleanup MT5 connection"""
        logger.info("Shutting down MT5...")
        mt5.shutdown()

async def main():
    """Main function"""
    server = None
    try:
        server = MT5WebSocketServer()
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested...")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        if server:
            server.cleanup()

if __name__ == "__main__":
    print("=== MT5 WebSocket Server ===")
    print("Make sure MetaTrader 5 terminal is running before starting this server.")
    print("Server will start on ws://0.0.0.0:8765")
    print("Press Ctrl+C to stop the server.")
    print()

    asyncio.run(main())