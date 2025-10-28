import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MT5RemoteClient:
    def __init__(self, host="localhost", port=8765):
        """
        Initialize MT5 Remote Client

        Args:
            host: WebSocket server host (default: localhost)
            port: WebSocket server port (default: 8765)
        """
        self.host = host
        self.port = port
        self.websocket_url = f"ws://{host}:{port}"
        self.websocket = None

    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            logger.info(f"Connecting to MT5 WebSocket server at {self.websocket_url}")
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info("Successfully connected to MT5 WebSocket server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MT5 WebSocket server: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from MT5 WebSocket server")

    async def send_request(self, request):
        """Send a request to the WebSocket server"""
        if not self.websocket:
            raise ConnectionError("Not connected to WebSocket server")

        try:
            message = json.dumps(request)
            await self.websocket.send(message)
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            raise

    async def test_connection(self):
        """Test connection to the WebSocket server"""
        try:
            response = await self.send_request({"type": "ping"})
            return response.get("status") == "success"
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def export_historical_data(self, symbol, timeframe, start_date, end_date, output_file):
        """
        Export historical data from remote MT5 server

        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            timeframe: Timeframe string (e.g., 'M5', 'H1', 'D1')
            start_date: Start date (datetime object)
            end_date: End date (datetime object)
            output_file: Output CSV filename
        """
        try:
            logger.info(f"Requesting {symbol} {timeframe} data from {start_date} to {end_date}")

            request = {
                "type": "export_historical",
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }

            response = await self.send_request(request)

            if response.get("status") == "success":
                data = response.get("data", [])
                count = response.get("count", 0)

                logger.info(f"Received {count} records from server")

                if count > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(data)

                    # Convert timestamp back to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                    # Create output directory if it doesn't exist
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)

                    # Export to CSV
                    df.to_csv(output_file, index=False)
                    logger.info(f"Successfully exported {count} rows to {output_file}")
                    return df
                else:
                    logger.warning("No data received from server")
                    return None
            else:
                error_msg = response.get("message", "Unknown error")
                logger.error(f"Server returned error: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"Error exporting historical data: {e}")
            return None

    async def get_current_market_data(self, symbol, output_file):
        """
        Get current multi-timeframe market data from remote MT5 server

        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            output_file: Output JSON filename
        """
        try:
            logger.info(f"Requesting current market data for {symbol}")

            request = {
                "type": "get_current_market",
                "symbol": symbol
            }

            response = await self.send_request(request)

            if response.get("status") == "success":
                data = response.get("data")

                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                # Export to JSON
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)

                logger.info(f"Successfully exported current market data to {output_file}")
                return data
            else:
                error_msg = response.get("message", "Unknown error")
                logger.error(f"Server returned error: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"Error getting current market data: {e}")
            return None

async def main():
    """Main function to export data"""
    # Configuration - adjust these values
    SERVER_HOST = "192.168.1.100"  # Replace with your Windows machine IP
    SERVER_PORT = 8765
    SYMBOL = "XAUUSD"

    # Initialize client
    client = MT5RemoteClient(host=SERVER_HOST, port=SERVER_PORT)

    try:
        # Connect to server
        if not await client.connect():
            logger.error("Failed to connect to server. Exiting.")
            return

        # Test connection
        if not await client.test_connection():
            logger.error("Connection test failed. Exiting.")
            return

        logger.info("Connection test successful!")

        # Create data directory
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)

        # 1. Export Historical Training Data (different timeframes)
        print("\n=== Exporting Historical Training Data ===")

        timeframes = ['M5', 'M15', 'H1', 'H4']

        # Export last 6 months of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        for timeframe in timeframes:
            print(f"\nExporting {timeframe} timeframe...")
            output_file = os.path.join(data_dir, f'xauusd_training_data_{timeframe.lower()}.csv')

            df = await client.export_historical_data(
                symbol=SYMBOL,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                output_file=output_file
            )

            if df is not None:
                print(f"✓ Successfully exported {len(df)} rows for {timeframe}")
            else:
                print(f"✗ Failed to export {timeframe} data")

        # 2. Export current market state for prediction
        print("\n=== Exporting Current Market State ===")
        current_data = await client.get_current_market_data(
            symbol=SYMBOL,
            output_file=os.path.join(data_dir, 'xauusd_current_market.json')
        )

        if current_data:
            print("✓ Successfully exported current market state")
        else:
            print("✗ Failed to export current market state")

        print("\n=== Export Complete ===")

    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    print("=== MT5 Remote Client ===")
    print("Make sure the MT5 WebSocket server is running on the Windows machine.")
    print("Update SERVER_HOST variable with your Windows machine IP address.")
    print()

    # Run the client
    asyncio.run(main())