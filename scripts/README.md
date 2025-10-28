# MT5 Remote Client (Linux)

This client connects to the Windows MT5 WebSocket server to fetch trading data for the RAG system.

## Setup Instructions

### 1. Prerequisites (Linux)

- Linux (Ubuntu/Debian/CentOS)
- Python 3.8+
- Network connectivity to Windows machine

### 2. Install Dependencies

```bash
pip install websockets pandas numpy
```

### 3. Configure Connection

Edit `mt5_remote_client.py` and update the server configuration:

```python
# Update these values
SERVER_HOST = "192.168.1.100"  # Your Windows machine IP
SERVER_PORT = 8765
SYMBOL = "XAUUSD"
```

### 4. Run the Client

```bash
python mt5_remote_client.py
```

The client will:
1. Connect to the Windows WebSocket server
2. Test the connection
3. Export historical data for multiple timeframes
4. Export current market state
5. Save all data to the `data/` directory

## Generated Files

The client will create the following files in the `data/` directory:

### Historical Training Data (CSV)
- `xauusd_training_data_m5.csv` - 5-minute data
- `xauusd_training_data_m15.csv` - 15-minute data
- `xauusd_training_data_h1.csv` - 1-hour data
- `xauusd_training_data_h4.csv` - 4-hour data

Each CSV contains:
- OHLCV data
- Technical indicators (RSI, MACD, EMAs, ATR, Bollinger Bands)
- Support/resistance levels
- Trend information
- Session data

### Current Market State (JSON)
- `xauusd_current_market.json` - Multi-timeframe snapshot

## Troubleshooting

### Connection Failed
1. Verify the Windows server IP address
2. Check that the WebSocket server is running on Windows
3. Ensure port 8765 is not blocked by firewall
4. Test network connectivity: `ping <windows_ip>`

### No Data Received
1. Make sure MT5 is running on Windows
2. Verify XAUUSD symbol is available in MT5
3. Check server logs for errors
4. Try a smaller date range first

### Network Issues
```bash
# Test WebSocket connection
python -c "import asyncio; import websockets; asyncio.run(websockets.connect('ws://<windows_ip>:8765'))"
```

### Firewall Issues
```bash
# Check if port is accessible
telnet <windows_ip> 8765
```

## Advanced Usage

You can also use the client programmatically:

```python
import asyncio
from mt5_remote_client import MT5RemoteClient

async def fetch_data():
    client = MT5RemoteClient(host="192.168.1.100")

    if await client.connect():
        data = await client.get_current_market_data("XAUUSD", "market.json")
        await client.disconnect()

    return data

# Run it
data = asyncio.run(fetch_data())
```

## Data Format

### CSV Export Format
```csv
timestamp,open,high,low,close,volume,rsi_14,macd,macd_signal,macd_hist,ema_20,ema_50,ema_200,atr_14,bb_upper,bb_middle,bb_lower,volume_avg,support_1,resistance_1,trend,session,day_of_week,hour
2025-01-15 15:30:00,2693.50,2694.21,2690.15,2693.77,45083,68.5,2.34,1.89,0.45,2686.50,2680.30,2665.80,4.23,2695.80,2688.50,2681.20,4500,2688.40,2695.00,BULLISH,US_SESSION,WEDNESDAY,15
```

### JSON Export Format
```json
{
  "query_timestamp": "2025-01-15 15:30:00",
  "symbol": "XAUUSD",
  "current_price": 2693.77,
  "multi_timeframe": {
    "M5": {
      "open": 2693.50,
      "high": 2694.21,
      "low": 2690.15,
      "close": 2693.77,
      "volume": 45083,
      "rsi_14": 68.5,
      "macd": 2.34,
      "macd_signal": 1.89,
      "ema_20": 2686.50,
      "ema_50": 2680.30,
      "ema_200": 2665.80,
      "atr_14": 4.23,
      "trend": "BULLISH"
    }
  }
}
```