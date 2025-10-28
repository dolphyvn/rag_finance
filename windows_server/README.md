# MT5 WebSocket Server (Windows)

This WebSocket server runs on your Windows machine with MT5 installed and provides remote access to MT5 data for your Linux RAG system.

## Setup Instructions

### 1. Prerequisites (Windows)

- Windows 10/11
- MetaTrader 5 installed and running
- Python 3.8+ installed
- Admin privileges (for MT5 integration)

### 2. Install Dependencies

```cmd
pip install MetaTrader5 pandas numpy websockets
```

### 3. Configure MT5

1. Open MetaTrader 5 terminal
2. Enable **Tools → Options → Expert Advisors → Allow algorithmic trading**
3. Enable **Tools → Options → Expert Advisors → Allow DLL imports**
4. Make sure XAUUSD (Gold) is available in your market watch

### 4. Start the Server

```cmd
python mt5_websocket_server.py
```

The server will start on `ws://0.0.0.0:8765` and accept connections from any IP address.

### 5. Firewall Configuration

You may need to allow Python through Windows Firewall for port 8765:

1. Open Windows Defender Firewall
2. Go to "Allow an app or feature through Windows Defender Firewall"
3. Add Python and allow port 8765 for "Private" networks

### 6. Test the Server

The server will display:
- Connection status
- Client connections
- Data export requests
- Any errors

## Server API

The WebSocket server accepts JSON requests with the following format:

### Export Historical Data
```json
{
    "type": "export_historical",
    "symbol": "XAUUSD",
    "timeframe": "M5",
    "start_date": "2024-01-01T00:00:00",
    "end_date": "2025-01-15T00:00:00"
}
```

### Get Current Market Data
```json
{
    "type": "get_current_market",
    "symbol": "XAUUSD"
}
```

### Ping (Connection Test)
```json
{
    "type": "ping"
}
```

## Response Format

All responses include a "status" field:
```json
{
    "status": "success",
    "data": [...],
    "count": 12345
}
```

Or for errors:
```json
{
    "status": "error",
    "message": "Error description"
}
```

## Troubleshooting

### MT5 Initialization Failed
- Make sure MT5 terminal is running
- Run the server as administrator
- Check if MT5 allows Python integration

### Connection Issues
- Check if port 8765 is blocked by firewall
- Verify network connectivity between machines
- Try using "localhost" instead of IP for local testing

### No Data Available
- Make sure the symbol (XAUUSD) is available in MT5
- Check if you have historical data for the requested timeframe
- Verify the date range is reasonable

## Security Notes

- The server accepts connections from any IP address
- Consider restricting access in production
- No authentication is implemented (add if needed)
- All data is transmitted unencrypted (consider WSS for production)

## Log Files

The server logs all activities to console. Consider redirecting to a file:

```cmd
python mt5_websocket_server.py > server.log 2>&1
```