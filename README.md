# RAG-Based Trading System for XAUUSD

A sophisticated Retrieval-Augmented Generation (RAG) system that analyzes XAUUSD (Gold) trading data using historical patterns and AI-powered analysis.

## 🚀 Features

- **Remote MT5 Data Access**: Fetch real-time and historical data from MetaTrader 5 via WebSocket
- **RAG-Powered Analysis**: Use historical trade setups to inform current market analysis
- **Multi-Timeframe Support**: Analyze data across M5, M15, H1, H4, and Daily timeframes
- **Technical Indicators**: RSI, MACD, EMAs, ATR, Bollinger Bands, and more
- **AI Trade Recommendations**: Get entry/exit points, DCA levels, and risk management
- **Pattern Recognition**: Identify breakouts, reversals, trend continuations, and more

## 📋 System Requirements

### Linux (RAG System)
- Python 3.8+
- Network connectivity to Windows MT5 machine
- Ollama (for LLM inference)

### Windows (MT5 Data Server)
- Windows 10/11
- MetaTrader 5 terminal
- Python 3.8+
- Admin privileges

## 🛠️ Quick Start

### 1. Windows Setup - MT5 WebSocket Server

```cmd
# Clone the repository on your Windows machine
git clone <repository_url>
cd rag_finance

# Install Windows dependencies
pip install -r requirements_windows.txt

# Start the MT5 WebSocket server
cd windows_server
python mt5_websocket_server.py
```

### 2. Linux Setup - RAG System

```bash
# Clone the repository
git clone <repository_url>
cd rag_finance

# Install dependencies
pip install -r requirements.txt

# Install Ollama (if not installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Qwen3:8b model
ollama pull qwen3:8b

# Configure the MT5 client IP
edit scripts/mt5_remote_client.py  # Update SERVER_HOST
```

### 3. Run the Complete Pipeline

```bash
# Step 1: Fetch data from MT5
python scripts/mt5_remote_client.py

# Step 2: Convert to RAG format
python scripts/rag_converter.py

# Step 3: Import into RAG system
python scripts/batch_import.py

# Step 4: Run analysis
python trading_rag.py
```

## 📁 Project Structure

```
rag_finance/
├── windows_server/          # Windows MT5 WebSocket server
│   ├── mt5_websocket_server.py
│   └── README.md
├── scripts/                 # Data processing scripts
│   ├── mt5_remote_client.py
│   ├── rag_converter.py
│   └── batch_import.py
├── data/                    # Data storage
├── trading_knowledge/       # ChromaDB vector store
├── trading_rag.py          # Main RAG system
├── requirements.txt        # Linux dependencies
├── requirements_windows.txt # Windows dependencies
└── README.md
```

## 🔧 Configuration

### MT5 Server Configuration

Edit `windows_server/mt5_websocket_server.py`:
```python
# Change server port if needed
await server.start_server(host="0.0.0.0", port=8765)
```

### Client Configuration

Edit `scripts/mt5_remote_client.py`:
```python
# Update with your Windows machine IP
SERVER_HOST = "192.168.1.100"  # Your Windows IP
SERVER_PORT = 8765
SYMBOL = "XAUUSD"
```

## 📊 Data Flow

```
Windows MT5 → WebSocket Server → Linux Client → RAG Converter → Vector DB → LLM Analysis
```

1. **Data Collection**: MT5 exports historical and real-time data
2. **Format Conversion**: Convert CSV to RAG-compatible JSON
3. **Knowledge Base**: Store in ChromaDB vector database
4. **Analysis**: Retrieve similar patterns + LLM generation
5. **Recommendations**: Structured trade analysis with risk management

## 🎯 Output Format

The system generates comprehensive trade analysis including:

- **Trade Direction**: LONG/SHORT/NEUTRAL with confidence %
- **Entry Strategy**: Optimal entry points
- **DCA Levels**: 3-4 levels for dollar-cost averaging
- **Risk Management**: Stop loss and take profit targets
- **Risk-Reward Ratios**: For each profit target
- **Historical Context**: Similar past setups with outcomes

## 📈 Example Analysis

```
=== Trading Analysis for XAUUSD ===

## Trade Direction: LONG (Confidence: 78%)

## Reasoning:
- Strong bullish momentum with RSI at 68.5
- Price breaking above resistance at 2695.00
- Volume spike indicating institutional interest
- Historical similar setups: 75% success rate

## Entry Strategy:
- Primary entry: 2693.77
- Alternative: On pullback to 2690.00

## DCA Levels:
1. 2690.00
2. 2688.40
3. 2686.50

## Risk Management:
- Stop Loss: 2683.00
- Take Profit: [2697.50, 2702.00, 2708.00]
- Risk-Reward: 1:2.5, 1:4.0, 1:6.0

## Key Risk Factors:
- Break of support at 2688.40
- Negative news catalysts
- Rejection at 2700.00 psychological level
```

## 🐛 Troubleshooting

### Common Issues

**Connection Failed**:
```bash
# Test network connectivity
ping <windows_ip>

# Check WebSocket connection
telnet <windows_ip> 8765
```

**MT5 Initialization Failed**:
- Ensure MT5 terminal is running
- Run server as administrator
- Check Python MT5 integration

**No Data Received**:
- Verify symbol availability in MT5
- Check server logs for errors
- Ensure proper date ranges

**Ollama Not Working**:
```bash
# Check Ollama status
ollama list

# Restart Ollama service
sudo systemctl restart ollama
```

## 📚 API Reference

### TradingRAG Class

```python
from trading_rag import TradingRAG

# Initialize
rag = TradingRAG(model_name="qwen3:8b")

# Add historical knowledge
rag.add_trading_knowledge(trade_data)

# Analyze current market
result = rag.generate_trade_analysis(current_market, price_history)
```

### MT5RemoteClient Class

```python
from scripts.mt5_remote_client import MT5RemoteClient

# Initialize
client = MT5RemoteClient(host="192.168.1.100")

# Connect and fetch data
await client.connect()
data = await client.get_current_market_data("XAUUSD", "output.json")
```

## 🔒 Security Considerations

- **Network Security**: Use VPN or secure network for WebSocket communication
- **Access Control**: Consider implementing authentication for production use
- **Data Privacy**: All analysis is stored locally in ChromaDB

## 🚀 Performance Optimization

- **Vector Search**: ChromaDB uses HNSW for fast similarity search
- **Batch Processing**: Import data in batches to avoid memory issues
- **Caching**: Similar setups are cached for faster retrieval

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the logs for detailed error messages
- Open an issue on GitHub

---

**⚠️ Disclaimer**: This system is for educational and research purposes. Always do thorough backtesting before live trading. Trading involves substantial risk of loss.