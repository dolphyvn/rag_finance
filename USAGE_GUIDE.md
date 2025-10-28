# Complete Usage Guide

This guide walks you through setting up and using the RAG Trading System step by step.

## 🎯 Overview

The system consists of two main components:
1. **Windows MT5 WebSocket Server** - Fetches data from MetaTrader 5
2. **Linux RAG System** - Analyzes data using AI and historical patterns

## 🔧 Prerequisites Checklist

### Windows Machine (MT5 Server)
- [ ] Windows 10/11 installed
- [ ] MetaTrader 5 terminal installed and working
- [ ] Python 3.8+ installed
- [ ] Admin privileges for MT5 integration
- [ ] XAUUSD (Gold) symbol available in MT5

### Linux Machine (RAG System)
- [ ] Linux (Ubuntu/Debian/CentOS)
- [ ] Python 3.8+ installed
- [ ] Network connectivity to Windows machine
- [ ] At least 4GB RAM available
- [ ] 10GB+ disk space for data

## 📋 Step-by-Step Setup

### Part 1: Windows MT5 Server Setup

#### 1.1 Install Dependencies
```cmd
# Open Command Prompt as Administrator
pip install MetaTrader5 pandas numpy websockets
```

#### 1.2 Configure MetaTrader 5
1. Open MT5 terminal
2. Go to **Tools → Options**
3. **Expert Advisors tab**:
   - ✅ Allow algorithmic trading
   - ✅ Allow DLL imports
4. Go to **Market Watch** and ensure XAUUSD is added
5. Keep MT5 terminal running

#### 1.3 Start WebSocket Server
```cmd
cd path\to\rag_finance\windows_server
python mt5_websocket_server.py
```

Expected output:
```
=== MT5 WebSocket Server ===
Make sure MetaTrader 5 terminal is running before starting this server.
Server will start on ws://0.0.0.0:8765
Press Ctrl+C to stop the server.

2025-01-28 10:00:00 - INFO - Initializing MT5...
2025-01-28 10:00:01 - INFO - MT5 WebSocket Server initialized successfully
2025-01-28 10:00:01 - INFO - Starting MT5 WebSocket server on 0.0.0.0:8765
2025-01-28 10:00:01 - INFO - Server started successfully. Waiting for connections...
```

#### 1.4 Configure Firewall (if needed)
If you see connection errors, you may need to:
1. Open Windows Defender Firewall
2. Click "Allow an app or feature through Windows Defender Firewall"
3. Find Python in the list and allow it for "Private" networks
4. Or manually create a rule for port 8765

### Part 2: Linux RAG System Setup

#### 2.1 Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

#### 2.2 Install Python Dependencies
```bash
cd /opt/works/personal/rag_finance
pip install -r requirements.txt
```

#### 2.3 Install Ollama (LLM Engine)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl start ollama
sudo systemctl enable ollama

# Pull the model (this may take a few minutes)
ollama pull qwen3:8b

# Verify installation
ollama list
```

#### 2.4 Configure MT5 Client Connection
Edit the connection settings:
```bash
nano scripts/mt5_remote_client.py
```

Update this line with your Windows machine's IP:
```python
SERVER_HOST = "192.168.1.100"  # Replace with your Windows IP
```

To find your Windows IP:
- Open Command Prompt on Windows
- Run: `ipconfig`
- Look for IPv4 Address (usually 192.168.x.x)

### Part 3: Data Collection and Processing

#### 3.1 Test Connection
```bash
# Test WebSocket connection
python scripts/mt5_remote_client.py
```

Expected output:
```
=== MT5 Remote Client ===
Make sure the MT5 WebSocket server is running on the Windows machine.
Update SERVER_HOST variable with your Windows machine IP address.

2025-01-28 10:05:00 - INFO - Connecting to MT5 WebSocket server at ws://192.168.1.100:8765
2025-01-28 10:05:01 - INFO - Successfully connected to MT5 WebSocket server
2025-01-28 10:05:01 - INFO - Connection test successful!

=== Exporting Historical Training Data ===
...
```

#### 3.2 Export Historical Data
The script will automatically:
- Fetch 6 months of historical data
- Export multiple timeframes (M5, M15, H1, H4)
- Save to `data/` directory

Expected files created:
```
data/
├── xauusd_training_data_m5.csv
├── xauusd_training_data_m15.csv
├── xauusd_training_data_h1.csv
├── xauusd_training_data_h4.csv
└── xauusd_current_market.json
```

#### 3.3 Convert to RAG Format
```bash
python scripts/rag_converter.py
```

This will:
- Analyze the CSV data for trade setups
- Identify successful trades (minimum 5 points profit)
- Create rich market context for each setup
- Generate two output files:
  - `data/xauusd_rag_verified.json` - Verified successful trades
  - `data/xauusd_rag_samples.json` - Sample trades for testing

Expected output:
```
Processing data/xauusd_training_data_m5.csv...
2025-01-28 10:10:00 - INFO - Loaded 43200 rows from data/xauusd_training_data_m5.csv
2025-01-28 10:10:30 - INFO - Successfully created 1247 RAG examples from data/xauusd_training_data_m5.csv
2025-01-28 10:10:30 - INFO - Output saved to data/xauusd_rag_verified.json
2025-01-28 10:10:45 - INFO - Successfully created 100 sample RAG examples
2025-01-28 10:10:45 - INFO - Output saved to data/xauusd_rag_samples.json

Summary:
Verified successful trades: 1247
Sample trades: 100
```

#### 3.4 Import to RAG Knowledge Base
```bash
python scripts/batch_import.py
```

This will:
- Load the JSON files created in previous step
- Validate each trade setup
- Import into ChromaDB vector database
- Show import statistics

Expected output:
```
=== Batch Import Tool for Trading RAG System ===
Initializing RAG system...
Found file: data/xauusd_rag_verified.json
Found file: data/xauusd_rag_samples.json

--- Importing data/xauusd_rag_verified.json ---
2025-01-28 10:15:00 - INFO - Starting batch import from data/xauusd_rag_verified.json
2025-01-28 10:15:00 - INFO - Loaded 1247 trades from data/xauusd_rag_verified.json
2025-01-28 10:15:00 - INFO - Found 0 existing trades in knowledge base
Processing batch 1/25 (50 trades)...
...
Import Results for xauusd_rag_verified.json:
  Total processed: 1247
  Successful: 1247
  Failed: 0
  Skipped duplicates: 0
  Success rate: 100.0%
```

### Part 4: Running the Analysis

#### 4.1 Test RAG System
```bash
python trading_rag.py
```

This will:
- Initialize the RAG system
- Show knowledge base statistics
- Generate a sample analysis

Expected output:
```
=== Trading RAG System ===
Initializing RAG system...
2025-01-28 10:20:00 - INFO - Loading embedding model...
2025-01-28 10:20:05 - INFO - Embedding model loaded successfully
2025-01-28 10:20:05 - INFO - Ollama initialized with model: qwen3:8b

Knowledge Base Stats:
  total_trades: 1347
  vector_db_path: ./trading_knowledge
  model: qwen3:8b
  embedding_model: sentence-transformers/all-MiniLM-L6-v2

Generating sample analysis...

=== Trading Analysis ===
[Detailed AI analysis will appear here...]
```

#### 4.2 Real-time Analysis (Optional)
For real-time analysis with current market data, you can modify the `trading_rag.py` to use the MT5 client:

```python
# In trading_rag.py main function
from scripts.mt5_remote_client import MT5RemoteClient

async def real_time_analysis():
    client = MT5RemoteClient()
    if await client.connect():
        result = await trading_rag.analyze_current_market(mt5_client=client)
        print(result['analysis'])
```

## 🎯 Understanding the Output

### Sample Analysis Output
```
# Trading Analysis for XAUUSD

## Trade Direction: LONG (Confidence: 78%)

## Reasoning:
- Strong bullish momentum with RSI at 68.5 indicating healthy strength
- Price breaking above resistance at 2695.00 with volume confirmation
- Historical similar setups show 75% success rate
- EMAs aligned bullish (20 > 50 > 200)

## Entry Strategy:
- Primary entry: 2693.77 (current market price)
- Alternative: Wait for pullback to 2690.00 for better risk/reward

## DCA Levels (if trade goes against you):
1. 2690.00 (-3.8 points)
2. 2688.40 (-5.4 points)
3. 2686.50 (-7.3 points)

## Risk Management:
- Stop Loss: 2683.00 (conservative)
- Take Profit Targets:
  - TP1: 2697.50 (Risk:Reward 1:1.5)
  - TP2: 2702.00 (Risk:Reward 1:3.0)
  - TP3: 2708.00 (Risk:Reward 1:5.0)

## Key Risk Factors:
- Break below support at 2688.40 could invalidate bullish setup
- Negative news catalysts could cause sharp reversals
- Rejection at 2700.00 psychological level

## Similar Historical Setups Found: 5
Setup 1 (WIN, BREAKOUT LONG, Profit: +8.2 points): [context summary]
Setup 2 (WIN, TREND_CONTINUATION LONG, Profit: +6.5 points): [context summary]
...
```

## 📊 Monitoring and Maintenance

### Check Knowledge Base Size
```bash
python -c "
from trading_rag import TradingRAG
rag = TradingRAG()
stats = rag.get_knowledge_base_stats()
print(f'Total trades in knowledge base: {stats[\"total_trades\"]}')
"
```

### Update Data Regularly
```bash
# Run weekly to keep data fresh
python scripts/mt5_remote_client.py
python scripts/rag_converter.py
python scripts/batch_import.py
```

### Log Files
Check system logs:
```bash
tail -f logs/rag_trading.log
```

## 🔧 Customization

### Change Timeframes
Edit `scripts/mt5_remote_client.py`:
```python
timeframes = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1']  # Add or remove timeframes
```

### Adjust Profit Thresholds
Edit `scripts/rag_converter.py`:
```python
converter.convert_csv_to_rag(csv_file, output_file, min_profit_points=8.0)  # Increase to 8 points
```

### Modify LLM Parameters
Edit `trading_rag.py`:
```python
response = self.ollama.generate(
    model=self.llm_model,
    prompt=prompt,
    options={
        'temperature': 0.2,  # Lower for more consistent responses
        'top_p': 0.95,
        'top_k': 40
    }
)
```

## 🚨 Troubleshooting

### Common Issues and Solutions

**Connection Refused Error**
```bash
# Check if Windows server is running
telnet <windows_ip> 8765

# If connection fails, check:
# 1. Windows firewall settings
# 2. MT5 server is running
# 3. IP address is correct
```

**Ollama Model Not Found**
```bash
# List available models
ollama list

# Pull the model if missing
ollama pull qwen3:8b

# Restart Ollama service
sudo systemctl restart ollama
```

**Memory Issues**
```bash
# Reduce batch size in batch_import.py
python scripts/batch_import.py  # Uses smaller batch size by default
```

**Slow Performance**
- Ensure sufficient RAM (4GB+ recommended)
- Use SSD for better I/O performance
- Consider reducing knowledge base size if too large

## 📈 Advanced Usage

### Custom Trade Setups
Add your own setup patterns in `scripts/rag_converter.py`:
```python
def _identify_custom_setup(self, df, index):
    # Your custom logic here
    if your_condition:
        return {
            'direction': 'LONG',
            'confidence': 0.8,
            'entry': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reasoning': 'Your custom reasoning'
        }
    return None

# Add to setup_patterns
self.setup_patterns['CUSTOM_SETUP'] = self._identify_custom_setup
```

### Multiple Symbol Support
Modify scripts to handle multiple symbols:
```python
symbols = ['XAUUSD', 'EURUSD', 'GBPUSD']
for symbol in symbols:
    # Process each symbol
    await client.export_historical_data(symbol, ...)
```

### Export Results to File
```python
# Save analysis results
result = trading_rag.generate_trade_analysis(current_market, price_history)
with open('analysis_results.json', 'w') as f:
    json.dump(result, f, indent=2)
```

## 📚 Next Steps

1. **Backtesting**: Implement a backtesting framework to validate the system
2. **Paper Trading**: Test with real-time data without risking capital
3. **Performance Monitoring**: Track accuracy and profitability over time
4. **Model Fine-tuning**: Experiment with different LLM models and parameters
5. **Additional Features**: Add sentiment analysis, economic calendar integration

---

**🎉 Congratulations!** You now have a fully functional RAG-based trading system. Remember to start with paper trading before risking real capital.