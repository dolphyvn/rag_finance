# EA-Based Data Export Setup Guide

This guide shows you how to use the MT5 Expert Advisor (EA) approach for exporting trading data to your RAG system.

## 🎯 Overview

The EA-based approach consists of:
1. **MT5 Expert Advisor** - Runs inside MT5 and exports data to CSV files
2. **Python File Processor** - Reads CSV files and converts them to RAG format
3. **Automated File Watcher** - Monitors folder for new files and processes them automatically

## 🏗️ Architecture

```
MT5 Terminal → EA (Expert Advisor) → CSV Files → Python Processor → RAG System → AI Analysis
```

## 📋 Prerequisites

### Windows Machine (MT5 + EA)
- [ ] Windows 10/11
- [ ] MetaTrader 5 terminal
- [ ] MT5 trading account (demo or live)
- [ ] XAUUSD symbol available
- [ ] Write permissions for export folder

### Linux Machine (RAG System)
- [ ] Linux (Ubuntu/Debian/CentOS)
- [ ] Python 3.8+
- [ ] Access to Windows shared folder or file transfer method
- [ ] Ollama installed (optional, for LLM)

## 🔧 Step-by-Step Setup

### Part 1: MT5 Expert Advisor Setup

#### 1.1 Create Export Folder
Create a folder on your Windows machine for EA exports:
```
C:\RAG_Data\
```

Ensure MT5 has write permissions to this folder.

#### 1.2 Compile and Install the EA

1. **Open MetaEditor** in MT5 (F4 key or Tools → MetaQuotes Language Editor)

2. **Create New EA**:
   - File → New → Expert Advisor (template)
   - Name: `RAG_DataExporter`
   - Delete template content and paste the EA code from `mt5_ea/RAG_DataExporter.mq5`

3. **Compile the EA**:
   - Press F7 or click Compile button
   - Fix any compilation errors if they occur

4. **Install EA on Chart**:
   - Open XAUUSD chart in MT5
   - Drag EA from Navigator to chart
   - Configure settings (see below)
   - Click OK

#### 1.3 Configure EA Settings

**Export Settings:**
- `ExportFolder`: `C:\RAG_Data\` (or your preferred path)
- `ExportInterval`: 60 seconds (how often to export)
- `ExportHistorical`: true (export historical data on start)
- `HistoricalDays`: 180 (days of historical data)

**Symbols and Timeframes:**
- `SymbolsToExport`: `XAUUSD` (comma-separated for multiple symbols)
- `TimeframesToExport`: `M5,M15,H1,H4,D1`

**File Settings:**
- `FilePrefix`: `XAUUSD_` (prefix for exported files)
- `CompressFiles`: false (whether to compress CSV files)

#### 1.4 Verify EA is Running

Check the Experts tab in MT5 Toolbox:
- Green smiley face = EA is running
- Red sad face = EA has errors
- Check journal logs for detailed status

Expected output in MT5 Journal:
```
=== RAG Data Exporter EA Initialized ===
Export folder: C:\RAG_Data\
Symbols to export: XAUUSD
Timeframes to export: M5,M15,H1,H4,D1
Starting historical data export...
Exporting historical data for XAUUSD M5...
Exported 43200 records to C:\RAG_Data\XAUUSD_training_data_m5.csv
Historical data export completed!
```

### Part 2: Linux System Setup

#### 2.1 Install Dependencies
```bash
cd /opt/works/personal/rag_finance
pip install -r requirements.txt
```

#### 2.2 Configure File Access

**Option A: Network Shared Folder**
```bash
# Install cifs-utils for Windows share access
sudo apt install cifs-utils

# Mount Windows share (adjust path and credentials)
sudo mkdir -p /mnt/rag_data
sudo mount -t cifs //192.168.1.100/RAG_Data /mnt/rag_data \
  -o username=your_windows_username,password=your_password,uid=$(id -u),gid=$(id -g)
```

**Option B: File Transfer (SCP/FTP)**
Set up automated file transfer from Windows to Linux using:
- Windows Task Scheduler + SCP client
- FTP server on Linux
- Cloud sync (Dropbox, OneDrive)

**Option C: Manual Transfer**
Copy files manually from Windows to Linux `data/raw_ea/` folder.

#### 2.3 Configure File Watcher

Create configuration file:
```bash
nano watcher_config.json
```

```json
{
  "watch_folder": "/mnt/rag_data",
  "check_interval": 30,
  "auto_convert_to_rag": true,
  "log_level": "INFO"
}
```

### Part 3: Automated Processing

#### 3.1 Test Manual Processing
```bash
# Test EA CSV processor
python scripts/ea_csv_processor.py

# Expected output:
# === EA CSV Processor ===
# Scanning for new files...
# Processing enhanced CSV: /mnt/rag_data/XAUUSD_enhanced_m5.csv
# Processed 43150 rows, saved to data/processed_XAUUSD_enhanced_m5.csv
# Files processed successfully!
```

#### 3.2 Start Automated File Watcher
```bash
# Start file watcher service
python scripts/file_watcher.py
```

The watcher will:
- Monitor the folder for new files
- Process CSV files and calculate indicators
- Convert to RAG format
- Import into knowledge base automatically

#### 3.3 Run as System Service (Optional)

Create systemd service:
```bash
sudo nano /etc/systemd/system/rag-watcher.service
```

```ini
[Unit]
Description=RAG Trading System File Watcher
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/opt/works/personal/rag_finance
ExecStart=/usr/bin/python3 scripts/file_watcher.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-watcher
sudo systemctl start rag-watcher
sudo systemctl status rag-watcher
```

## 📊 File Types Generated

### Basic CSV Files (OHLCV only)
```
XAUUSD_training_data_m5.csv
XAUUSD_training_data_m15.csv
XAUUSD_training_data_h1.csv
XAUUSD_training_data_h4.csv
```

**Format:**
```csv
timestamp,open,high,low,close,volume,spread
2025-01-28 10:00:00,2693.50,2694.21,2690.15,2693.77,45083,30
```

### Enhanced CSV Files (with indicators)
```
XAUUSD_enhanced_m5.csv
XAUUSD_enhanced_m15.csv
```

**Format:**
```csv
timestamp,open,high,low,close,volume,spread,rsi_14,macd,macd_signal,macd_hist,ema_20,ema_50,ema_200,atr_14,trend
2025-01-28 10:00:00,2693.50,2694.21,2690.15,2693.77,45083,30,68.5,2.34,1.89,0.45,2686.50,2680.30,2665.80,4.23,BULLISH
```

### Current Market JSON
```
XAUUSD_current_market.json
```

**Format:**
```json
{
  "query_timestamp": "2025-01-28 10:00:00",
  "symbol": "XAUUSD",
  "current_price": 2693.77,
  "ask": 2693.80,
  "bid": 2693.77,
  "spread": 30,
  "volume": 1500,
  "session": "US_SESSION",
  "day_of_week": "TUESDAY",
  "hour": 10
}
```

## 🔄 Processing Pipeline

### Automated Workflow
1. **EA Export** → CSV files created every 60 seconds
2. **File Detection** → Watcher detects new files
3. **CSV Processing** → Calculate indicators, enhance data
4. **RAG Conversion** → Convert to RAG format with trade setups
5. **Knowledge Base Import** → Store in vector database
6. **AI Analysis** → Ready for trading analysis

### Manual Processing Commands
```bash
# Step 1: Process CSV files
python scripts/ea_csv_processor.py

# Step 2: Convert to RAG format
python scripts/rag_converter.py

# Step 3: Import to knowledge base
python scripts/batch_import.py

# Step 4: Run analysis
python trading_rag.py
```

## 📈 Monitoring and Troubleshooting

### Check EA Status
- **MT5 Experts Tab**: Look for green smiley face
- **MT5 Journal Tab**: Check for export messages
- **Export Folder**: Verify files are being created

### Check Processing Status
```bash
# Check file watcher logs
tail -f logs/file_watcher.log

# Check processed files
ls -la data/processed_*
ls -la data/rag_*.json

# Check knowledge base
python -c "from trading_rag import TradingRAG; print(TradingRAG().get_knowledge_base_stats())"
```

### Common Issues and Solutions

**EA Not Exporting Files:**
- Verify EA is running (green smiley face)
- Check export folder permissions
- Ensure MT5 allows DLL imports
- Check MT5 Journal for errors

**File Watcher Not Detecting Files:**
- Verify folder path in configuration
- Check file permissions on Linux
- Ensure network share is mounted correctly
- Check watcher logs for errors

**Processing Errors:**
- Check CSV file format (should be UTF-8 encoded)
- Verify required columns are present
- Check disk space
- Review detailed error logs

**Performance Issues:**
- Increase `check_interval` in watcher config
- Reduce `ExportInterval` in EA settings
- Monitor CPU and memory usage

## ⚙️ Advanced Configuration

### Custom EA Parameters
```mql5
// Add to EA input parameters
input string   CustomSymbols   = "EURUSD,GBPUSD";     // Additional symbols
input int      MaxCandlesPerFile = 50000;             // Max candles per file
input bool     EnableCompression = true;              // Compress large files
```

### Custom Processing Rules
```python
# Modify ea_csv_processor.py to add custom indicators
def calculate_custom_indicator(df):
    # Your custom logic here
    df['custom_indicator'] = your_calculation(df)
    return df
```

### Multiple Folder Monitoring
```python
# Watch multiple folders
folders = ["/mnt/rag_data", "/backup/rag_data", "/cloud/rag_data"]
for folder in folders:
    watcher = FileWatcher(watch_folder=folder)
    watcher.start_watching()
```

## 🔒 Security Considerations

### File Access Security
- Use dedicated user account for file access
- Set appropriate folder permissions
- Consider encryption for sensitive data
- Use SSH tunnels for remote file transfer

### Network Security
- Use VPN for network share access
- Configure firewall rules appropriately
- Use SFTP instead of CIFS if possible
- Regularly update MT5 and system patches

## 📊 Performance Optimization

### EA Optimization
- Increase `ExportInterval` to reduce disk I/O
- Use `EnableCompression` for large datasets
- Limit historical data days for faster initial export

### Processing Optimization
- Use SSD storage for better I/O performance
- Increase `check_interval` to reduce CPU usage
- Process files in batches during off-peak hours

### System Monitoring
```bash
# Monitor disk usage
df -h /mnt/rag_data

# Monitor system resources
htop

# Monitor file watcher process
ps aux | grep file_watcher
```

## 🚀 Next Steps

1. **Backtest System**: Validate with historical data
2. **Paper Trading**: Test with real-time data
3. **Performance Tuning**: Optimize for your specific setup
4. **Multiple Symbols**: Add other currency pairs or assets
5. **Advanced Analytics**: Implement custom indicators and strategies

---

## 📞 Support

For issues with:
- **EA**: Check MT5 Journal and Experts log
- **File Processing**: Review Python logs in `logs/` directory
- **Network**: Test folder access and permissions
- **Performance**: Monitor system resources and adjust intervals

The EA-based approach provides reliable, automated data export with minimal network dependencies, making it ideal for production environments.