//+------------------------------------------------------------------+
//|                                           RAG_DataExporter.mq5 |
//|                        Copyright 2023, RAG Trading System       |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, RAG Trading System"
#property link      "https://example.com"
#property version   "1.00"
#property description "RAG Trading System - Data Exporter EA"

//--- Input parameters
input group "Export Settings"
input string   ExportFolder   = "C:\\RAG_Data\\";      // Export folder path
input int      ExportInterval  = 60;                    // Export interval in seconds
input bool     ExportHistorical = true;                // Export historical data on start
input int      HistoricalDays   = 180;                  // Historical days to export

input group "Symbols and Timeframes"
input string   SymbolsToExport = "XAUUSD";             // Symbols to export (comma separated)
input string   TimeframesToExport = "M5,M15,H1,H4,D1"; // Timeframes to export

input group "File Settings"
input string   FilePrefix      = "XAUUSD_";            // File name prefix
input bool     CompressFiles    = false;               // Compress CSV files

//--- Global variables
datetime lastExportTime = 0;
string exportSymbols[];
string exportTimeframes[];
int fileHandle = -1;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("=== RAG Data Exporter EA Initialized ===");

    // Create export folder if it doesn't exist
    if(!FolderCreate(ExportFolder, FILE_COMMON))
    {
        Print("Error creating export folder: ", GetLastError());
        return(INIT_FAILED);
    }

    // Parse symbols
    ParseStringParameter(SymbolsToExport, exportSymbols);

    // Parse timeframes
    ParseStringParameter(TimeframesToExport, exportTimeframes);

    Print("Export folder: ", ExportFolder);
    Print("Symbols to export: ", SymbolsToExport);
    Print("Timeframes to export: ", TimeframesToExport);

    // Export historical data if requested
    if(ExportHistorical)
    {
        ExportAllHistoricalData();
    }

    // Set timer for regular exports
    EventSetTimer(ExportInterval);

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    CloseAllFiles();
    Print("=== RAG Data Exporter EA Deinitialized ===");
}

//+------------------------------------------------------------------+
//| Expert timer function                                           |
//+------------------------------------------------------------------+
void OnTimer()
{
    datetime currentTime = TimeCurrent();

    // Export current data for all symbols and timeframes
    for(int i = 0; i < ArraySize(exportSymbols); i++)
    {
        string symbol = exportSymbols[i];

        // Export current market data
        ExportCurrentMarketData(symbol);

        // Export timeframe data
        for(int j = 0; j < ArraySize(exportTimeframes); j++)
        {
            string tfStr = exportTimeframes[j];
            ENUM_TIMEFRAMES tf = StringToTimeframe(tfStr);

            if(tf != PERIOD_CURRENT)
            {
                ExportTimeframeData(symbol, tf, tfStr);
            }
        }
    }

    lastExportTime = currentTime;
    Print("Data export completed at: ", TimeToString(currentTime));
}

//+------------------------------------------------------------------+
//| Parse comma-separated string parameter                           |
//+------------------------------------------------------------------+
void ParseStringParameter(string inputValue, string &output[])
{
    StringReplace(inputValue, " ", "");  // Remove spaces
    string parts[];
    StringSplit(inputValue, ',', parts);

    ArrayResize(output, ArraySize(parts));
    for(int i = 0; i < ArraySize(parts); i++)
    {
        output[i] = parts[i];
    }
}

//+------------------------------------------------------------------+
//| Convert string timeframe to ENUM_TIMEFRAMES                      |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES StringToTimeframe(string tfStr)
{
    if(tfStr == "M1")  return PERIOD_M1;
    if(tfStr == "M5")  return PERIOD_M5;
    if(tfStr == "M15") return PERIOD_M15;
    if(tfStr == "M30") return PERIOD_M30;
    if(tfStr == "H1")  return PERIOD_H1;
    if(tfStr == "H4")  return PERIOD_H4;
    if(tfStr == "D1")  return PERIOD_D1;
    if(tfStr == "W1")  return PERIOD_W1;
    if(tfStr == "MN1") return PERIOD_MN1;

    return PERIOD_CURRENT;  // Invalid timeframe
}

//+------------------------------------------------------------------+
//| Export all historical data                                      |
//+------------------------------------------------------------------+
void ExportAllHistoricalData()
{
    Print("Starting historical data export...");

    datetime endDate = TimeCurrent();
    datetime startDate = endDate - (HistoricalDays * 24 * 60 * 60);

    for(int i = 0; i < ArraySize(exportSymbols); i++)
    {
        string symbol = exportSymbols[i];

        for(int j = 0; j < ArraySize(exportTimeframes); j++)
        {
            string tfStr = exportTimeframes[j];
            ENUM_TIMEFRAMES tf = StringToTimeframe(tfStr);

            if(tf != PERIOD_CURRENT)
            {
                Print("Exporting historical data for ", symbol, " ", tfStr, "...");
                ExportHistoricalData(symbol, tf, tfStr, startDate, endDate);
            }
        }
    }

    Print("Historical data export completed!");
}

//+------------------------------------------------------------------+
//| Export historical data for specific symbol and timeframe         |
//+------------------------------------------------------------------+
void ExportHistoricalData(string symbol, ENUM_TIMEFRAMES tf, string tfStr, datetime startDate, datetime endDate)
{
    // Get historical rates
    MqlRates rates[];
    int copied = CopyRates(symbol, tf, startDate, endDate, rates);

    if(copied <= 0)
    {
        Print("Error copying historical rates for ", symbol, " ", tfStr, ": ", GetLastError());
        return;
    }

    // Create filename
    string lowerTfStr = tfStr;
    StringToLower(lowerTfStr);
    string filename = ExportFolder + FilePrefix + "training_data_" + lowerTfStr + ".csv";

    // Open file
    int handle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI, ",");
    if(handle == INVALID_HANDLE)
    {
        Print("Error opening file ", filename, ": ", GetLastError());
        return;
    }

    // Write header
    FileWrite(handle, "timestamp", "open", "high", "low", "close", "volume", "spread");

    // Write data
    for(int i = 0; i < copied; i++)
    {
        FileWrite(handle,
            TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS),
            DoubleToString(rates[i].open, 5),
            DoubleToString(rates[i].high, 5),
            DoubleToString(rates[i].low, 5),
            DoubleToString(rates[i].close, 5),
            IntegerToString(rates[i].tick_volume),
            IntegerToString(rates[i].spread)
        );
    }

    FileClose(handle);
    Print("Exported ", copied, " records to ", filename);
}

//+------------------------------------------------------------------+
//| Export current market data                                       |
//+------------------------------------------------------------------+
void ExportCurrentMarketData(string symbol)
{
    MqlTick tick;
    if(!SymbolInfoTick(symbol, tick))
    {
        Print("Error getting tick for ", symbol, ": ", GetLastError());
        return;
    }

    // Get market info
    double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    long volume = SymbolInfoLong(symbol, SYMBOL_VOLUME);
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);

    // Get session info
    datetime currentTimeVar = TimeCurrent();
    int hour = TimeHour(currentTimeVar);
    string session = GetSessionString(hour);
    string dayOfWeek = TimeDayOfWeekToString(TimeDayOfWeek(currentTimeVar));

    // Create JSON data
    string jsonFilename = ExportFolder + FilePrefix + "current_market.json";
    int handle = FileOpen(jsonFilename, FILE_WRITE | FILE_TXT | FILE_ANSI);
    if(handle != INVALID_HANDLE)
    {
        FileWrite(handle, "{");
        FileWrite(handle, "  \"query_timestamp\": \"" + TimeToString(currentTimeVar, TIME_DATE|TIME_SECONDS) + "\",");
        FileWrite(handle, "  \"symbol\": \"" + symbol + "\",");
        FileWrite(handle, "  \"current_price\": " + DoubleToString(bid, digits) + ",");
        FileWrite(handle, "  \"ask\": " + DoubleToString(ask, digits) + ",");
        FileWrite(handle, "  \"bid\": " + DoubleToString(bid, digits) + ",");
        FileWrite(handle, "  \"spread\": " + IntegerToString((int)((ask - bid) / point)) + ",");
        FileWrite(handle, "  \"volume\": " + IntegerToString(volume) + ",");
        FileWrite(handle, "  \"session\": \"" + session + "\",");
        FileWrite(handle, "  \"day_of_week\": \"" + dayOfWeek + "\",");
        FileWrite(handle, "  \"hour\": " + IntegerToString(hour));
        FileWrite(handle, "}");
        FileClose(handle);
    }
}

//+------------------------------------------------------------------+
//| Export timeframe data                                           |
//+------------------------------------------------------------------+
void ExportTimeframeData(string symbol, ENUM_TIMEFRAMES tf, string tfStr)
{
    // Get recent candles (last 200 for indicator calculations)
    MqlRates rates[];
    int copied = CopyRates(symbol, tf, 0, 200, rates);

    if(copied <= 0)
    {
        Print("Error copying rates for ", symbol, " ", tfStr, ": ", GetLastError());
        return;
    }

    // Calculate indicators
    double rsi[], macd[], macdSignal[], macdHist[], ema20[], ema50[], ema200[], atr[];
    ArrayResize(rsi, copied);
    ArrayResize(macd, copied);
    ArrayResize(macdSignal, copied);
    ArrayResize(macdHist, copied);
    ArrayResize(ema20, copied);
    ArrayResize(ema50, copied);
    ArrayResize(ema200, copied);
    ArrayResize(atr, copied);

    // Calculate technical indicators
    for(int i = 0; i < copied; i++)
    {
        // RSI (14)
        if(i >= 14)
            rsi[i] = CalculateRSI(symbol, tf, i, 14);

        // EMAs
        ema20[i] = CalculateEMA(symbol, tf, i, 20);
        ema50[i] = CalculateEMA(symbol, tf, i, 50);
        ema200[i] = CalculateEMA(symbol, tf, i, 200);

        // ATR (14)
        if(i >= 14)
            atr[i] = CalculateATR(symbol, tf, i, 14);
    }

    // MACD (12, 26, 9)
    CalculateMACD(symbol, tf, macd, macdSignal, macdHist);

    // Determine trend
    string trend = DetermineTrend(rates[copied-1].close, ema20[copied-1], ema50[copied-1], ema200[copied-1]);

    // Create enhanced CSV filename
    string lowerTfStr = tfStr;
    StringToLower(lowerTfStr);
    string enhancedFilename = ExportFolder + FilePrefix + "enhanced_" + lowerTfStr + ".csv";

    // Open enhanced file
    int handle = FileOpen(enhancedFilename, FILE_WRITE | FILE_CSV | FILE_ANSI, ",");
    if(handle == INVALID_HANDLE)
    {
        Print("Error opening enhanced file ", enhancedFilename, ": ", GetLastError());
        return;
    }

    // Write enhanced header
    FileWrite(handle,
        "timestamp", "open", "high", "low", "close", "volume", "spread",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "ema_20", "ema_50", "ema_200", "atr_14", "trend"
    );

    // Write enhanced data (last 50 candles with indicators)
    int startIdx = MathMax(0, copied - 50);
    for(int i = startIdx; i < copied; i++)
    {
        FileWrite(handle,
            TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS),
            DoubleToString(rates[i].open, 5),
            DoubleToString(rates[i].high, 5),
            DoubleToString(rates[i].low, 5),
            DoubleToString(rates[i].close, 5),
            IntegerToString(rates[i].tick_volume),
            IntegerToString(rates[i].spread),
            DoubleToString(rsi[i], 2),
            DoubleToString(macd[i], 5),
            DoubleToString(macdSignal[i], 5),
            DoubleToString(macdHist[i], 5),
            DoubleToString(ema20[i], 5),
            DoubleToString(ema50[i], 5),
            DoubleToString(ema200[i], 5),
            DoubleToString(atr[i], 5),
            trend
        );
    }

    FileClose(handle);
    Print("Exported enhanced data to ", enhancedFilename);
}

//+------------------------------------------------------------------+
//| Calculate RSI                                                   |
//+------------------------------------------------------------------+
double CalculateRSI(string symbol, ENUM_TIMEFRAMES tf, int shift, int period)
{
    double prices[];
    ArrayResize(prices, period + 1);

    for(int i = 0; i <= period; i++)
    {
        prices[i] = iClose(symbol, tf, shift + i);
    }

    double gains = 0, losses = 0;
    for(int i = 1; i <= period; i++)
    {
        double change = prices[i-1] - prices[i];
        if(change > 0)
            gains += change;
        else
            losses -= change;
    }

    gains /= period;
    losses /= period;

    if(losses == 0) return 100;
    double rs = gains / losses;
    return 100 - (100 / (1 + rs));
}

//+------------------------------------------------------------------+
//| Calculate EMA                                                   |
//+------------------------------------------------------------------+
double CalculateEMA(string symbol, ENUM_TIMEFRAMES tf, int shift, int period)
{
    double ema = iClose(symbol, tf, shift);
    double multiplier = 2.0 / (period + 1);

    for(int i = 1; i <= period; i++)
    {
        double price = iClose(symbol, tf, shift + i);
        ema = (price * multiplier) + (ema * (1 - multiplier));
    }

    return ema;
}

//+------------------------------------------------------------------+
//| Calculate ATR                                                   |
//+------------------------------------------------------------------+
double CalculateATR(string symbol, ENUM_TIMEFRAMES tf, int shift, int period)
{
    double trSum = 0;

    for(int i = 0; i < period; i++)
    {
        double high = iHigh(symbol, tf, shift + i);
        double low = iLow(symbol, tf, shift + i);
        double closePrev = iClose(symbol, tf, shift + i + 1);

        double tr = MathMax(high - low, MathMax(MathAbs(high - closePrev), MathAbs(low - closePrev)));
        trSum += tr;
    }

    return trSum / period;
}

//+------------------------------------------------------------------+
//| Calculate MACD                                                  |
//+------------------------------------------------------------------+
void CalculateMACD(string symbol, ENUM_TIMEFRAMES tf, double &macd[], double &signal[], double &histogram[])
{
    int size = ArraySize(macd);
    double ema12[], ema26[];
    ArrayResize(ema12, size);
    ArrayResize(ema26, size);

    // Calculate EMAs
    for(int i = 0; i < size; i++)
    {
        ema12[i] = CalculateEMA(symbol, tf, i, 12);
        ema26[i] = CalculateEMA(symbol, tf, i, 26);
        macd[i] = ema12[i] - ema26[i];
    }

    // Calculate signal line (9-period EMA of MACD)
    for(int i = 0; i < size; i++)
    {
        signal[i] = CalculateEMAFromArray(macd, i, 9);
        histogram[i] = macd[i] - signal[i];
    }
}

//+------------------------------------------------------------------+
//| Calculate EMA from array                                       |
//+------------------------------------------------------------------+
double CalculateEMAFromArray(double &array[], int shift, int period)
{
    double ema = array[shift];
    double multiplier = 2.0 / (period + 1);

    for(int i = 1; i <= period; i++)
    {
        if(shift + i < ArraySize(array))
        {
            ema = (array[shift + i] * multiplier) + (ema * (1 - multiplier));
        }
    }

    return ema;
}

//+------------------------------------------------------------------+
//| Determine trend based on EMAs                                   |
//+------------------------------------------------------------------+
string DetermineTrend(double price, double ema20, double ema50, double ema200)
{
    if(price > ema20 && ema20 > ema50 && ema50 > ema200)
        return "BULLISH";
    else if(price < ema20 && ema20 < ema50 && ema50 < ema200)
        return "BEARISH";
    else
        return "NEUTRAL";
}

//+------------------------------------------------------------------+
//| Get session string                                               |
//+------------------------------------------------------------------+
string GetSessionString(int hour)
{
    if(hour >= 0 && hour < 8)
        return "ASIAN_SESSION";
    else if(hour >= 8 && hour < 13)
        return "LONDON_SESSION";
    else if(hour >= 13 && hour < 20)
        return "US_SESSION";
    else
        return "AFTER_HOURS";
}

//+------------------------------------------------------------------+
//| Convert day of week to string                                   |
//+------------------------------------------------------------------+
string TimeDayOfWeekToString(int dayOfWeek)
{
    switch(dayOfWeek)
    {
        case 0: return "SUNDAY";
        case 1: return "MONDAY";
        case 2: return "TUESDAY";
        case 3: return "WEDNESDAY";
        case 4: return "THURSDAY";
        case 5: return "FRIDAY";
        case 6: return "SATURDAY";
        default: return "UNKNOWN";
    }
}

//+------------------------------------------------------------------+
//| Close all open files                                            |
//+------------------------------------------------------------------+
void CloseAllFiles()
{
    if(fileHandle != INVALID_HANDLE)
    {
        FileClose(fileHandle);
        fileHandle = INVALID_HANDLE;
    }
}

//+------------------------------------------------------------------+
//| Expert tick function (optional)                                  |
//+------------------------------------------------------------------+
void OnTick()
{
    // Optional: Process on each tick if needed
    // Most processing is done via timer
}

//+------------------------------------------------------------------+