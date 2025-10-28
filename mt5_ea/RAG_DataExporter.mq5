//+------------------------------------------------------------------+
//|                                  RAG_Enhanced_MultiTimeframeExporter_Fixed.mq5 |
//|                        Enhanced Multi-Timeframe Data Exporter for RAG Trading System |
//+------------------------------------------------------------------+
#property copyright "RAG-Enhanced Multi-Timeframe Data Exporter"
#property version   "2.00"
#property strict

//--- Input parameters
input group "===== SYMBOL SETTINGS =====";
input string Symbols = "XAUUSD,EURUSD,BTCUSD"; // Symbols to export (comma separated)
input bool UseCurrentSymbolOnly = false; // Use only current chart symbol

input group "===== EXPORT SETTINGS =====";
input string ExportPath = ""; // Export path (empty = terminal data folder)
input bool AutoExport = true; // Auto-export on attach
input bool IncludeSupportResistance = true; // Calculate S/R levels
input bool AutoUpload = true; // Auto-upload to server
input string UploadURL = "https://ml.ovh139.aliases.me/upload"; // Upload URL

input group "===== RAG TRAINING DATA SETTINGS =====";
input bool EnableRAGFormat = true; // Enable RAG-ready data format
input bool IncludeTechnicalIndicators = true; // Include RSI, MACD, EMA calculations
input bool IncludeMarketContext = true; // Include session, day of week, market sentiment
input bool IncludePatternRecognition = true; // Include candlestick pattern analysis
input bool IncludeTrendAnalysis = true; // Include multi-timeframe trend analysis
input int MinProfitPoints = 5; // Minimum profit for trade examples
input bool SimulateTradeOutcomes = true; // Simulate trade outcomes for training data

input group "===== HISTORICAL EXPORT SETTINGS =====";
input bool EnableHistoricalExport = false; // Enable historical export for ML training
input int HistoricalDaysBack = 30; // Number of days to export (1-365)
input bool ExportDailyTargetOnly = true; // Current day: export daily OHLC only (target data)
input bool IncludeWeekends = false; // Include weekend data in exports

input group "===== TECHNICAL INDICATOR SETTINGS =====";
input int RSI_Period = 14; // RSI calculation period
input int MACD_Fast = 12; // MACD fast EMA period
input int MACD_Slow = 26; // MACD slow EMA period
input int MACD_Signal = 9; // MACD signal line period
input int EMA_Short = 20; // Short-term EMA period
input int EMA_Medium = 50; // Medium-term EMA period
input int EMA_Long = 200; // Long-term EMA period
input int ATR_Period = 14; // ATR calculation period
input int BB_Period = 20; // Bollinger Bands period
input double BB_Deviation = 2.0; // Bollinger Bands deviation

input group "===== TIMEFRAMES =====";
input bool ExportM1 = true; // Export 1-Min
input int M1_Candles = 200; // 1-Min candles
input bool ExportM5 = true; // Export 5-Min
input int M5_Candles = 200; // 5-Min candles
input bool ExportM15 = true; // Export 15-Min
input int M15_Candles = 150; // 15-Min candles
input bool ExportM30 = true; // Export 30-Min
input int M30_Candles = 100; // 30-Min candles
input bool ExportH1 = true; // Export 1-Hour
input int H1_Candles = 100; // 1-Hour candles
input bool ExportH4 = true; // Export 4-Hour
input int H4_Candles = 48; // 4-Hour candles
input bool ExportD1 = true; // Export Daily
input int D1_Candles = 30; // Daily candles

input group "===== SUPPORT/RESISTANCE SETTINGS =====";
input int SR_LookbackBars = 100; // Lookback bars for S/R
input int SR_TouchTolerance = 10; // Price touch tolerance (points)
input int SR_MinTouches = 2; // Minimum touches to confirm level
input int SR_MaxLevels = 10; // Maximum S/R levels to find

input group "===== MARKET PROFILE SETTINGS =====";
input bool EnableMarketProfile = true; // Enable Market Profile calculations
input double ValueAreaPercentage = 70.0; // Percentage for Value Area calculation

//--- Global variables
struct SRLevel
{
    double price;
    int touches;
    string type; // "Support" or "Resistance"
};

struct TechnicalIndicators
{
    double rsi;
    double macd;
    double macd_signal;
    double macd_histogram;
    double ema_short;
    double ema_medium;
    double ema_long;
    double atr;
    double bb_upper;
    double bb_middle;
    double bb_lower;
    long volume_avg;
    string trend;
};

struct MarketContext
{
    string session;
    string day_of_week;
    int hour;
    double spread;
    double volatility;
    string market_sentiment;
};

struct CandlestickPattern
{
    string name;
    string signal; // "BULLISH", "BEARISH", "NEUTRAL"
    double confidence;
};

struct TradeExample
{
    string id;
    datetime timestamp;
    string symbol;
    double entry_price;
    double stop_loss;
    double take_profit1;
    double take_profit2;
    double take_profit3;
    double dca1;
    double dca2;
    double dca3;
    string direction; // "LONG", "SHORT", "NEUTRAL"
    string setup_type;
    string outcome;
    double profit_points;
    bool win;
    double confidence;
    string reasoning;
};

// Global variables
string g_symbolList[]; // Parsed symbol list
int g_symbolCount = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("========================================");
    Print("RAG-Enhanced Multi-Timeframe Data Exporter Started");
    Print("========================================");

    // Parse symbol list
    ParseSymbols();

    if(AutoExport)
    {
        Print("Starting enhanced export for ", g_symbolCount, " symbol(s)...");

        if(EnableHistoricalExport)
        {
            Print("Historical RAG training export mode enabled - exporting ", HistoricalDaysBack, " days of data...");
            ExportRAGHistoricalData();
        }
        else
        {
            ExportAllSymbolsRAG();
        }
    }

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Parse symbols from input string                                   |
//+------------------------------------------------------------------+
void ParseSymbols()
{
    if(UseCurrentSymbolOnly)
    {
        g_symbolCount = 1;
        ArrayResize(g_symbolList, 1);
        g_symbolList[0] = _Symbol;
        Print("Using current symbol only: ", _Symbol);
        return;
    }

    g_symbolCount = StringSplit(Symbols, ',', g_symbolList);

    // Trim whitespace from symbol names
    for(int i = 0; i < g_symbolCount; i++)
    {
        StringTrimLeft(g_symbolList[i]);
        StringTrimRight(g_symbolList[i]);
    }

    Print("Loaded ", g_symbolCount, " symbols: ", Symbols);
}

//+------------------------------------------------------------------+
//| Export RAG historical data for ML training                        |
//+------------------------------------------------------------------+
void ExportRAGHistoricalData()
{
    // Validate historical days parameter
    if(HistoricalDaysBack < 1 || HistoricalDaysBack > 365)
    {
        Print("ERROR: HistoricalDaysBack must be between 1 and 365. Current value: ", HistoricalDaysBack);
        return;
    }

    datetime currentTime = TimeCurrent();
    Print("Starting RAG historical export for ", HistoricalDaysBack, " days...");

    // Collect valid trading days
    datetime validDates[];
    int validCount = 0;

    for(int dayOffset = 0; dayOffset <= HistoricalDaysBack; dayOffset++)
    {
        datetime checkDate = currentTime - (dayOffset * 24 * 3600);
        MqlDateTime checkDt;
        TimeToStruct(checkDate, checkDt);

        if(!IncludeWeekends && (checkDt.day_of_week == 0 || checkDt.day_of_week == 6))
        {
            continue;
        }

        ArrayResize(validDates, validCount + 1);
        validDates[validCount] = checkDate;
        validCount++;
    }

    // Create RAG training data files
    for(int i = validCount - 1; i >= 0; i--)
    {
        datetime currentDate = validDates[i];
        string currentDateStr = TimeToString(currentDate, TIME_DATE);

        for(int symbolIdx = 0; symbolIdx < g_symbolCount; symbolIdx++)
        {
            string symbol = g_symbolList[symbolIdx];

            // Export RAG training data (full analysis)
            ExportRAGTrainingData(symbol, currentDate, currentDateStr);
        }
    }

    Print("RAG historical export completed!");
}

//+------------------------------------------------------------------+
//| Export RAG training data for specific date                        |
//+------------------------------------------------------------------+
void ExportRAGTrainingData(string symbol, datetime currentDate, string dateStr)
{
    string filename = "RAG_" + symbol + "_" + dateStr + ".csv";

    if(ExportPath != "")
    {
        if(!MQLInfoInteger(MQL_TESTER))
            filename = ExportPath + "\\" + filename;
    }

    int fileHandle = FileOpen(filename, FILE_WRITE|FILE_ANSI|FILE_TXT);

    if(fileHandle == INVALID_HANDLE)
    {
        Print("Failed to create RAG training file: ", filename);
        return;
    }

    Print("Exporting RAG training data: ", filename);

    // Write RAG training header
    FileWrite(fileHandle, "RAG Training Data for " + symbol + " - " + dateStr);
    FileWrite(fileHandle, "Generated: " + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS));
    FileWrite(fileHandle, "");

    // Export multi-timeframe analysis with indicators
    if(ExportM5)
    {
        ExportRAGTimeframe(fileHandle, symbol, PERIOD_M5, M5_Candles, currentDate, "M5");
    }

    if(ExportM15)
    {
        ExportRAGTimeframe(fileHandle, symbol, PERIOD_M15, M15_Candles, currentDate, "M15");
    }

    if(ExportH1)
    {
        ExportRAGTimeframe(fileHandle, symbol, PERIOD_H1, H1_Candles, currentDate, "H1");
    }

    if(ExportH4)
    {
        ExportRAGTimeframe(fileHandle, symbol, PERIOD_H4, H4_Candles, currentDate, "H4");
    }

    // Generate trade examples based on analysis
    if(SimulateTradeOutcomes)
    {
        FileWrite(fileHandle, "");
        FileWrite(fileHandle, "===========================================");
        FileWrite(fileHandle, "SIMULATED TRADE EXAMPLES FOR RAG TRAINING");
        FileWrite(fileHandle, "===========================================");
        GenerateTradeExamples(fileHandle, symbol, currentDate);
    }

    FileClose(fileHandle);

    // Upload to server if enabled
    if(AutoUpload)
    {
        UploadFileToServer(filename);
    }
}

//+------------------------------------------------------------------+
//| Export RAG-format timeframe data                                  |
//+------------------------------------------------------------------+
void ExportRAGTimeframe(int fileHandle, string symbol, ENUM_TIMEFRAMES timeframe, int candles, datetime analysisDate, string tfName)
{
    FileWrite(fileHandle, "");
    FileWrite(fileHandle, "===========================================");
    FileWrite(fileHandle, "RAG ANALYSIS: " + tfName + " TIMEFRAME");
    FileWrite(fileHandle, "===========================================");

    // Get rate data
    MqlRates rates[];
    ArraySetAsSeries(rates, true);

    datetime endDate = analysisDate + (24 * 3600);
    int copied = CopyRates(symbol, timeframe, endDate, candles * 2, rates);

    if(copied <= 0)
    {
        FileWrite(fileHandle, "ERROR: Failed to copy data for " + tfName);
        return;
    }

    // Calculate technical indicators
    TechnicalIndicators indicators[];
    ArrayResize(indicators, copied);

    for(int i = 0; i < copied; i++)
    {
        CalculateTechnicalIndicators(rates, i, indicators[i]);
    }

    // Write column headers for RAG format
    FileWrite(fileHandle, "Timestamp,Open,High,Low,Close,Volume,RSI,MACD,MACD_Signal,MACD_Hist,EMA_Short,EMA_Medium,EMA_Long,ATR,BB_Upper,BB_Middle,BB_Lower,Volume_Avg,Trend,Session,Day_of_Week,Hour,Pattern,Signal,Confidence");

    // Write data rows with full analysis
    for(int i = copied - 1; i >= 0; i--)
    {
        if(rates[i].time >= analysisDate && rates[i].time < analysisDate + (24 * 3600))
        {
            // Calculate market context
            MarketContext context;
            CalculateMarketContext(rates[i].time, context);

            // Identify candlestick pattern
            CandlestickPattern pattern;
            IdentifyCandlestickPattern(rates, i, pattern);

            string line = StringFormat("%s,%.5f,%.5f,%.5f,%.5f,%lld,%.2f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%lld,%s,%s,%s,%d,%s,%s,%.1f",
                                       TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES),
                                       rates[i].open,
                                       rates[i].high,
                                       rates[i].low,
                                       rates[i].close,
                                       rates[i].tick_volume,
                                       indicators[i].rsi,
                                       indicators[i].macd,
                                       indicators[i].macd_signal,
                                       indicators[i].macd_histogram,
                                       indicators[i].ema_short,
                                       indicators[i].ema_medium,
                                       indicators[i].ema_long,
                                       indicators[i].atr,
                                       indicators[i].bb_upper,
                                       indicators[i].bb_middle,
                                       indicators[i].bb_lower,
                                       indicators[i].volume_avg,
                                       indicators[i].trend,
                                       context.session,
                                       context.day_of_week,
                                       context.hour,
                                       pattern.name,
                                       pattern.signal,
                                       pattern.confidence);

            FileWrite(fileHandle, line);
        }
    }

    // Calculate and export support/resistance levels
    if(IncludeSupportResistance)
    {
        FileWrite(fileHandle, "");
        FileWrite(fileHandle, "Support/Resistance Analysis:");
        SRLevel supports[], resistances[];
        CalculateSupportResistance(symbol, timeframe, supports, resistances);

        FileWrite(fileHandle, "Support Levels:");
        for(int i = 0; i < ArraySize(supports); i++)
        {
            FileWrite(fileHandle, StringFormat("Level_%d: %.5f (Touches: %d)", i+1, supports[i].price, supports[i].touches));
        }

        FileWrite(fileHandle, "Resistance Levels:");
        for(int i = 0; i < ArraySize(resistances); i++)
        {
            FileWrite(fileHandle, StringFormat("Level_%d: %.5f (Touches: %d)", i+1, resistances[i].price, resistances[i].touches));
        }
    }
}

//+------------------------------------------------------------------+
//| Calculate technical indicators                                     |
//+------------------------------------------------------------------+
void CalculateTechnicalIndicators(MqlRates &rates[], int index, TechnicalIndicators &indicators)
{
    if(index < RSI_Period || index < MACD_Slow || index < EMA_Long || index < ATR_Period || index < BB_Period)
    {
        // Not enough data for calculation
        indicators.rsi = 50;
        indicators.macd = 0;
        indicators.macd_signal = 0;
        indicators.macd_histogram = 0;
        indicators.ema_short = rates[index].close;
        indicators.ema_medium = rates[index].close;
        indicators.ema_long = rates[index].close;
        indicators.atr = 0;
        indicators.bb_upper = rates[index].close;
        indicators.bb_middle = rates[index].close;
        indicators.bb_lower = rates[index].close;
        indicators.volume_avg = rates[index].tick_volume;
        indicators.trend = "NEUTRAL";
        return;
    }

    // Calculate RSI
    double gains = 0, losses = 0;
    for(int i = index - RSI_Period + 1; i <= index; i++)
    {
        double change = rates[i].close - rates[i-1].close;
        if(change > 0) gains += change;
        else losses -= change;
    }
    double avgGain = gains / RSI_Period;
    double avgLoss = losses / RSI_Period;
    double rs = avgGain / avgLoss;
    indicators.rsi = 100 - (100 / (1 + rs));

    // Calculate EMAs
    double sum_short = 0, sum_medium = 0, sum_long = 0;
    for(int i = index - EMA_Long + 1; i <= index; i++)
    {
        if(i >= index - EMA_Short + 1) sum_short += rates[i].close;
        if(i >= index - EMA_Medium + 1) sum_medium += rates[i].close;
        sum_long += rates[i].close;
    }
    indicators.ema_short = sum_short / EMA_Short;
    indicators.ema_medium = sum_medium / EMA_Medium;
    indicators.ema_long = sum_long / EMA_Long;

    // Calculate MACD
    double ema_fast = 0, ema_slow = 0;
    for(int i = index - MACD_Slow + 1; i <= index; i++)
    {
        if(i >= index - MACD_Fast + 1) ema_fast += rates[i].close;
        ema_slow += rates[i].close;
    }
    ema_fast /= MACD_Fast;
    ema_slow /= MACD_Slow;
    indicators.macd = ema_fast - ema_slow;

    // Simple MACD signal calculation
    double macd_sum = 0;
    for(int i = index - MACD_Signal + 1; i <= index; i++)
    {
        macd_sum += indicators.macd;
    }
    indicators.macd_signal = macd_sum / MACD_Signal;
    indicators.macd_histogram = indicators.macd - indicators.macd_signal;

    // Calculate ATR
    double tr_sum = 0;
    for(int i = index - ATR_Period + 1; i <= index; i++)
    {
        double high_low = rates[i].high - rates[i].low;
        double high_close = MathAbs(rates[i].high - rates[i-1].close);
        double low_close = MathAbs(rates[i].low - rates[i-1].close);
        double tr = MathMax(high_low, MathMax(high_close, low_close));
        tr_sum += tr;
    }
    indicators.atr = tr_sum / ATR_Period;

    // Calculate Bollinger Bands
    double sum = 0, sum_sq = 0;
    for(int i = index - BB_Period + 1; i <= index; i++)
    {
        sum += rates[i].close;
        sum_sq += rates[i].close * rates[i].close;
    }
    indicators.bb_middle = sum / BB_Period;
    double variance = (sum_sq - sum * sum / BB_Period) / BB_Period;
    double std_dev = MathSqrt(variance);
    indicators.bb_upper = indicators.bb_middle + BB_Deviation * std_dev;
    indicators.bb_lower = indicators.bb_middle - BB_Deviation * std_dev;

    // Calculate volume average
    long vol_sum = 0;
    for(int i = index - 20 + 1; i <= index; i++)
    {
        vol_sum += rates[i].tick_volume;
    }
    indicators.volume_avg = vol_sum / 20;

    // Determine trend
    if(rates[index].close > indicators.ema_short &&
       indicators.ema_short > indicators.ema_medium &&
       indicators.ema_medium > indicators.ema_long)
    {
        indicators.trend = "BULLISH";
    }
    else if(rates[index].close < indicators.ema_short &&
            indicators.ema_short < indicators.ema_medium &&
            indicators.ema_medium < indicators.ema_long)
    {
        indicators.trend = "BEARISH";
    }
    else
    {
        indicators.trend = "NEUTRAL";
    }
}

//+------------------------------------------------------------------+
//| Calculate market context                                          |
//+------------------------------------------------------------------+
void CalculateMarketContext(datetime timestamp, MarketContext &context)
{
    MqlDateTime dt;
    TimeToStruct(timestamp, dt);

    context.hour = dt.hour;

    // Determine session
    if(dt.hour >= 0 && dt.hour < 8)
        context.session = "ASIAN_SESSION";
    else if(dt.hour >= 8 && dt.hour < 13)
        context.session = "LONDON_SESSION";
    else if(dt.hour >= 13 && dt.hour < 20)
        context.session = "US_SESSION";
    else
        context.session = "AFTER_HOURS";

    // Determine day of week name
    switch(dt.day_of_week)
    {
        case 0: context.day_of_week = "SUNDAY"; break;
        case 1: context.day_of_week = "MONDAY"; break;
        case 2: context.day_of_week = "TUESDAY"; break;
        case 3: context.day_of_week = "WEDNESDAY"; break;
        case 4: context.day_of_week = "THURSDAY"; break;
        case 5: context.day_of_week = "FRIDAY"; break;
        case 6: context.day_of_week = "SATURDAY"; break;
        default: context.day_of_week = "UNKNOWN"; break;
    }

    // Calculate spread (simplified)
    context.spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);

    // Market sentiment based on time and session
    if(context.session == "US_SESSION" || context.session == "LONDON_SESSION")
        context.market_sentiment = "RISK_ON";
    else
        context.market_sentiment = "RISK_OFF";
}

//+------------------------------------------------------------------+
//| Identify candlestick patterns                                     |
//+------------------------------------------------------------------+
void IdentifyCandlestickPattern(MqlRates &rates[], int index, CandlestickPattern &pattern)
{
    if(index < 2)
    {
        pattern.name = "INSUFFICIENT_DATA";
        pattern.signal = "NEUTRAL";
        pattern.confidence = 0.0;
        return;
    }

    double open = rates[index].open;
    double close = rates[index].close;
    double high = rates[index].high;
    double low = rates[index].low;
    double body = MathAbs(close - open);
    double range = high - low;
    double upperShadow = high - MathMax(open, close);
    double lowerShadow = MathMin(open, close) - low;

    pattern.confidence = 0.0;

    // Doji pattern
    if(body < range * 0.1)
    {
        pattern.name = "DOJI";
        pattern.signal = "NEUTRAL";
        pattern.confidence = 80.0;
        return;
    }

    // Hammer pattern
    if(lowerShadow > body * 2 && upperShadow < body * 0.2)
    {
        pattern.name = "HAMMER";
        pattern.signal = "BULLISH";
        pattern.confidence = 75.0;
        return;
    }

    // Shooting star pattern
    if(upperShadow > body * 2 && lowerShadow < body * 0.2)
    {
        pattern.name = "SHOOTING_STAR";
        pattern.signal = "BEARISH";
        pattern.confidence = 75.0;
        return;
    }

    // Bullish engulfing
    if(index >= 1 &&
       rates[index-1].close < rates[index-1].open && // Previous bearish
       close > open && // Current bullish
       close > rates[index-1].open && // Engulfs previous open
       open < rates[index-1].close) // Engulfs previous close
    {
        pattern.name = "BULLISH_ENGULFING";
        pattern.signal = "BULLISH";
        pattern.confidence = 85.0;
        return;
    }

    // Bearish engulfing
    if(index >= 1 &&
       rates[index-1].close > rates[index-1].open && // Previous bullish
       close < open && // Current bearish
       close < rates[index-1].open && // Engulfs previous open
       open > rates[index-1].close) // Engulfs previous close
    {
        pattern.name = "BEARISH_ENGULFING";
        pattern.signal = "BEARISH";
        pattern.confidence = 85.0;
        return;
    }

    // Default pattern
    pattern.name = "STANDARD_CANDLE";
    pattern.signal = close > open ? "BULLISH" : "BEARISH";
    pattern.confidence = 50.0;
}

//+------------------------------------------------------------------+
//| Generate trade examples for RAG training                           |
//+------------------------------------------------------------------+
void GenerateTradeExamples(int fileHandle, string symbol, datetime analysisDate)
{
    // Get H1 data for trade simulation
    MqlRates rates[];
    ArraySetAsSeries(rates, true);
    int copied = CopyRates(symbol, PERIOD_H1, analysisDate + (24 * 3600), 100, rates);

    if(copied < 50)
    {
        FileWrite(fileHandle, "Insufficient data for trade example generation");
        return;
    }

    // Write header
    FileWrite(fileHandle, "Trade_ID,Timestamp,Entry_Price,Stop_Loss,TP1,TP2,TP3,DCA1,DCA2,DCA3,Direction,Setup_Type,Outcome,Profit_Points,Win,Confidence,Reasoning");

    int tradeCount = 0;

    // Generate trade examples based on patterns
    for(int i = 10; i < copied - 20; i++)
    {
        TechnicalIndicators currentInd;
        CalculateTechnicalIndicators(rates, i, currentInd);

        // Look for potential setups
        if(currentInd.trend == "BULLISH" && currentInd.rsi < 70)
        {
            // Bullish setup
            TradeExample trade;
            trade.id = "TRADE_" + IntegerToString(++tradeCount);
            trade.timestamp = rates[i].time;
            trade.symbol = symbol;
            trade.entry_price = rates[i].close;
            trade.stop_loss = rates[i].close - (2 * currentInd.atr);
            trade.take_profit1 = rates[i].close + currentInd.atr;
            trade.take_profit2 = rates[i].close + (2 * currentInd.atr);
            trade.take_profit3 = rates[i].close + (3 * currentInd.atr);
            trade.dca1 = rates[i].close - (0.5 * currentInd.atr);
            trade.dca2 = rates[i].close - currentInd.atr;
            trade.dca3 = rates[i].close - (1.5 * currentInd.atr);
            trade.direction = "LONG";
            trade.setup_type = "BULLISH_TREND_CONTINUATION";
            trade.confidence = 75.0;
            trade.reasoning = "Price above EMAs, RSI not overbought, bullish trend confirmed";

            // Simulate outcome
            SimulateTradeOutcome(rates, i, trade);

            // Write trade example
            WriteTradeExample(fileHandle, trade);
        }

        if(currentInd.trend == "BEARISH" && currentInd.rsi > 30)
        {
            // Bearish setup
            TradeExample trade;
            trade.id = "TRADE_" + IntegerToString(++tradeCount);
            trade.timestamp = rates[i].time;
            trade.symbol = symbol;
            trade.entry_price = rates[i].close;
            trade.stop_loss = rates[i].close + (2 * currentInd.atr);
            trade.take_profit1 = rates[i].close - currentInd.atr;
            trade.take_profit2 = rates[i].close - (2 * currentInd.atr);
            trade.take_profit3 = rates[i].close - (3 * currentInd.atr);
            trade.dca1 = rates[i].close + (0.5 * currentInd.atr);
            trade.dca2 = rates[i].close + currentInd.atr;
            trade.dca3 = rates[i].close + (1.5 * currentInd.atr);
            trade.direction = "SHORT";
            trade.setup_type = "BEARISH_TREND_CONTINUATION";
            trade.confidence = 75.0;
            trade.reasoning = "Price below EMAs, RSI not oversold, bearish trend confirmed";

            // Simulate outcome
            SimulateTradeOutcome(rates, i, trade);

            // Write trade example
            WriteTradeExample(fileHandle, trade);
        }
    }

    FileWrite(fileHandle, "");
    FileWrite(fileHandle, "Generated ", tradeCount, " trade examples for RAG training");
}

//+------------------------------------------------------------------+
//| Simulate trade outcome                                            |
//+------------------------------------------------------------------+
void SimulateTradeOutcome(MqlRates &rates[], int startIndex, TradeExample &trade)
{
    double maxProfit = 0;
    double maxLoss = 0;
    bool hitTP = false;
    bool hitSL = false;

    // Simulate trade over next 20 candles
    for(int i = startIndex + 1; i < MathMin(startIndex + 20, (int)ArraySize(rates)); i++)
    {
        double currentHigh = rates[i].high;
        double currentLow = rates[i].low;

        if(trade.direction == "LONG")
        {
            maxProfit = MathMax(maxProfit, currentHigh - trade.entry_price);
            maxLoss = MathMax(maxLoss, trade.entry_price - currentLow);

            if(currentHigh >= trade.take_profit1)
            {
                trade.outcome = "TP1_HIT";
                trade.profit_points = trade.take_profit1 - trade.entry_price;
                trade.win = true;
                hitTP = true;
                break;
            }
            else if(currentLow <= trade.stop_loss)
            {
                trade.outcome = "STOP_LOSS";
                trade.profit_points = trade.stop_loss - trade.entry_price;
                trade.win = false;
                hitSL = true;
                break;
            }
        }
        else // SHORT
        {
            maxProfit = MathMax(maxProfit, trade.entry_price - currentLow);
            maxLoss = MathMax(maxLoss, currentHigh - trade.entry_price);

            if(currentLow <= trade.take_profit1)
            {
                trade.outcome = "TP1_HIT";
                trade.profit_points = trade.entry_price - trade.take_profit1;
                trade.win = true;
                hitTP = true;
                break;
            }
            else if(currentHigh >= trade.stop_loss)
            {
                trade.outcome = "STOP_LOSS";
                trade.profit_points = trade.entry_price - trade.stop_loss;
                trade.win = false;
                hitSL = true;
                break;
            }
        }
    }

    // If neither TP nor SL was hit within the simulation period
    if(!hitTP && !hitSL)
    {
        double finalPrice = rates[MathMin(startIndex + 19, (int)ArraySize(rates) - 1)].close;

        if(trade.direction == "LONG")
        {
            trade.profit_points = finalPrice - trade.entry_price;
        }
        else
        {
            trade.profit_points = trade.entry_price - finalPrice;
        }

        trade.outcome = trade.profit_points > 0 ? "PARTIAL_PROFIT" : "PARTIAL_LOSS";
        trade.win = trade.profit_points > 0;
    }

    // Convert to points
    trade.profit_points = trade.profit_points / _Point;
}

//+------------------------------------------------------------------+
//| Write trade example to CSV                                        |
//+------------------------------------------------------------------+
void WriteTradeExample(int fileHandle, TradeExample &trade)
{
    string line = StringFormat("%s,%s,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%s,%s,%s,%.2f,%s,%.1f,%s",
                               trade.id,
                               TimeToString(trade.timestamp, TIME_DATE|TIME_MINUTES),
                               trade.entry_price,
                               trade.stop_loss,
                               trade.take_profit1,
                               trade.take_profit2,
                               trade.take_profit3,
                               trade.dca1,
                               trade.dca2,
                               trade.dca3,
                               trade.direction,
                               trade.setup_type,
                               trade.outcome,
                               trade.profit_points,
                               trade.win ? "TRUE" : "FALSE",
                               trade.confidence,
                               trade.reasoning);

    FileWrite(fileHandle, line);
}

//+------------------------------------------------------------------+
//| Calculate Support and Resistance levels                           |
//+------------------------------------------------------------------+
void CalculateSupportResistance(string symbol, ENUM_TIMEFRAMES timeframe, SRLevel &supports[], SRLevel &resistances[])
{
    MqlRates rates[];
    ArraySetAsSeries(rates, true);

    int copied = CopyRates(symbol, timeframe, 0, SR_LookbackBars, rates);

    if(copied <= 0)
    {
        Print("Failed to copy data for S/R calculation. Error: ", GetLastError());
        return;
    }

    double tolerance = SR_TouchTolerance * _Point;
    SRLevel tempLevels[];
    ArrayResize(tempLevels, 0);

    // Find swing highs and lows
    for(int i = 2; i < copied - 2; i++)
    {
        // Check for swing high (resistance)
        if(rates[i].high >= rates[i-1].high &&
           rates[i].high >= rates[i-2].high &&
           rates[i].high >= rates[i+1].high &&
           rates[i].high >= rates[i+2].high)
        {
            AddOrUpdateLevel(tempLevels, rates[i].high, "Resistance", tolerance);
        }

        // Check for swing low (support)
        if(rates[i].low <= rates[i-1].low &&
           rates[i].low <= rates[i-2].low &&
           rates[i].low <= rates[i+1].low &&
           rates[i].low <= rates[i+2].low)
        {
            AddOrUpdateLevel(tempLevels, rates[i].low, "Support", tolerance);
        }
    }

    // Separate into support and resistance, filter by minimum touches
    ArrayResize(supports, 0);
    ArrayResize(resistances, 0);

    for(int i = 0; i < ArraySize(tempLevels); i++)
    {
        if(tempLevels[i].touches >= SR_MinTouches)
        {
            if(tempLevels[i].type == "Support")
            {
                int size = ArraySize(supports);
                ArrayResize(supports, size + 1);
                supports[size] = tempLevels[i];
            }
            else
            {
                int size = ArraySize(resistances);
                ArrayResize(resistances, size + 1);
                resistances[size] = tempLevels[i];
            }
        }
    }

    // Sort by touches and limit to max levels
    SortLevelsByTouches(supports);
    SortLevelsByTouches(resistances);

    if(ArraySize(supports) > SR_MaxLevels)
        ArrayResize(supports, SR_MaxLevels);

    if(ArraySize(resistances) > SR_MaxLevels)
        ArrayResize(resistances, SR_MaxLevels);
}

//+------------------------------------------------------------------+
//| Add or update a level in the array                                |
//+------------------------------------------------------------------+
void AddOrUpdateLevel(SRLevel &levels[], double price, string type, double tolerance)
{
    for(int i = 0; i < ArraySize(levels); i++)
    {
        if(MathAbs(levels[i].price - price) <= tolerance && levels[i].type == type)
        {
            levels[i].touches++;
            levels[i].price = (levels[i].price * (levels[i].touches - 1) + price) / levels[i].touches;
            return;
        }
    }

    int size = ArraySize(levels);
    ArrayResize(levels, size + 1);
    levels[size].price = price;
    levels[size].touches = 1;
    levels[size].type = type;
}

//+------------------------------------------------------------------+
//| Sort levels by number of touches                                 |
//+------------------------------------------------------------------+
void SortLevelsByTouches(SRLevel &levels[])
{
    int size = ArraySize(levels);

    for(int i = 0; i < size - 1; i++)
    {
        for(int j = 0; j < size - i - 1; j++)
        {
            if(levels[j].touches < levels[j + 1].touches)
            {
                SRLevel temp = levels[j];
                levels[j] = levels[j + 1];
                levels[j + 1] = temp;
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Export all symbols with RAG enhancement                           |
//+------------------------------------------------------------------+
void ExportAllSymbolsRAG()
{
    for(int i = 0; i < g_symbolCount; i++)
    {
        Print("Exporting enhanced RAG data for symbol: ", g_symbolList[i]);
        ExportRAGSymbolData(g_symbolList[i]);
    }
}

//+------------------------------------------------------------------+
//| Export enhanced RAG data for specific symbol                      |
//+------------------------------------------------------------------+
void ExportRAGSymbolData(string symbol)
{
    string filename = "RAG_" + symbol + "_" + TimeToString(TimeCurrent(), TIME_DATE) + ".csv";

    if(ExportPath != "")
    {
        if(!MQLInfoInteger(MQL_TESTER))
            filename = ExportPath + "\\" + filename;
    }

    int fileHandle = FileOpen(filename, FILE_WRITE|FILE_ANSI|FILE_TXT);

    if(fileHandle == INVALID_HANDLE)
    {
        Print("Failed to create RAG file: ", filename);
        return;
    }

    Print("Exporting enhanced RAG data to: ", filename);

    // Write header
    FileWrite(fileHandle, "Enhanced RAG Data Export for " + symbol);
    FileWrite(fileHandle, "Generated: " + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS));
    FileWrite(fileHandle, "");

    // Export multi-timeframe RAG data
    ExportRAGTimeframe(fileHandle, symbol, PERIOD_M5, M5_Candles, TimeCurrent(), "M5");
    ExportRAGTimeframe(fileHandle, symbol, PERIOD_M15, M15_Candles, TimeCurrent(), "M15");
    ExportRAGTimeframe(fileHandle, symbol, PERIOD_H1, H1_Candles, TimeCurrent(), "H1");
    ExportRAGTimeframe(fileHandle, symbol, PERIOD_H4, H4_Candles, TimeCurrent(), "H4");

    FileClose(fileHandle);

    // Upload to server if enabled
    if(AutoUpload)
    {
        UploadFileToServer(filename);
    }
}

//+------------------------------------------------------------------+
//| Upload file to server                                             |
//+------------------------------------------------------------------+
void UploadFileToServer(string filename)
{
    Print("Uploading file to server: ", UploadURL);

    int fileHandle = FileOpen(filename, FILE_READ|FILE_ANSI|FILE_TXT);

    if(fileHandle == INVALID_HANDLE)
    {
        Print("Failed to open file for upload: ", filename);
        return;
    }

    string fileContent = "";
    while(!FileIsEnding(fileHandle))
    {
        fileContent += FileReadString(fileHandle) + "\n";
    }
    FileClose(fileHandle);

    Print("File size: ", StringLen(fileContent), " bytes");

    string boundary = "----WebKitFormBoundary" + IntegerToString(GetTickCount());
    string contentType = "multipart/form-data; boundary=" + boundary;

    string postData = "";
    postData += "--" + boundary + "\r\n";
    postData += "Content-Disposition: form-data; name=\"file\"; filename=\"" + filename + "\"\r\n";
    postData += "Content-Type: text/csv\r\n\r\n";
    postData += fileContent + "\r\n";
    postData += "--" + boundary + "--\r\n";

    char post[];
    char result[];
    string headers;

    StringToCharArray(postData, post, 0, StringLen(postData));
    ArrayResize(post, ArraySize(post) - 1);

    string requestHeaders = "Content-Type: " + contentType + "\r\n";

    ResetLastError();
    int res = WebRequest(
        "POST",
        UploadURL,
        requestHeaders,
        5000,
        post,
        result,
        headers
    );

    if(res == -1)
    {
        int error = GetLastError();
        Print("Upload failed! Error: ", error);
        return;
    }

    string resultString = CharArrayToString(result);
    Print("Upload response code: ", res);

    if(res == 200)
    {
        Print("File uploaded successfully!");
    }
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("RAG-Enhanced Multi-Timeframe Data Exporter Stopped");
}