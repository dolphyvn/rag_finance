import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import logging
import asyncio
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingRAG:
    def __init__(self, model_name="qwen3:8b", vector_db_path="./trading_knowledge"):
        """
        Initialize the Trading RAG system

        Args:
            model_name: LLM model name (for Ollama)
            vector_db_path: Path to store vector database
        """
        # Initialize vector store
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="trading_setups",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")

        # Initialize LLM settings
        self.llm_model = model_name

        # Check if Ollama is available
        try:
            import ollama
            self.ollama = ollama
            logger.info(f"Ollama initialized with model: {model_name}")
        except ImportError:
            logger.error("Ollama not installed. Install with: pip install ollama")
            self.ollama = None

    def add_trading_knowledge(self, trade_data):
        """
        Add historical trade setups to knowledge base

        Args:
            trade_data: Dictionary containing trade information
        """
        try:
            # Create rich text description for embedding
            context_text = f"""
            Trading Setup Analysis:

            Symbol: {trade_data['symbol']}
            Date: {trade_data['timestamp']}
            Price: {trade_data['price']}
            Setup Type: {trade_data['setup_type']}
            Direction: {trade_data['direction']}
            Confidence: {trade_data['confidence']}

            Market Context:
            {trade_data['context']}

            Trade Details:
            Entry Reason: {trade_data['entry_reason']}
            Entry Price: {trade_data['entry']}
            DCA Levels: {', '.join(map(str, trade_data['dca_levels']))}
            Stop Loss: {trade_data['stop_loss']}
            Take Profit: {', '.join(map(str, trade_data['take_profit']))}

            Outcome: {trade_data.get('actual_outcome', 'UNKNOWN')}
            Success: {trade_data.get('success', 'UNKNOWN')}
            """

            # Generate embedding
            embedding = self.embedder.encode(context_text).tolist()

            # Store in vector DB
            self.collection.add(
                embeddings=[embedding],
                documents=[context_text],
                metadatas=[{
                    "symbol": trade_data['symbol'],
                    "price": trade_data['price'],
                    "timestamp": trade_data['timestamp'],
                    "setup_type": trade_data['setup_type'],
                    "direction": trade_data['direction'],
                    "confidence": trade_data['confidence'],
                    "success": trade_data.get('success', False),
                    "profit": trade_data.get('actual_profit', 0),
                    "tags": ','.join(trade_data['tags'])
                }],
                ids=[trade_data['id']]
            )

            logger.debug(f"Added trade {trade_data['id']} to knowledge base")
            return True

        except Exception as e:
            logger.error(f"Error adding trade knowledge: {e}")
            return False

    def retrieve_similar_setups(self, current_market_context, top_k=5, setup_type_filter=None, direction_filter=None):
        """
        Find similar historical trade setups

        Args:
            current_market_context: Dictionary with current market information
            top_k: Number of similar setups to retrieve
            setup_type_filter: Filter by setup type (optional)
            direction_filter: Filter by direction (optional)
        """
        try:
            # Create query from current market
            query_text = f"""
            Current Market Analysis:
            Symbol: {current_market_context['symbol']}
            Price: {current_market_context['price']}
            Session: {current_market_context.get('session', 'UNKNOWN')}

            Technical Indicators:
            RSI: {current_market_context.get('rsi', 'N/A')}
            MACD: {current_market_context.get('macd', 'N/A')}
            Trend: {current_market_context.get('trend', 'N/A')}

            Support: {current_market_context.get('support', 'N/A')}
            Resistance: {current_market_context.get('resistance', 'N/A')}

            Analysis: {current_market_context.get('analysis', '')}
            """

            # Generate query embedding
            query_embedding = self.embedder.encode(query_text).tolist()

            # Build where clause for filtering
            where_clause = {"symbol": current_market_context['symbol']}

            if setup_type_filter:
                where_clause["setup_type"] = setup_type_filter

            if direction_filter:
                where_clause["direction"] = direction_filter

            # Search vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause
            )

            logger.info(f"Retrieved {len(results['documents'][0])} similar setups")
            return results

        except Exception as e:
            logger.error(f"Error retrieving similar setups: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def calculate_technical_indicators(self, price_data):
        """
        Calculate real-time technical indicators

        Args:
            price_data: List of dictionaries with OHLCV data
        """
        try:
            df = pd.DataFrame(price_data)

            # Calculate EMAs
            df['EMA20'] = df['close'].ewm(span=20).mean()
            df['EMA50'] = df['close'].ewm(span=50).mean()
            df['EMA200'] = df['close'].ewm(span=200).mean()

            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Calculate MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']

            # Calculate ATR
            df['TR'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift()),
                    abs(df['low'] - df['close'].shift())
                )
            )
            df['ATR'] = df['TR'].rolling(window=14).mean()

            # Calculate Bollinger Bands
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            df['BB_std'] = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
            df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)

            # Return latest indicators
            latest = df.iloc[-1]
            return {
                'rsi': float(latest['RSI']),
                'macd': float(latest['MACD']),
                'macd_signal': float(latest['MACD_signal']),
                'macd_hist': float(latest['MACD_hist']),
                'ema_20': float(latest['EMA20']),
                'ema_50': float(latest['EMA50']),
                'ema_200': float(latest['EMA200']),
                'atr': float(latest['ATR']),
                'bb_upper': float(latest['BB_upper']),
                'bb_middle': float(latest['BB_middle']),
                'bb_lower': float(latest['BB_lower']),
                'trend': self._determine_trend(latest)
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}

    def _determine_trend(self, latest_data):
        """Determine trend based on EMAs"""
        if latest_data['close'] > latest_data['EMA20'] > latest_data['EMA50'] > latest_data['EMA200']:
            return 'BULLISH'
        elif latest_data['close'] < latest_data['EMA20'] < latest_data['EMA50'] < latest_data['EMA200']:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def find_support_resistance(self, price_data, window=20):
        """
        Identify key support and resistance levels

        Args:
            price_data: List of dictionaries with OHLCV data
            window: Window size for finding S/R levels
        """
        try:
            df = pd.DataFrame(price_data)

            # Find local peaks (resistance)
            resistance_levels = []
            for i in range(window, len(df) - window):
                if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
                    resistance_levels.append(df['high'].iloc[i])

            # Find local troughs (support)
            support_levels = []
            for i in range(window, len(df) - window):
                if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
                    support_levels.append(df['low'].iloc[i])

            # Get unique levels and sort
            resistance = sorted(set(resistance_levels), reverse=True)[:3]
            support = sorted(set(support_levels), reverse=True)[:3]

            return {
                'resistance': resistance,
                'support': support,
                'nearest_resistance': resistance[0] if resistance else None,
                'nearest_support': support[0] if support else None
            }
        except Exception as e:
            logger.error(f"Error finding support/resistance: {e}")
            return {'resistance': [], 'support': [], 'nearest_resistance': None, 'nearest_support': None}

    def generate_trade_analysis(self, current_market, price_history):
        """
        Main RAG pipeline: Retrieve + Generate

        Args:
            current_market: Dictionary with current market data
            price_history: List of price data dictionaries
        """
        try:
            logger.info("Starting trade analysis...")

            # Step 1: Calculate technical indicators
            indicators = self.calculate_technical_indicators(price_history)

            # Step 2: Find support/resistance
            sr_levels = self.find_support_resistance(price_history)

            # Step 3: Create market context
            market_analysis = f"""
            Current Market Analysis for {current_market['symbol']}:

            Price Action:
            - Current Price: {current_market['price']}
            - Session: {current_market.get('session', 'UNKNOWN')}

            Technical Indicators:
            - RSI: {indicators.get('rsi', 'N/A'):.2f}
            - MACD: {indicators.get('macd', 'N/A'):.4f}
            - MACD Signal: {indicators.get('macd_signal', 'N/A'):.4f}
            - EMA20: {indicators.get('ema_20', 'N/A'):.2f}
            - EMA50: {indicators.get('ema_50', 'N/A'):.2f}
            - EMA200: {indicators.get('ema_200', 'N/A'):.2f}
            - ATR: {indicators.get('atr', 'N/A'):.2f}
            - Trend: {indicators.get('trend', 'NEUTRAL')}

            Support/Resistance Levels:
            - Nearest Support: {sr_levels.get('nearest_support', 'N/A')}
            - Nearest Resistance: {sr_levels.get('nearest_resistance', 'N/A')}

            Additional Context:
            {current_market.get('additional_context', '')}
            """

            # Step 4: Retrieve similar historical setups
            similar_setups = self.retrieve_similar_setups({
                'symbol': current_market['symbol'],
                'price': current_market['price'],
                'analysis': market_analysis,
                'rsi': indicators.get('rsi'),
                'macd': indicators.get('macd'),
                'trend': indicators.get('trend'),
                'session': current_market.get('session'),
                'support': sr_levels.get('nearest_support'),
                'resistance': sr_levels.get('nearest_resistance')
            }, top_k=5)

            # Step 5: Build LLM prompt with retrieved context
            prompt = self._build_analysis_prompt(
                market_analysis,
                similar_setups,
                current_market['symbol']
            )

            # Step 6: Generate analysis using LLM
            if self.ollama:
                response = self._call_llm(prompt)
            else:
                response = self._generate_mock_analysis(market_analysis, similar_setups)

            return {
                'analysis': response,
                'indicators': indicators,
                'sr_levels': sr_levels,
                'similar_setups_count': len(similar_setups['documents'][0]) if similar_setups['documents'] else 0,
                'market_analysis': market_analysis
            }

        except Exception as e:
            logger.error(f"Error generating trade analysis: {e}")
            return {
                'analysis': f"Error generating analysis: {str(e)}",
                'indicators': {},
                'sr_levels': {},
                'similar_setups_count': 0,
                'market_analysis': ""
            }

    def _build_analysis_prompt(self, current_analysis, similar_setups, symbol):
        """Construct prompt with retrieved context"""

        # Extract similar setup summaries
        historical_context = ""
        if similar_setups['documents'] and similar_setups['documents'][0]:
            historical_context = "\n### Similar Historical Setups:\n"
            for i, (doc, meta) in enumerate(zip(
                similar_setups['documents'][0],
                similar_setups['metadatas'][0]
            )):
                outcome = "WIN" if meta.get('success') else "LOSS"
                profit = meta.get('profit', 0)
                setup_type = meta.get('setup_type', 'UNKNOWN')
                direction = meta.get('direction', 'UNKNOWN')

                historical_context += f"""
Setup {i+1} ({outcome}, {setup_type} {direction}, Profit: {profit:+.1f} points):
{doc[:300]}...
---
"""

        prompt = f"""You are an expert trading analyst specializing in {symbol} (Gold vs USD).

### Current Market Analysis:
{current_analysis}

{historical_context}

### Your Task:
Based on the current market conditions and similar historical setups, provide a detailed trade analysis including:

1. **Trade Direction** (LONG/SHORT/NEUTRAL) with confidence percentage
2. **Reasoning** - Why this direction? What are the key factors?
3. **Entry Strategy** - Primary and alternative entry points
4. **DCA Levels** - 3-4 levels for dollar-cost averaging if trade goes against you
5. **Stop Loss** - Conservative and aggressive options
6. **Take Profit Targets** - Multiple levels with position sizing recommendations
7. **Risk-Reward Ratio** for each target level
8. **Key Risk Factors** - What could invalidate this setup?
9. **Trade Management** - How to manage the position

### Response Format:
Provide structured, actionable analysis. Be specific with price levels and percentages.
Consider the historical setups' success rates and outcomes in your analysis.

Focus on risk management and provide realistic expectations."""

        return prompt

    def _call_llm(self, prompt):
        """Call LLM via Ollama"""
        try:
            response = self.ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.3,  # Lower for more consistent analysis
                    'top_p': 0.9,
                    'top_k': 40,
                    'num_ctx': 8192  # Context window
                }
            )
            return response['response']
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return f"Error calling LLM: {str(e)}"

    def _generate_mock_analysis(self, market_analysis, similar_setups):
        """Generate mock analysis when Ollama is not available"""
        return f"""
# Trading Analysis for XAUUSD

## Trade Direction: NEUTRAL (Confidence: 60%)

## Reasoning:
Based on current market conditions and historical patterns, the market appears to be in a consolidation phase. The RSI is neutral, and price is trading between key support and resistance levels.

## Entry Strategy:
- Wait for a clear breakout above resistance or below support
- Consider entering on confirmation of trend direction

## Key Levels:
- **Support**: [Calculate from market data]
- **Resistance**: [Calculate from market data]

## Risk Management:
- **Stop Loss**: Place below recent swing low (for LONG) or above swing high (for SHORT)
- **Take Profit**: Target 2:1 risk-reward ratio

## Note:
This is a mock analysis. Install Ollama and pull qwen3:8b model for real AI-powered analysis:
```bash
pip install ollama
ollama pull qwen3:8b
```

## Similar Historical Setups Found: {len(similar_setups['documents'][0]) if similar_setups['documents'] else 0}

---

{market_analysis}
"""

    def analyze_current_market(self, mt5_client=None):
        """
        Analyze current market using remote MT5 data

        Args:
            mt5_client: MT5RemoteClient instance (optional)
        """
        try:
            if not mt5_client:
                logger.warning("No MT5 client provided. Using sample data.")
                return self._generate_sample_analysis()

            # Get current market data
            current_data = asyncio.run(mt5_client.get_current_market_data("XAUUSD", "temp_market.json"))

            if not current_data:
                logger.error("Failed to get current market data")
                return None

            # Convert to price history format
            price_history = self._convert_multi_timeframe_to_history(current_data)

            # Create current market context
            m5_data = current_data.get('multi_timeframe', {}).get('M5', {})
            current_market = {
                'symbol': current_data['symbol'],
                'price': current_data.get('current_price', m5_data.get('close')),
                'session': self._get_current_session(),
                'additional_context': f"Real-time analysis from MT5 server"
            }

            # Generate analysis
            return self.generate_trade_analysis(current_market, price_history)

        except Exception as e:
            logger.error(f"Error analyzing current market: {e}")
            return None

    def _convert_multi_timeframe_to_history(self, current_data):
        """Convert current multi-timeframe data to price history format"""
        # This is a simplified conversion - in practice, you'd want more historical data
        m5_data = current_data.get('multi_timeframe', {}).get('M5', {})

        # Create synthetic price history around current price
        current_price = m5_data.get('close', 2700)
        atr = m5_data.get('atr_14', 5)

        price_history = []
        for i in range(100):  # Create 100 synthetic candles
            offset = (i - 50) * atr * 0.1
            price = current_price + offset + np.random.normal(0, atr * 0.05)

            price_history.append({
                'open': price - np.random.uniform(-1, 1),
                'high': price + abs(np.random.normal(0, atr * 0.3)),
                'low': price - abs(np.random.normal(0, atr * 0.3)),
                'close': price,
                'volume': int(np.random.uniform(1000, 5000))
            })

        return price_history

    def _get_current_session(self):
        """Determine current trading session"""
        hour = datetime.now().hour
        if 0 <= hour < 8:
            return 'ASIAN_SESSION'
        elif 8 <= hour < 13:
            return 'LONDON_SESSION'
        elif 13 <= hour < 20:
            return 'US_SESSION'
        else:
            return 'AFTER_HOURS'

    def _generate_sample_analysis(self):
        """Generate a sample analysis for demonstration"""
        logger.info("Generating sample analysis...")

        # Create sample market data
        current_market = {
            'symbol': 'XAUUSD',
            'price': 2693.77,
            'session': 'US_SESSION',
            'additional_context': 'Sample analysis for demonstration'
        }

        # Create sample price history
        price_history = []
        base_price = 2693.77
        for i in range(100):
            price = base_price + np.sin(i * 0.1) * 10 + np.random.normal(0, 2)
            price_history.append({
                'open': price - 0.5,
                'high': price + abs(np.random.normal(0, 1)),
                'low': price - abs(np.random.normal(0, 1)),
                'close': price,
                'volume': int(np.random.uniform(1000, 5000))
            })

        return self.generate_trade_analysis(current_market, price_history)

    def get_knowledge_base_stats(self):
        """Get statistics about the knowledge base"""
        try:
            count = self.collection.count()
            return {
                'total_trades': count,
                'vector_db_path': "./trading_knowledge",
                'model': self.llm_model,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
            }
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {}

def main():
    """Main function for testing"""
    print("=== Trading RAG System ===")
    print("Initializing RAG system...")

    # Initialize RAG system
    trading_rag = TradingRAG()

    # Show knowledge base stats
    stats = trading_rag.get_knowledge_base_stats()
    print(f"Knowledge Base Stats: {stats}")

    # Generate sample analysis
    print("\nGenerating sample analysis...")
    result = trading_rag._generate_sample_analysis()

    if result:
        print("\n=== Trading Analysis ===")
        print(result['analysis'])
        print(f"\nSimilar setups found: {result['similar_setups_count']}")
    else:
        print("Failed to generate analysis")

if __name__ == "__main__":
    main()