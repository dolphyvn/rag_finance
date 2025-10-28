import json
import os
import sys
from datetime import datetime
import logging

# Add parent directory to path to import trading_rag
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_rag import TradingRAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchImporter:
    def __init__(self, rag_system=None):
        """
        Initialize batch importer

        Args:
            rag_system: TradingRAG instance (optional, will create if not provided)
        """
        self.rag_system = rag_system or TradingRAG()
        self.import_stats = {
            'total_processed': 0,
            'successful_imports': 0,
            'failed_imports': 0,
            'skipped_duplicates': 0,
            'errors': []
        }

    def import_rag_knowledge(self, json_file, batch_size=100, skip_duplicates=True):
        """
        Batch import historical trade setups into RAG system

        Args:
            json_file: JSON file with RAG training examples
            batch_size: Number of trades to process in each batch
            skip_duplicates: Whether to skip duplicate trades
        """
        logger.info(f"Starting batch import from {json_file}")

        if not os.path.exists(json_file):
            logger.error(f"File not found: {json_file}")
            return False

        try:
            # Load trade data
            with open(json_file, 'r') as f:
                trades = json.load(f)

            logger.info(f"Loaded {len(trades)} trades from {json_file}")

            # Get existing trade IDs to check for duplicates
            existing_ids = set()
            if skip_duplicates:
                try:
                    existing_data = self.rag_system.collection.get()
                    existing_ids = set(existing_data['ids'])
                    logger.info(f"Found {len(existing_ids)} existing trades in knowledge base")
                except Exception as e:
                    logger.warning(f"Could not check for duplicates: {e}")

            # Process trades in batches
            total_batches = (len(trades) + batch_size - 1) // batch_size

            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(trades))
                batch_trades = trades[start_idx:end_idx]

                logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_trades)} trades)")

                batch_success = 0
                batch_failed = 0
                batch_skipped = 0

                for trade in batch_trades:
                    self.import_stats['total_processed'] += 1

                    # Check for duplicate
                    if skip_duplicates and trade['id'] in existing_ids:
                        batch_skipped += 1
                        self.import_stats['skipped_duplicates'] += 1
                        continue

                    # Validate trade data
                    if not self._validate_trade_data(trade):
                        batch_failed += 1
                        self.import_stats['failed_imports'] += 1
                        continue

                    # Add to RAG system
                    if self.rag_system.add_trading_knowledge(trade):
                        batch_success += 1
                        self.import_stats['successful_imports'] += 1
                        existing_ids.add(trade['id'])  # Add to existing IDs to avoid duplicates in this batch
                    else:
                        batch_failed += 1
                        self.import_stats['failed_imports'] += 1

                logger.info(f"Batch {batch_num + 1} results: {batch_success} successful, {batch_failed} failed, {batch_skipped} skipped")

                # Small delay to prevent overwhelming the system
                if batch_num < total_batches - 1:
                    import time
                    time.sleep(0.1)

            logger.info(f"Batch import completed: {self.import_stats['successful_imports']} successful, "
                       f"{self.import_stats['failed_imports']} failed, {self.import_stats['skipped_duplicates']} skipped")

            return True

        except Exception as e:
            logger.error(f"Error during batch import: {e}")
            self.import_stats['errors'].append(str(e))
            return False

    def _validate_trade_data(self, trade):
        """Validate trade data structure"""
        required_fields = [
            'id', 'timestamp', 'symbol', 'price', 'setup_type',
            'direction', 'confidence', 'context', 'entry_reason',
            'entry', 'dca_levels', 'stop_loss', 'take_profit', 'tags'
        ]

        for field in required_fields:
            if field not in trade:
                logger.warning(f"Trade {trade.get('id', 'unknown')} missing required field: {field}")
                return False

        # Validate data types and ranges
        try:
            # Validate numeric fields
            numeric_fields = ['price', 'entry', 'stop_loss']
            for field in numeric_fields:
                if not isinstance(trade[field], (int, float)) or trade[field] <= 0:
                    logger.warning(f"Trade {trade['id']} has invalid {field}: {trade[field]}")
                    return False

            # Validate lists
            list_fields = ['dca_levels', 'take_profit', 'tags']
            for field in list_fields:
                if not isinstance(trade[field], list) or len(trade[field]) == 0:
                    logger.warning(f"Trade {trade['id']} has invalid {field}: {trade[field]}")
                    return False

            # Validate confidence
            if not (0 <= trade['confidence'] <= 1):
                logger.warning(f"Trade {trade['id']} has invalid confidence: {trade['confidence']}")
                return False

            # Validate direction
            if trade['direction'] not in ['LONG', 'SHORT', 'NEUTRAL']:
                logger.warning(f"Trade {trade['id']} has invalid direction: {trade['direction']}")
                return False

        except Exception as e:
            logger.warning(f"Trade {trade.get('id', 'unknown')} validation error: {e}")
            return False

        return True

    def import_multiple_files(self, file_pattern="data/*.json", batch_size=100):
        """
        Import multiple JSON files matching a pattern

        Args:
            file_pattern: Glob pattern for JSON files
            batch_size: Batch size for processing
        """
        import glob

        files = glob.glob(file_pattern)
        logger.info(f"Found {len(files)} files matching pattern: {file_pattern}")

        if not files:
            logger.warning(f"No files found matching pattern: {file_pattern}")
            return False

        total_success = True
        for file_path in files:
            logger.info(f"Importing from file: {file_path}")
            success = self.import_rag_knowledge(file_path, batch_size=batch_size)
            if not success:
                total_success = False
                logger.error(f"Failed to import from: {file_path}")

        return total_success

    def get_import_summary(self):
        """Get summary of import statistics"""
        return {
            'import_summary': {
                'total_processed': self.import_stats['total_processed'],
                'successful_imports': self.import_stats['successful_imports'],
                'failed_imports': self.import_stats['failed_imports'],
                'skipped_duplicates': self.import_stats['skipped_duplicates'],
                'success_rate': (self.import_stats['successful_imports'] / max(1, self.import_stats['total_processed'])) * 100,
                'errors': self.import_stats['errors']
            }
        }

    def reset_stats(self):
        """Reset import statistics"""
        self.import_stats = {
            'total_processed': 0,
            'successful_imports': 0,
            'failed_imports': 0,
            'skipped_duplicates': 0,
            'errors': []
        }

def main():
    """Main function"""
    print("=== Batch Import Tool for Trading RAG System ===")

    # Initialize batch importer
    print("Initializing RAG system...")
    importer = BatchImporter()

    # Check for data files
    data_dir = 'data'
    potential_files = [
        os.path.join(data_dir, 'xauusd_rag_verified.json'),
        os.path.join(data_dir, 'xauusd_rag_samples.json'),
    ]

    files_to_import = []
    for file_path in potential_files:
        if os.path.exists(file_path):
            files_to_import.append(file_path)
            print(f"Found file: {file_path}")

    if not files_to_import:
        print("No RAG data files found. Please run rag_converter.py first.")
        return

    # Import each file
    for file_path in files_to_import:
        print(f"\n--- Importing {file_path} ---")

        # Reset stats for this file
        importer.reset_stats()

        # Import the file
        success = importer.import_rag_knowledge(
            json_file=file_path,
            batch_size=50,  # Smaller batches for better error handling
            skip_duplicates=True
        )

        # Show results
        summary = importer.get_import_summary()
        print(f"\nImport Results for {os.path.basename(file_path)}:")
        print(f"  Total processed: {summary['import_summary']['total_processed']}")
        print(f"  Successful: {summary['import_summary']['successful_imports']}")
        print(f"  Failed: {summary['import_summary']['failed_imports']}")
        print(f"  Skipped duplicates: {summary['import_summary']['skipped_duplicates']}")
        print(f"  Success rate: {summary['import_summary']['success_rate']:.1f}%")

        if summary['import_summary']['errors']:
            print(f"  Errors: {len(summary['import_summary']['errors'])}")
            for error in summary['import_summary']['errors'][:5]:  # Show first 5 errors
                print(f"    - {error}")

    # Get final knowledge base stats
    print(f"\n--- Final Knowledge Base Stats ---")
    stats = importer.rag_system.get_knowledge_base_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\n=== Import Complete ===")

if __name__ == "__main__":
    main()