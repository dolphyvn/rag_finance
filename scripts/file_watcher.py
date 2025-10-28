import time
import os
import logging
from datetime import datetime
from pathlib import Path
import threading
import json

from ea_csv_processor import EA_CSV_Processor
from rag_converter import RAGDataConverter
from batch_import import BatchImporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/file_watcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FileWatcher:
    def __init__(self, watch_folder, check_interval=30):
        """
        Initialize file watcher service

        Args:
            watch_folder: Folder to watch for new files
            check_interval: Check interval in seconds
        """
        self.watch_folder = watch_folder
        self.check_interval = check_interval
        self.running = False
        self.processed_files = set()
        self.processor = EA_CSV_Processor()
        self.rag_converter = RAGDataConverter()
        self.batch_importer = BatchImporter()

        # Load processed files history
        self.history_file = "data/watcher_history.json"
        self._load_history()

        # Create logs directory
        os.makedirs("logs", exist_ok=True)

    def _load_history(self):
        """Load processed files history"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('processed_files', []))
                logger.info(f"Loaded {len(self.processed_files)} files from history")
        except Exception as e:
            logger.warning(f"Could not load history file: {e}")
            self.processed_files = set()

    def _save_history(self):
        """Save processed files history"""
        try:
            data = {
                'processed_files': list(self.processed_files),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save history file: {e}")

    def _scan_for_files(self):
        """Scan watch folder for new files"""
        if not os.path.exists(self.watch_folder):
            logger.warning(f"Watch folder does not exist: {self.watch_folder}")
            return []

        new_files = []
        try:
            for filename in os.listdir(self.watch_folder):
                file_path = os.path.join(self.watch_folder, filename)

                if not os.path.isfile(file_path):
                    continue

                # Check if file is complete (not being written)
                try:
                    file_size = os.path.getsize(file_path)
                    time.sleep(1)  # Wait 1 second
                    new_size = os.path.getsize(file_path)

                    if file_size != new_size:
                        continue  # File is still being written
                except:
                    continue

                # Check if already processed
                file_key = f"{filename}_{file_size}"
                if file_key in self.processed_files:
                    continue

                new_files.append({
                    'filename': filename,
                    'path': file_path,
                    'size': file_size,
                    'modified': os.path.getmtime(file_path)
                })

        except Exception as e:
            logger.error(f"Error scanning folder: {e}")

        return new_files

    def _process_file(self, file_info):
        """Process a single file"""
        filename = file_info['filename']
        file_path = file_info['path']
        file_key = f"{filename}_{file_info['size']}"

        try:
            logger.info(f"Processing file: {filename}")

            # Determine file type and process accordingly
            if filename.endswith('.csv'):
                success = self._process_csv_file(file_path, filename)
            elif filename.endswith('.json') and 'current_market' in filename:
                success = self._process_json_file(file_path, filename)
            else:
                logger.info(f"Skipping unsupported file: {filename}")
                return False

            if success:
                self.processed_files.add(file_key)
                self._save_history()
                logger.info(f"Successfully processed: {filename}")
                return True
            else:
                logger.warning(f"Failed to process: {filename}")
                return False

        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            return False

    def _process_csv_file(self, file_path, filename):
        """Process CSV file"""
        try:
            # Copy to local raw folder
            raw_path = os.path.join("data", "raw_ea", filename)
            os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            import shutil
            shutil.copy2(file_path, raw_path)

            # Process CSV based on type
            if 'enhanced' in filename:
                processed_path = self.processor.process_enhanced_csv(raw_path)
            else:
                processed_path = self.processor.process_basic_csv(raw_path)

            if processed_path and 'enhanced' in filename:
                # Convert to RAG format and import
                return self._convert_and_import_to_rag(processed_path)

            return processed_path is not None

        except Exception as e:
            logger.error(f"Error processing CSV file {filename}: {e}")
            return False

    def _process_json_file(self, file_path, filename):
        """Process JSON file"""
        try:
            processed_path = self.processor.process_current_market_json(file_path)
            return processed_path is not None
        except Exception as e:
            logger.error(f"Error processing JSON file {filename}: {e}")
            return False

    def _convert_and_import_to_rag(self, csv_path):
        """Convert processed CSV to RAG format and import"""
        try:
            logger.info("Converting to RAG format...")

            # Convert to RAG format
            rag_output = csv_path.replace('.csv', '_rag.json')
            rag_output = rag_output.replace('processed_', 'rag_')

            rag_examples = self.rag_converter.convert_csv_to_rag(
                csv_path,
                rag_output,
                min_profit_points=5.0
            )

            if rag_examples:
                logger.info(f"Created {len(rag_examples)} RAG examples")

                # Import to knowledge base
                logger.info("Importing to knowledge base...")
                success = self.batch_import.import_rag_knowledge(
                    rag_output,
                    batch_size=50,
                    skip_duplicates=True
                )

                if success:
                    logger.info("Successfully imported to knowledge base")
                    return True
                else:
                    logger.error("Failed to import to knowledge base")
                    return False

            return False

        except Exception as e:
            logger.error(f"Error converting/importing to RAG: {e}")
            return False

    def start_watching(self):
        """Start the file watcher service"""
        logger.info("Starting file watcher service...")
        self.running = True

        logger.info(f"Watching folder: {self.watch_folder}")
        logger.info(f"Check interval: {self.check_interval} seconds")

        while self.running:
            try:
                # Scan for new files
                new_files = self._scan_for_files()

                if new_files:
                    logger.info(f"Found {len(new_files)} new files")

                    # Process each file
                    for file_info in new_files:
                        if not self.running:  # Check if stopped during processing
                            break
                        self._process_file(file_info)

                # Wait before next check
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping...")
                break
            except Exception as e:
                logger.error(f"Error in watcher loop: {e}")
                time.sleep(self.check_interval)

        self.running = False
        logger.info("File watcher service stopped")

    def stop_watching(self):
        """Stop the file watcher service"""
        logger.info("Stopping file watcher service...")
        self.running = False

    def get_status(self):
        """Get watcher status"""
        return {
            'running': self.running,
            'watch_folder': self.watch_folder,
            'check_interval': self.check_interval,
            'processed_files_count': len(self.processed_files),
            'folder_exists': os.path.exists(self.watch_folder)
        }

class FileWatcherService:
    """Service wrapper for file watcher"""
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.watcher = None
        self.watcher_thread = None

    def _load_config(self):
        """Load configuration"""
        default_config = {
            "watch_folder": "C:\\RAG_Data\\",
            "check_interval": 30,
            "auto_convert_to_rag": True,
            "log_level": "INFO"
        }

        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    default_config.update(config)
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")

        return default_config

    def start(self):
        """Start the service"""
        if self.watcher and self.watcher.running:
            logger.warning("Service is already running")
            return

        self.watcher = FileWatcher(
            watch_folder=self.config['watch_folder'],
            check_interval=self.config['check_interval']
        )

        # Start in separate thread
        self.watcher_thread = threading.Thread(target=self.watcher.start_watching)
        self.watcher_thread.daemon = True
        self.watcher_thread.start()

        logger.info("File watcher service started")

    def stop(self):
        """Stop the service"""
        if self.watcher:
            self.watcher.stop_watching()
            if self.watcher_thread:
                self.watcher_thread.join(timeout=10)
            logger.info("File watcher service stopped")

    def get_status(self):
        """Get service status"""
        if self.watcher:
            status = self.watcher.get_status()
            status['config'] = self.config
            return status
        return {'running': False, 'config': self.config}

def main():
    """Main function"""
    print("=== File Watcher Service ===")

    # Load configuration
    config_file = "watcher_config.json"
    service = FileWatcherService(config_file)

    # Create default config if it doesn't exist
    if not os.path.exists(config_file):
        default_config = {
            "watch_folder": "C:\\RAG_Data\\",
            "check_interval": 30,
            "auto_convert_to_rag": True,
            "log_level": "INFO"
        }
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Created default config file: {config_file}")
        print("Please update the watch_folder path and restart.")

    # Start service
    try:
        service.start()
        print("File watcher service started. Press Ctrl+C to stop.")

        # Keep main thread alive
        while True:
            time.sleep(1)
            status = service.get_status()
            if not status['running']:
                break

    except KeyboardInterrupt:
        print("\nStopping service...")
        service.stop()
        print("Service stopped.")

if __name__ == "__main__":
    main()