"""
Simple logging test without PyTorch dependencies.
"""

import os
import sys
import tempfile
import logging
import time
from pathlib import Path

def create_simple_logger(log_dir, experiment_name):
    """Create a simple logger similar to our Logger class."""
    from datetime import datetime
    
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(f"test.{experiment_name}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file, file_handler, console_handler


def test_simple_logging():
    """Test the logging functionality."""
    print("=== Simple Logging Test ===")
    
    # Create temporary directory
    test_dir = Path(tempfile.mkdtemp(prefix="simple_logging_test_"))
    print(f"Test directory: {test_dir}")
    
    try:
        # Create logger
        logger, log_file, file_handler, console_handler = create_simple_logger(
            test_dir / "logs", "simple_test"
        )
        
        print(f"Log file: {log_file}")
        
        # Log some messages
        logger.info("Starting test...")
        logger.info("Step 0 - test_loss: 1.500000, test_accuracy: 0.800000")
        logger.info("Step 1 - test_loss: 1.200000, test_accuracy: 0.850000")
        logger.info("Test completed successfully!")
        
        # Flush handlers
        file_handler.flush()
        console_handler.flush()
        
        # Wait a moment to ensure file is written
        time.sleep(0.1)
        
        # Read log file
        if log_file.exists():
            with open(log_file, 'r') as f:
                content = f.read()
            
            print(f"\nLog file size: {len(content)} chars")
            print(f"Contains 'test_loss': {'test_loss' in content}")
            print(f"Contains 'test_accuracy': {'test_accuracy' in content}")
            
            if content:
                print(f"\nLog file content:\n{content}")
                
                if "test_loss" in content and "test_accuracy" in content:
                    print("\n✅ SUCCESS: Log file contains expected content!")
                    return True
                else:
                    print("\n❌ FAILURE: Log file missing expected content!")
                    return False
            else:
                print("\n❌ FAILURE: Log file is empty!")
                return False
        else:
            print(f"\n❌ FAILURE: Log file doesn't exist: {log_file}")
            return False
    
    except Exception as e:
        print(f"\n❌ FAILURE: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(test_dir)
            print(f"\nCleaned up: {test_dir}")
        except:
            print(f"\nCould not clean up: {test_dir}")


if __name__ == "__main__":
    success = test_simple_logging()
    print(f"\nFinal result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
    sys.exit(0 if success else 1) 