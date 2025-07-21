"""
Debug test to isolate logging issues.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logging_utils import Logger


def test_logging_functionality():
    """Test logging functionality in isolation."""
    print("=== Debug Logging Test ===")
    
    # Create temporary directory
    test_dir = Path(tempfile.mkdtemp(prefix="logging_debug_test_"))
    print(f"Test directory: {test_dir}")
    
    try:
        # Simple test configuration
        test_config = {
            "experiment_name": "logging_debug_test",
            "logging": {
                "use_wandb": False,
                "use_tensorboard": False,
                "log_dir": str(test_dir / "logs"),
                "save_dir": str(test_dir / "checkpoints"),
                "results_dir": str(test_dir / "results"),
                "log_level": "INFO"
            }
        }
        
        print("1. Creating logger...")
        logger = Logger(test_config)
        print(f"   Logger created: {logger.experiment_name}")
        
        print("2. Checking log directory...")
        log_dir = Path(test_config["logging"]["log_dir"])
        print(f"   Log directory: {log_dir}")
        print(f"   Directory exists: {log_dir.exists()}")
        
        print("3. Logging test messages...")
        logger.log_step({"debug_loss": 2.5, "debug_accuracy": 0.6}, step=0)
        logger.log_step({"debug_loss": 2.0, "debug_accuracy": 0.7}, step=1)
        logger.log_step({"debug_loss": 1.5, "debug_accuracy": 0.8}, step=2)
        
        # Explicit logging
        logger.logger.info("Direct logger test message")
        
        print("4. Flushing logger...")
        logger.flush()
        
        print("5. Checking log files...")
        log_files = list(log_dir.glob("*.log"))
        print(f"   Found {len(log_files)} log files: {[f.name for f in log_files]}")
        
        if log_files:
            # Read each log file
            for i, log_file in enumerate(log_files):
                print(f"\n   === Log File {i+1}: {log_file.name} ===")
                print(f"   Size: {log_file.stat().st_size} bytes")
                
                with open(log_file, 'r') as f:
                    content = f.read()
                
                print(f"   Content length: {len(content)} characters")
                
                # Check for our test messages
                has_debug_loss = "debug_loss" in content
                has_debug_accuracy = "debug_accuracy" in content
                has_direct_message = "Direct logger test message" in content
                
                print(f"   Contains 'debug_loss': {has_debug_loss}")
                print(f"   Contains 'debug_accuracy': {has_debug_accuracy}")
                print(f"   Contains direct message: {has_direct_message}")
                
                if content:
                    print(f"   Last 300 chars:")
                    print(f"   {content[-300:]}")
                else:
                    print(f"   File is empty!")
                
                if has_debug_loss and has_debug_accuracy:
                    print(f"   ✅ Log file {log_file.name} contains expected content!")
                    return True
        
        print("❌ No log files contain the expected test messages")
        
        # Additional debugging
        print("\n6. Logger handler info...")
        print(f"   Logger name: {logger.logger.name}")
        print(f"   Logger level: {logger.logger.level}")
        print(f"   Logger handlers: {logger.logger.handlers}")
        
        for handler in logger.logger.handlers:
            print(f"   Handler: {handler}")
            if hasattr(handler, 'baseFilename'):
                print(f"     File: {handler.baseFilename}")
                print(f"     Level: {handler.level}")
        
        return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            logger.finish()
        except:
            pass
        
        # Remove test directory
        import shutil
        try:
            shutil.rmtree(test_dir)
            print(f"Cleaned up test directory: {test_dir}")
        except:
            print(f"Could not clean up test directory: {test_dir}")


if __name__ == "__main__":
    success = test_logging_functionality()
    print(f"\nTest result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
    sys.exit(0 if success else 1) 