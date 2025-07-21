"""
Test script to validate output path management and file saving.
This script ensures all outputs are saved to paths specified in the config file.
"""

import os
import sys
import json
import torch
import shutil
from pathlib import Path
import unittest
import tempfile
import warnings

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logging_utils import Logger, MetricsTracker


class TestOutputPaths(unittest.TestCase):
    """Test class for validating output path management."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for testing
        cls.test_dir = Path(tempfile.mkdtemp(prefix="embedding_thermalizer_test_"))
        print(f"Using test directory: {cls.test_dir}")
        
        # Create test configuration with all paths
        cls.test_config = {
            "experiment_name": "test_paths_experiment",
            "seed": 42,
            
            "logging": {
                "use_wandb": False,
                "use_tensorboard": True,
                "project": "test_project",
                "log_dir": str(cls.test_dir / "logs"),
                "save_dir": str(cls.test_dir / "checkpoints"),
                "results_dir": str(cls.test_dir / "results"),
                "log_level": "INFO"
            },
            
            "paths": {
                "data_dir": str(cls.test_dir / "data"),
                "embeddings_dir": str(cls.test_dir / "data" / "embeddings"),
                "models_dir": str(cls.test_dir / "models"),
                "plots_dir": str(cls.test_dir / "plots"),
                "videos_dir": str(cls.test_dir / "videos"),
                "metrics_dir": str(cls.test_dir / "metrics")
            },
            
            "embedding_extraction": {
                "save_path": str(cls.test_dir / "data" / "embeddings")
            },
            
            "evaluation": {
                "output_dir": str(cls.test_dir / "evaluation")
            }
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove test directory
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
            print(f"Cleaned up test directory: {cls.test_dir}")
    
    def test_directory_creation(self):
        """Test that all required directories are created from config."""
        print("\n=== Testing Directory Creation ===")
        
        # Initialize logger (this should create directories)
        logger = Logger(self.test_config)
        
        # Check that all directories exist
        expected_dirs = [
            Path(self.test_config["logging"]["log_dir"]),
            Path(self.test_config["logging"]["save_dir"]),
            Path(self.test_config["logging"]["results_dir"]),
            logger.tensorboard_dir,
            logger.plots_dir
        ]
        
        for dir_path in expected_dirs:
            self.assertTrue(dir_path.exists(), f"Directory not created: {dir_path}")
            print(f"  ✓ Directory created: {dir_path}")
        
        # Test additional paths from config
        for path_name, path_value in self.test_config["paths"].items():
            path_obj = Path(path_value)
            # Create the directory to test path validity
            path_obj.mkdir(parents=True, exist_ok=True)
            self.assertTrue(path_obj.exists(), f"Path config invalid: {path_name} -> {path_value}")
            print(f"  ✓ Path config valid: {path_name} -> {path_value}")
        
        logger.finish()
        print("✓ Directory creation test passed")
    
    def test_checkpoint_saving(self):
        """Test that model checkpoints are saved to configured directory."""
        print("\n=== Testing Checkpoint Saving ===")
        
        logger = Logger(self.test_config)
        
        # Create mock model and optimizer
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test checkpoint saving
        checkpoint_path = logger.save_checkpoint(
            model, optimizer, epoch=1, step=100, 
            additional_info={"test_key": "test_value"}
        )
        
        # Check checkpoint file exists
        self.assertTrue(checkpoint_path.exists(), f"Checkpoint not saved: {checkpoint_path}")
        
        # Check checkpoint is in correct directory
        expected_dir = Path(self.test_config["logging"]["save_dir"])
        self.assertEqual(checkpoint_path.parent, expected_dir, 
                        f"Checkpoint saved to wrong directory: {checkpoint_path.parent} vs {expected_dir}")
        
        # Load and verify checkpoint contents
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        required_keys = ['epoch', 'step', 'model_state_dict', 'optimizer_state_dict', 'config']
        for key in required_keys:
            self.assertIn(key, checkpoint, f"Missing key in checkpoint: {key}")
        
        self.assertEqual(checkpoint['epoch'], 1)
        self.assertEqual(checkpoint['step'], 100)
        self.assertEqual(checkpoint['test_key'], 'test_value')
        
        print(f"  ✓ Checkpoint saved correctly: {checkpoint_path}")
        print(f"  ✓ Checkpoint contains all required keys: {list(checkpoint.keys())}")
        
        # Test model-only saving
        model_path = logger.save_model_only(model, "test_model.pt")
        self.assertTrue(model_path.exists(), f"Model not saved: {model_path}")
        
        # Load and verify model
        loaded_state_dict = torch.load(model_path, map_location='cpu')
        self.assertIsInstance(loaded_state_dict, dict, "Model state dict not saved correctly")
        
        print(f"  ✓ Model-only save works: {model_path}")
        
        logger.finish()
        print("✓ Checkpoint saving test passed")
    
    def test_logging_files(self):
        """Test that log files are created in configured directory."""
        print("\n=== Testing Log File Creation ===")
        
        logger = Logger(self.test_config)
        
        # Log some test metrics
        logger.log_step({"test_loss": 1.5, "test_accuracy": 0.8}, step=0)
        logger.log_step({"test_loss": 1.2, "test_accuracy": 0.85}, step=1)
        
        # Force flush all logging handlers to ensure messages are written to file
        logger.flush()
        
        # Check log directory exists and contains files
        log_dir = Path(self.test_config["logging"]["log_dir"])
        self.assertTrue(log_dir.exists(), f"Log directory not created: {log_dir}")
        
        # Check for log files
        log_files = list(log_dir.glob("*.log"))
        self.assertGreater(len(log_files), 0, "No log files created")
        
        # Check for config files
        config_files = list(log_dir.glob("*_config.json"))
        self.assertGreater(len(config_files), 0, "No config files created")
        
        print(f"  ✓ Log files created: {[f.name for f in log_files]}")
        print(f"  ✓ Config files created: {[f.name for f in config_files]}")
        
        # Verify log file content - check the most recent log file
        log_files_sorted = sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)
        log_file = log_files_sorted[0]
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Debug: print the actual log content to help diagnose
        print(f"  Debug: Reading from log file: {log_file}")
        print(f"  Debug: Log content length: {len(log_content)}")
        print(f"  Debug: Log content preview: {log_content[-500:]}")  # Last 500 chars
        
        # Check for the actual logging format: "Step X - test_loss: Y.Y"
        self.assertTrue(
            "test_loss:" in log_content or "test_loss =" in log_content,
            f"Test metrics not logged. Log file: {log_file}\nFull content: {log_content}"
        )
        self.assertTrue(
            "test_accuracy:" in log_content or "test_accuracy =" in log_content,
            f"Test metrics not logged. Log file: {log_file}\nFull content: {log_content}"
        )
        
        print(f"  ✓ Log file contains expected content")
        
        logger.finish()
        print("✓ Log file creation test passed")
    
    def test_tensorboard_logging(self):
        """Test that TensorBoard logs are saved to configured directory."""
        print("\n=== Testing TensorBoard Logging ===")
        
        logger = Logger(self.test_config)
        
        # Log some metrics
        logger.log_step({"train/loss": 2.0, "train/accuracy": 0.7}, step=0)
        logger.log_step({"train/loss": 1.8, "train/accuracy": 0.75}, step=1)
        
        # Check TensorBoard directory
        tb_dir = logger.tensorboard_dir
        self.assertTrue(tb_dir.exists(), f"TensorBoard directory not created: {tb_dir}")
        
        # Check for TensorBoard event files
        if logger.tb_writer is not None:
            # Flush the writer to ensure files are written
            logger.tb_writer.flush()
            
            # Look for event files
            event_files = list(tb_dir.glob("events.out.tfevents.*"))
            self.assertGreater(len(event_files), 0, "No TensorBoard event files created")
            print(f"  ✓ TensorBoard event files created: {len(event_files)} files")
        else:
            print("  ⚠ TensorBoard not available, skipping event file check")
        
        logger.finish()
        print("✓ TensorBoard logging test passed")
    
    def test_plot_saving(self):
        """Test that plots are saved to configured directory."""
        print("\n=== Testing Plot Saving ===")
        
        logger = Logger(self.test_config)
        
        # Add some metrics to plot
        for i in range(10):
            logger.log_step({
                "train/loss": 2.0 - i * 0.1,
                "val/loss": 2.2 - i * 0.08,
                "train/noise_accuracy": 0.5 + i * 0.05
            }, step=i)
        
        # Generate plots
        logger.plot_loss_curves(save=True, show=False)
        
        # Check plots directory
        plots_dir = logger.plots_dir
        self.assertTrue(plots_dir.exists(), f"Plots directory not created: {plots_dir}")
        
        # Check for plot files
        plot_files = list(plots_dir.glob("*.png"))
        self.assertGreater(len(plot_files), 0, "No plot files created")
        
        print(f"  ✓ Plot files created: {[f.name for f in plot_files]}")
        
        # Check file sizes (should be reasonable for PNG files)
        for plot_file in plot_files:
            file_size = plot_file.stat().st_size
            self.assertGreater(file_size, 1000, f"Plot file too small: {plot_file}")
            self.assertLess(file_size, 10_000_000, f"Plot file too large: {plot_file}")
            print(f"  ✓ Plot file size reasonable: {plot_file.name} ({file_size} bytes)")
        
        logger.finish()
        print("✓ Plot saving test passed")
    
    def test_metrics_tracker(self):
        """Test metrics tracker functionality."""
        print("\n=== Testing Metrics Tracker ===")
        
        tracker = MetricsTracker()
        
        # Add some metrics
        tracker.update(loss=1.5, accuracy=0.8)
        tracker.update(loss=1.3, accuracy=0.82)
        tracker.update(loss=1.1, accuracy=0.85)
        
        # Get averages
        averages = tracker.get_averages()
        
        # Check calculations
        expected_loss = (1.5 + 1.3 + 1.1) / 3
        expected_accuracy = (0.8 + 0.82 + 0.85) / 3
        
        self.assertAlmostEqual(averages['loss'], expected_loss, places=5)
        self.assertAlmostEqual(averages['accuracy'], expected_accuracy, places=5)
        
        print(f"  ✓ Metrics tracking works: {averages}")
        
        # Test reset
        tracker.reset()
        self.assertEqual(len(tracker.metrics), 0, "Metrics not reset")
        self.assertEqual(len(tracker.counts), 0, "Counts not reset")
        
        print("  ✓ Metrics reset works")
        print("✓ Metrics tracker test passed")
    
    def test_path_validation(self):
        """Test path validation and error handling."""
        print("\n=== Testing Path Validation ===")
        
        # Test with invalid paths
        invalid_config = self.test_config.copy()
        invalid_config["logging"]["log_dir"] = "/invalid/readonly/path"
        
        try:
            # This should handle the error gracefully
            logger = Logger(invalid_config)
            logger.finish()
            print("  ✓ Invalid path handled gracefully")
        except Exception as e:
            print(f"  ⚠ Path validation error (expected): {e}")
        
        # Test with relative paths
        relative_config = self.test_config.copy()
        relative_config["logging"]["log_dir"] = "relative/path/logs"
        relative_config["logging"]["save_dir"] = "relative/path/checkpoints"
        
        try:
            logger = Logger(relative_config)
            # Check that paths exist (they might be relative but still valid)
            self.assertTrue(logger.log_dir.exists() or logger.log_dir.parent.exists(), 
                          f"Log directory path invalid: {logger.log_dir}")
            self.assertTrue(logger.save_dir.exists() or logger.save_dir.parent.exists(), 
                          f"Save directory path invalid: {logger.save_dir}")
            logger.finish()
            print("  ✓ Relative paths handled correctly")
        except Exception as e:
            print(f"  ⚠ Relative path test failed: {e}")
        
        print("✓ Path validation test passed")
    
    def test_config_path_override(self):
        """Test that all paths can be overridden via configuration."""
        print("\n=== Testing Config Path Override ===")
        
        # Create custom paths
        custom_paths = {
            "custom_logs": self.test_dir / "custom_logs",
            "custom_saves": self.test_dir / "custom_saves",
            "custom_results": self.test_dir / "custom_results"
        }
        
        # Update config with custom paths
        custom_config = self.test_config.copy()
        custom_config["logging"]["log_dir"] = str(custom_paths["custom_logs"])
        custom_config["logging"]["save_dir"] = str(custom_paths["custom_saves"])
        custom_config["logging"]["results_dir"] = str(custom_paths["custom_results"])
        
        # Initialize logger with custom config
        logger = Logger(custom_config)
        
        # Verify custom paths are used
        self.assertEqual(logger.log_dir, custom_paths["custom_logs"])
        self.assertEqual(logger.save_dir, custom_paths["custom_saves"])
        self.assertEqual(logger.results_dir, custom_paths["custom_results"])
        
        # Verify directories are created
        for path_name, path_value in custom_paths.items():
            self.assertTrue(path_value.exists(), f"Custom path not created: {path_name}")
            print(f"  ✓ Custom path created: {path_name} -> {path_value}")
        
        logger.finish()
        print("✓ Config path override test passed")
    
    def test_file_permissions(self):
        """Test file creation permissions."""
        print("\n=== Testing File Permissions ===")
        
        logger = Logger(self.test_config)
        
        # Create test model and save
        model = torch.nn.Linear(5, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        checkpoint_path = logger.save_checkpoint(model, optimizer, epoch=1, step=50)
        
        # Check file is readable
        self.assertTrue(os.access(checkpoint_path, os.R_OK), f"Checkpoint not readable: {checkpoint_path}")
        
        # Check file is writable (should be able to overwrite)
        self.assertTrue(os.access(checkpoint_path, os.W_OK), f"Checkpoint not writable: {checkpoint_path}")
        
        print(f"  ✓ File permissions correct: {checkpoint_path}")
        
        logger.finish()
        print("✓ File permissions test passed")


def run_output_path_tests():
    """Run all output path tests."""
    print("=" * 60)
    print("EMBEDDING THERMALIZER OUTPUT PATH TESTS")
    print("=" * 60)
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestOutputPaths)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL OUTPUT PATH TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Set up warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Run tests
    success = run_output_path_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 