"""
Master test runner for embedding thermalizer infrastructure.
Runs all test suites and provides comprehensive validation.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import warnings

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import test modules
from test_pipeline_dimensions import run_dimension_tests
from test_output_paths import run_output_path_tests


def run_quick_validation():
    """Run quick validation tests that check basic functionality."""
    print("=" * 60)
    print("EMBEDDING THERMALIZER QUICK VALIDATION")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch available: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy available: {np.__version__}")
    except ImportError:
        print("❌ NumPy not available")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib available: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib not available")
        return False
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✓ TensorBoard available")
    except ImportError:
        print("⚠ TensorBoard not available (optional)")
    
    try:
        import wandb
        print("✓ Weights & Biases available")
    except ImportError:
        print("⚠ Weights & Biases not available (optional)")
    
    # Test basic imports from our modules
    try:
        from utils.logging_utils import Logger
        from utils.embedding_utils import EmbeddingConverter
        from utils.model_loading import load_gpt_model
        print("✓ All utility modules import successfully")
    except ImportError as e:
        print(f"❌ Module import failed: {e}")
        return False
    
    # Test configuration loading
    try:
        config_path = "embedding_thermalizer/configs/experiment_config.json"
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("✓ Configuration file loads successfully")
        else:
            print("⚠ Configuration file not found (will be created during setup)")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False
    
    print("=" * 60)
    print("🎉 QUICK VALIDATION PASSED!")
    print("=" * 60)
    return True


def run_comprehensive_tests():
    """Run all comprehensive test suites."""
    print("=" * 60)
    print("EMBEDDING THERMALIZER COMPREHENSIVE TESTS")
    print("=" * 60)
    
    test_results = {}
    total_start_time = time.time()
    
    # Run pipeline dimension tests
    print("\n" + "=" * 40)
    print("1. PIPELINE DIMENSION TESTS")
    print("=" * 40)
    start_time = time.time()
    test_results['dimensions'] = run_dimension_tests()
    test_results['dimensions_time'] = time.time() - start_time
    
    # Run output path tests
    print("\n" + "=" * 40)
    print("2. OUTPUT PATH TESTS")
    print("=" * 40)
    start_time = time.time()
    test_results['paths'] = run_output_path_tests()
    test_results['paths_time'] = time.time() - start_time
    
    # Summary
    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in test_results.items():
        if test_name.endswith('_time'):
            continue
        status = "✓ PASSED" if passed else "❌ FAILED"
        duration = test_results.get(f"{test_name}_time", 0)
        print(f"{test_name.upper():20} {status:10} ({duration:.2f}s)")
        if not passed:
            all_passed = False
    
    print("-" * 60)
    print(f"Total test time: {total_time:.2f}s")
    
    if all_passed:
        print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED - Check details above")
    
    print("=" * 60)
    return all_passed


def run_integration_test():
    """Run a simplified integration test to verify the full pipeline."""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST")
    print("=" * 60)
    
    try:
        import torch
        from utils.embedding_utils import EmbeddingConverter, create_embedding_corruption
        from utils.logging_utils import Logger
        
        # Create test configuration
        test_config = {
            "experiment_name": "integration_test",
            "logging": {
                "use_wandb": False,
                "use_tensorboard": False,
                "log_dir": "embedding_thermalizer/tests/integration/logs",
                "save_dir": "embedding_thermalizer/tests/integration/checkpoints",
                "results_dir": "embedding_thermalizer/tests/integration/results"
            }
        }
        
        print("1. Testing Logger initialization...")
        logger = Logger(test_config)
        print("   ✓ Logger initialized")
        
        print("2. Testing embedding conversion...")
        vocab_size, embed_dim = 1024, 256
        batch_size, seq_len = 2, 16
        
        # Create mock components
        codebook = torch.randn(vocab_size, embed_dim)
        converter = EmbeddingConverter(None, codebook, "vqgan", "cpu")
        
        # Test conversion pipeline
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        embeddings = converter.tokens_to_embeddings(tokens)
        reconstructed_tokens = converter.embeddings_to_tokens(embeddings)
        
        assert embeddings.shape == (batch_size, seq_len, embed_dim), f"Wrong embedding shape: {embeddings.shape}"
        assert reconstructed_tokens.shape == tokens.shape, f"Wrong token shape: {reconstructed_tokens.shape}"
        print("   ✓ Token-embedding conversion works")
        
        print("3. Testing corruption methods...")
        clean_embeddings = torch.randn(batch_size, embed_dim)
        corrupted, info = create_embedding_corruption(clean_embeddings, "gaussian", 0.1)
        assert corrupted.shape == clean_embeddings.shape, "Corruption changed shape"
        print(f"   ✓ Corruption works ({info})")
        
        print("4. Testing logging...")
        logger.log_step({"test_loss": 1.5, "test_accuracy": 0.8}, step=0)
        logger.log_step({"test_loss": 1.2, "test_accuracy": 0.85}, step=1)
        print("   ✓ Logging works")
        
        print("5. Testing model saving...")
        mock_model = torch.nn.Linear(embed_dim, 128)
        mock_optimizer = torch.optim.Adam(mock_model.parameters())
        checkpoint_path = logger.save_checkpoint(mock_model, mock_optimizer, epoch=1, step=10)
        assert checkpoint_path.exists(), "Checkpoint not saved"
        print(f"   ✓ Model saving works: {checkpoint_path}")
        
        logger.finish()
        
        print("\n🎉 INTEGRATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_report(results):
    """Create a detailed test report."""
    report_path = Path("embedding_thermalizer/tests/test_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# Embedding Thermalizer Test Report\n\n")
        f.write(f"**Test Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Test Results Summary\n\n")
        for test_name, passed in results.items():
            if test_name.endswith('_time'):
                continue
            status = "✅ PASSED" if passed else "❌ FAILED"
            duration = results.get(f"{test_name}_time", 0)
            f.write(f"- **{test_name.title()}:** {status} ({duration:.2f}s)\n")
        
        f.write("\n## System Information\n\n")
        try:
            import torch
            f.write(f"- **PyTorch Version:** {torch.__version__}\n")
            f.write(f"- **CUDA Available:** {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"- **GPU:** {torch.cuda.get_device_name()}\n")
        except ImportError:
            f.write("- **PyTorch:** Not available\n")
        
        f.write(f"- **Python Version:** {sys.version}\n")
        f.write(f"- **Platform:** {sys.platform}\n")
        
        f.write("\n## Next Steps\n\n")
        if all(results[k] for k in results.keys() if not k.endswith('_time')):
            f.write("✅ All tests passed! You can proceed with:\n")
            f.write("1. Data preparation: `bash embedding_thermalizer/scripts/run_data_preparation.sh`\n")
            f.write("2. Training: `bash embedding_thermalizer/scripts/run_training.sh`\n")
            f.write("3. Evaluation: `bash embedding_thermalizer/scripts/run_evaluation.sh`\n")
        else:
            f.write("❌ Some tests failed. Please address the issues before proceeding.\n")
    
    print(f"\n📄 Test report saved: {report_path}")
    return report_path


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Embedding Thermalizer Test Runner")
    parser.add_argument("--mode", choices=["quick", "comprehensive", "integration", "all"], 
                       default="all", help="Test mode to run")
    parser.add_argument("--report", action="store_true", help="Generate test report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Suppress warnings unless verbose
    if not args.verbose:
        warnings.filterwarnings("ignore")
    
    print(f"\nStarting Embedding Thermalizer Tests (mode: {args.mode})")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    overall_success = True
    
    if args.mode in ["quick", "all"]:
        print("\n" + "🚀 RUNNING QUICK VALIDATION...")
        results['quick_validation'] = run_quick_validation()
        overall_success &= results['quick_validation']
    
    if args.mode in ["comprehensive", "all"]:
        print("\n" + "🧪 RUNNING COMPREHENSIVE TESTS...")
        results.update({
            'dimensions': False,
            'paths': False
        })
        
        # Only run comprehensive tests if quick validation passed
        if args.mode == "all" and not results.get('quick_validation', True):
            print("⚠ Skipping comprehensive tests due to quick validation failure")
        else:
            comprehensive_success = run_comprehensive_tests()
            # Update results with individual test results
            # (These get updated inside run_comprehensive_tests)
            overall_success &= comprehensive_success
    
    if args.mode in ["integration", "all"]:
        print("\n" + "🔧 RUNNING INTEGRATION TEST...")
        results['integration'] = run_integration_test()
        overall_success &= results['integration']
    
    # Generate report if requested
    if args.report:
        create_test_report(results)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    
    if overall_success:
        print("🎉 ALL TESTS PASSED! Your embedding thermalizer setup is ready.")
        print("\nNext steps:")
        print("1. Update config paths in embedding_thermalizer/configs/experiment_config.json")
        print("2. Run data preparation: bash embedding_thermalizer/scripts/run_data_preparation.sh")
        print("3. Start training: bash embedding_thermalizer/scripts/run_training.sh")
    else:
        print("❌ SOME TESTS FAILED. Please fix the issues before proceeding.")
        print("\nCheck the test output above for specific failures.")
    
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main() 