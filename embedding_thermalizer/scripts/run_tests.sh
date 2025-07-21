#!/bin/bash

# Embedding Thermalizer Test Runner Script
# This script runs comprehensive tests to validate the infrastructure

set -e  # Exit on any error

echo "=== Embedding Thermalizer Test Suite ==="
echo "Starting tests at $(date)"

# Configuration
PYTHON_PATH="$(which python)"
TEST_MODE="all"
GENERATE_REPORT=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            TEST_MODE="$2"
            shift 2
            ;;
        --report)
            GENERATE_REPORT=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --mode [quick|comprehensive|integration|all]  Test mode (default: all)"
            echo "  --report                                       Generate test report"
            echo "  --verbose                                      Verbose output"
            echo "  -h, --help                                     Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup environment
echo "Using Python: $PYTHON_PATH"
echo "Test mode: $TEST_MODE"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check Python version
PYTHON_VERSION=$($PYTHON_PATH --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if we're in the right directory
if [ ! -f "embedding_thermalizer/configs/experiment_config.json" ]; then
    echo "Warning: experiment_config.json not found. Tests will create a default configuration."
fi

# Check basic dependencies
echo ""
echo "=== Checking Dependencies ==="

# Check PyTorch
if $PYTHON_PATH -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
    echo "✓ PyTorch available"
else
    echo "❌ PyTorch not available - please install: pip install torch"
    exit 1
fi

# Check CUDA
if [ "$($PYTHON_PATH -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    GPU_NAME=$($PYTHON_PATH -c 'import torch; print(torch.cuda.get_device_name())' 2>/dev/null || echo "Unknown GPU")
    echo "✓ CUDA available - GPU: $GPU_NAME"
else
    echo "⚠ CUDA not available - tests will run on CPU"
fi

# Check other dependencies
DEPS=("numpy" "matplotlib" "pathlib" "json")
for dep in "${DEPS[@]}"; do
    if $PYTHON_PATH -c "import $dep" 2>/dev/null; then
        echo "✓ $dep available"
    else
        echo "❌ $dep not available - please install: pip install $dep"
        exit 1
    fi
done

# Check optional dependencies
echo ""
echo "=== Optional Dependencies ==="
if $PYTHON_PATH -c "from torch.utils.tensorboard import SummaryWriter" 2>/dev/null; then
    echo "✓ TensorBoard available"
else
    echo "⚠ TensorBoard not available (install with: pip install tensorboard)"
fi

if $PYTHON_PATH -c "import wandb" 2>/dev/null; then
    echo "✓ Weights & Biases available"
else
    echo "⚠ Weights & Biases not available (install with: pip install wandb)"
fi

# Create test directories
echo ""
echo "=== Setting up Test Environment ==="
mkdir -p embedding_thermalizer/tests/logs
mkdir -p embedding_thermalizer/tests/temp
mkdir -p embedding_thermalizer/tests/results

echo "✓ Test directories created"

# Run the tests
echo ""
echo "=== Running Tests ==="

cd embedding_thermalizer

# Build command
TEST_CMD="$PYTHON_PATH tests/run_all_tests.py --mode $TEST_MODE"

if [ "$GENERATE_REPORT" = true ]; then
    TEST_CMD="$TEST_CMD --report"
fi

if [ "$VERBOSE" = true ]; then
    TEST_CMD="$TEST_CMD --verbose"
fi

echo "Executing: $TEST_CMD"
echo ""

# Run tests and capture exit code
if $TEST_CMD; then
    TEST_SUCCESS=true
    echo ""
    echo "✅ ALL TESTS PASSED!"
else
    TEST_SUCCESS=false
    echo ""
    echo "❌ SOME TESTS FAILED!"
fi

cd ..

# Display results
echo ""
echo "=== Test Results Summary ==="
echo "Test mode: $TEST_MODE"
echo "Test time: $(date)"

if [ "$TEST_SUCCESS" = true ]; then
    echo "Status: ✅ PASSED"
    
    # Show generated files
    if [ -d "embedding_thermalizer/tests" ]; then
        echo ""
        echo "Generated test files:"
        find embedding_thermalizer/tests -name "*.log" -o -name "*.md" -o -name "*.json" | head -10 | sed 's/^/  /'
    fi
    
    echo ""
    echo "🚀 Your embedding thermalizer infrastructure is ready!"
    echo ""
    echo "Next steps:"
    echo "1. Configure your experiment:"
    echo "   nano embedding_thermalizer/configs/experiment_config.json"
    echo ""
    echo "2. Run data preparation:"
    echo "   bash embedding_thermalizer/scripts/run_data_preparation.sh"
    echo ""
    echo "3. Start training:"
    echo "   bash embedding_thermalizer/scripts/run_training.sh"
    echo ""
    echo "4. Run evaluation:"
    echo "   bash embedding_thermalizer/scripts/run_evaluation.sh"
    
else
    echo "Status: ❌ FAILED"
    echo ""
    echo "Please review the test output above and fix any issues."
    echo "Common solutions:"
    echo "  - Install missing dependencies"
    echo "  - Check file permissions"
    echo "  - Verify Python environment"
    echo "  - Check available disk space"
    echo ""
    echo "For detailed logs, check:"
    echo "  - embedding_thermalizer/tests/logs/"
    
    if [ -f "embedding_thermalizer/tests/test_report.md" ]; then
        echo "  - embedding_thermalizer/tests/test_report.md"
    fi
fi

# Show test report if generated
if [ "$GENERATE_REPORT" = true ] && [ -f "embedding_thermalizer/tests/test_report.md" ]; then
    echo ""
    echo "📄 Test report generated: embedding_thermalizer/tests/test_report.md"
    echo ""
    echo "Report preview:"
    head -20 embedding_thermalizer/tests/test_report.md | sed 's/^/  /'
fi

echo ""
echo "=== Test Complete ==="
echo "Finished at $(date)"

# Exit with appropriate code
if [ "$TEST_SUCCESS" = true ]; then
    exit 0
else
    exit 1
fi 