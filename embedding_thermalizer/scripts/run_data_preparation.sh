#!/bin/bash

# Embedding Thermalizer Data Preparation Script
# This script extracts clean embeddings from real data and generates corrupted/drifted embeddings

set -e  # Exit on any error

echo "=== Embedding Thermalizer Data Preparation ==="
echo "Starting data preparation at $(date)"

# Configuration
CONFIG_FILE="embedding_thermalizer/configs/experiment_config.json"
PYTHON_PATH="$(which python)"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    echo "Please create the configuration file first."
    exit 1
fi

# Check if Python environment is properly set up
echo "Using Python: $PYTHON_PATH"
echo "Python version: $($PYTHON_PATH --version)"

# Check CUDA availability
echo "CUDA available: $($PYTHON_PATH -c 'import torch; print(torch.cuda.is_available())')"
if [ "$($PYTHON_PATH -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "CUDA devices: $($PYTHON_PATH -c 'import torch; print(torch.cuda.device_count())')"
    echo "Current device: $($PYTHON_PATH -c 'import torch; print(torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\")')"
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p /projects/0/prjs0951/Varun/embedding_thermalizer/logs
mkdir -p /projects/0/prjs0951/Varun/embedding_thermalizer/data/embeddings
mkdir -p /projects/0/prjs0951/Varun/embedding_thermalizer/Results/embedding_thermalizer

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "Environment setup complete."
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run data preparation
echo ""
echo "=== Starting Data Preparation ==="
echo "This may take several hours depending on dataset size and model complexity..."

cd embedding_thermalizer

$PYTHON_PATH data/prepare_embedding_data.py \
    --config_path configs/experiment_config.json \
    2>&1 | tee /projects/0/prjs0951/Varun/embedding_thermalizer/logs/data_preparation_$(date +%Y%m%d_%H%M%S).log

# Check if data preparation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Data Preparation Completed Successfully ==="
    echo "Finished at $(date)"
    
    # Display summary of generated data
    echo ""
    echo "=== Data Summary ==="
    if [ -f "/projects/0/prjs0951/Varun/embedding_thermalizer/data/embeddings/metadata.json" ]; then
        echo "Metadata:"
        cat /projects/0/prjs0951/Varun/embedding_thermalizer/data/embeddings/metadata.json | python -m json.tool
    fi
    
    # List generated files
    echo ""
    echo "Generated files:"
    ls -lh /projects/0/prjs0951/Varun/embedding_thermalizer/data/embeddings/
    
    echo ""
    echo "Data preparation completed successfully!"
    echo "You can now proceed to training the thermalizer models."
    echo "Run: bash embedding_thermalizer/scripts/run_training.sh"
    
else
    echo ""
    echo "=== Data Preparation Failed ==="
    echo "Please check the log files for error details:"
    echo "  - embedding_thermalizer/logs/"
    echo ""
    echo "Common issues and solutions:"
    echo "  1. Missing pretrained models: Update paths in experiment_config.json"
    echo "  2. CUDA memory error: Reduce batch_size in config"
    echo "  3. Dataset not found: Check dataset_name and paths in config"
    echo "  4. Permission errors: Check write permissions for data directory"
    exit 1
fi

echo ""
echo "=== Next Steps ==="
echo "1. Review the generated embedding data in data/embeddings/"
echo "2. Check the metadata.json file for data statistics"
echo "3. Proceed to training with: bash scripts/run_training.sh"
echo "4. Monitor the training logs in logs/ directory"

cd .. 