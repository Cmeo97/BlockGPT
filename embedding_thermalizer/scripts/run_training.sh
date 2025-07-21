#!/bin/bash

# Embedding Thermalizer Training Script
# This script trains both the noise classifier and diffusion model components

set -e  # Exit on any error

echo "=== Embedding Thermalizer Training ==="
echo "Starting training at $(date)"

# Configuration
CONFIG_FILE="embedding_thermalizer/configs/experiment_config.json"
PYTHON_PATH="$(which python)"

# Parse command line arguments
TRAIN_MODE="both"  # Default: train both models
RESUME_FROM=""
GPU_ID=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            TRAIN_MODE="$2"
            shift 2
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --mode [both|classifier|diffusion]  Training mode (default: both)"
            echo "  --resume PATH                       Resume from checkpoint"
            echo "  --gpu ID                           GPU ID to use (default: 0)"
            echo "  --config PATH                      Config file path"
            echo "  -h, --help                         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate training mode
if [[ ! "$TRAIN_MODE" =~ ^(both|classifier|diffusion)$ ]]; then
    echo "Error: Invalid training mode '$TRAIN_MODE'. Use 'both', 'classifier', or 'diffusion'."
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    exit 1
fi

# Check if embedding data exists
DATA_DIR="embedding_thermalizer/data/embeddings"
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Embedding data directory not found at $DATA_DIR"
    echo "Please run data preparation first: bash embedding_thermalizer/scripts/run_data_preparation.sh"
    exit 1
fi

# Validate data files
REQUIRED_FILES=("clean_embeddings.pt" "corrupted_embeddings.pt" "noise_levels.pt" "metadata.json")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "Error: Required data file not found: $DATA_DIR/$file"
        echo "Please run data preparation first."
        exit 1
    fi
done

# Setup environment
echo "Using Python: $PYTHON_PATH"
echo "Training mode: $TRAIN_MODE"
echo "GPU ID: $GPU_ID"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Create directories
mkdir -p embedding_thermalizer/logs/training
mkdir -p embedding_thermalizer/checkpoints
mkdir -p Results/embedding_thermalizer

echo "Environment setup complete."

cd embedding_thermalizer

# Function to train noise classifier
train_noise_classifier() {
    echo ""
    echo "=== Training Noise Classifier ==="
    echo "Starting noise classifier training at $(date)"
    
    CLASSIFIER_LOG="logs/training/noise_classifier_$(date +%Y%m%d_%H%M%S).log"
    
    $PYTHON_PATH training/train_embedding_thermalizer.py \
        --config configs/experiment_config.json \
        --mode noise_classifier \
        --output_dir checkpoints/noise_classifier \
        ${RESUME_FROM:+--resume_from $RESUME_FROM} \
        2>&1 | tee "$CLASSIFIER_LOG"
    
    if [ $? -eq 0 ]; then
        echo "Noise classifier training completed successfully!"
        echo "Log saved to: $CLASSIFIER_LOG"
        
        # Display training summary
        echo ""
        echo "=== Noise Classifier Training Summary ==="
        if [ -f "checkpoints/noise_classifier/training_summary.json" ]; then
            cat checkpoints/noise_classifier/training_summary.json | python -m json.tool
        fi
        
        return 0
    else
        echo "Noise classifier training failed!"
        echo "Check log file: $CLASSIFIER_LOG"
        return 1
    fi
}

# Function to train diffusion model
train_diffusion_model() {
    echo ""
    echo "=== Training Diffusion Model ==="
    echo "Starting diffusion model training at $(date)"
    
    DIFFUSION_LOG="logs/training/diffusion_model_$(date +%Y%m%d_%H%M%S).log"
    
    $PYTHON_PATH training/train_embedding_thermalizer.py \
        --config configs/experiment_config.json \
        --mode diffusion \
        --output_dir checkpoints/diffusion_model \
        ${RESUME_FROM:+--resume_from $RESUME_FROM} \
        2>&1 | tee "$DIFFUSION_LOG"
    
    if [ $? -eq 0 ]; then
        echo "Diffusion model training completed successfully!"
        echo "Log saved to: $DIFFUSION_LOG"
        
        # Display training summary
        echo ""
        echo "=== Diffusion Model Training Summary ==="
        if [ -f "checkpoints/diffusion_model/training_summary.json" ]; then
            cat checkpoints/diffusion_model/training_summary.json | python -m json.tool
        fi
        
        return 0
    else
        echo "Diffusion model training failed!"
        echo "Check log file: $DIFFUSION_LOG"
        return 1
    fi
}

# Function to display system info
display_system_info() {
    echo ""
    echo "=== System Information ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "User: $(whoami)"
    echo "Working directory: $(pwd)"
    echo "Python version: $($PYTHON_PATH --version)"
    
    if [ "$($PYTHON_PATH -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
        echo "CUDA version: $($PYTHON_PATH -c 'import torch; print(torch.version.cuda)')"
        echo "GPU: $($PYTHON_PATH -c 'import torch; print(torch.cuda.get_device_name())' 2>/dev/null || echo 'Unknown')"
        echo "GPU memory: $($PYTHON_PATH -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB\")' 2>/dev/null || echo 'Unknown')"
    else
        echo "CUDA: Not available"
    fi
    
    echo "Available disk space:"
    df -h . | tail -1
}

# Display system information
display_system_info

# Execute training based on mode
case $TRAIN_MODE in
    "classifier")
        train_noise_classifier
        TRAIN_SUCCESS=$?
        ;;
    "diffusion")
        train_diffusion_model
        TRAIN_SUCCESS=$?
        ;;
    "both")
        echo "Training both models sequentially..."
        train_noise_classifier
        CLASSIFIER_SUCCESS=$?
        
        if [ $CLASSIFIER_SUCCESS -eq 0 ]; then
            train_diffusion_model
            DIFFUSION_SUCCESS=$?
            
            if [ $DIFFUSION_SUCCESS -eq 0 ]; then
                TRAIN_SUCCESS=0
            else
                TRAIN_SUCCESS=1
            fi
        else
            echo "Skipping diffusion model training due to classifier training failure."
            TRAIN_SUCCESS=1
        fi
        ;;
esac

cd ..

# Final summary
echo ""
echo "=== Training Complete ==="
echo "Finished at $(date)"

if [ $TRAIN_SUCCESS -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    echo ""
    echo "=== Generated Models ==="
    find embedding_thermalizer/checkpoints -name "*.pt" -o -name "*.pth" | head -10
    
    echo ""
    echo "=== Next Steps ==="
    echo "1. Review training logs in embedding_thermalizer/logs/training/"
    echo "2. Check model checkpoints in embedding_thermalizer/checkpoints/"
    echo "3. Run evaluation: bash embedding_thermalizer/scripts/run_evaluation.sh"
    echo "4. Visualize results using the generated plots"
    
else
    echo "❌ Training failed!"
    echo ""
    echo "=== Troubleshooting ==="
    echo "1. Check training logs in embedding_thermalizer/logs/training/"
    echo "2. Verify data integrity in embedding_thermalizer/data/embeddings/"
    echo "3. Check GPU memory usage and reduce batch_size if needed"
    echo "4. Ensure proper environment setup and dependencies"
    echo ""
    echo "Common fixes:"
    echo "  - Reduce batch_size in configs/experiment_config.json"
    echo "  - Check CUDA memory with: nvidia-smi"
    echo "  - Verify data files exist and are not corrupted"
    
    exit 1
fi 