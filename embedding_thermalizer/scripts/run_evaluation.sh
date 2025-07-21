#!/bin/bash

# Embedding Thermalizer Evaluation Script
# This script evaluates the trained thermalizer by comparing GPT generation with and without thermalization

set -e  # Exit on any error

echo "=== Embedding Thermalizer Evaluation ==="
echo "Starting evaluation at $(date)"

# Configuration
CONFIG_FILE="embedding_thermalizer/configs/experiment_config.json"
PYTHON_PATH="$(which python)"

# Parse command line arguments
EVAL_MODE="full"  # full, quick, or comparison
GPU_ID=0
OUTPUT_DIR="Results/embedding_thermalizer/evaluation"
CLASSIFIER_CHECKPOINT=""
DIFFUSION_CHECKPOINT=""
NUM_SEQUENCES=100
SEQUENCE_LENGTH=50

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            EVAL_MODE="$2"
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
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --classifier-checkpoint)
            CLASSIFIER_CHECKPOINT="$2"
            shift 2
            ;;
        --diffusion-checkpoint)
            DIFFUSION_CHECKPOINT="$2"
            shift 2
            ;;
        --num-sequences)
            NUM_SEQUENCES="$2"
            shift 2
            ;;
        --sequence-length)
            SEQUENCE_LENGTH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --mode [full|quick|comparison]      Evaluation mode (default: full)"
            echo "  --gpu ID                            GPU ID to use (default: 0)"
            echo "  --config PATH                       Config file path"
            echo "  --output DIR                        Output directory"
            echo "  --classifier-checkpoint PATH        Noise classifier checkpoint"
            echo "  --diffusion-checkpoint PATH         Diffusion model checkpoint"
            echo "  --num-sequences N                   Number of sequences to evaluate"
            echo "  --sequence-length N                 Length of each sequence"
            echo "  -h, --help                          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate evaluation mode
if [[ ! "$EVAL_MODE" =~ ^(full|quick|comparison)$ ]]; then
    echo "Error: Invalid evaluation mode '$EVAL_MODE'. Use 'full', 'quick', or 'comparison'."
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    exit 1
fi

# Check if trained models exist
CHECKPOINT_DIR="embedding_thermalizer/checkpoints"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found at $CHECKPOINT_DIR"
    echo "Please run training first: bash embedding_thermalizer/scripts/run_training.sh"
    exit 1
fi

# Auto-detect latest checkpoints if not provided
if [ -z "$CLASSIFIER_CHECKPOINT" ]; then
    CLASSIFIER_CHECKPOINT=$(find "$CHECKPOINT_DIR/noise_classifier" -name "*.pt" -type f -exec ls -t {} + | head -1 2>/dev/null || echo "")
    if [ -z "$CLASSIFIER_CHECKPOINT" ]; then
        echo "Error: No noise classifier checkpoint found. Please provide --classifier-checkpoint"
        exit 1
    fi
    echo "Using classifier checkpoint: $CLASSIFIER_CHECKPOINT"
fi

if [ -z "$DIFFUSION_CHECKPOINT" ]; then
    DIFFUSION_CHECKPOINT=$(find "$CHECKPOINT_DIR/diffusion_model" -name "*.pt" -type f -exec ls -t {} + | head -1 2>/dev/null || echo "")
    if [ -z "$DIFFUSION_CHECKPOINT" ]; then
        echo "Error: No diffusion model checkpoint found. Please provide --diffusion-checkpoint"
        exit 1
    fi
    echo "Using diffusion checkpoint: $DIFFUSION_CHECKPOINT"
fi

# Setup environment
echo "Evaluation mode: $EVAL_MODE"
echo "GPU ID: $GPU_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Number of sequences: $NUM_SEQUENCES"
echo "Sequence length: $SEQUENCE_LENGTH"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p embedding_thermalizer/logs/evaluation
mkdir -p "$OUTPUT_DIR/videos"
mkdir -p "$OUTPUT_DIR/metrics"
mkdir -p "$OUTPUT_DIR/plots"

echo "Environment setup complete."

cd embedding_thermalizer

# Function to run full evaluation
run_full_evaluation() {
    echo ""
    echo "=== Running Full Evaluation ==="
    echo "This includes generation comparison, metrics computation, and visualization"
    
    EVAL_LOG="logs/evaluation/full_evaluation_$(date +%Y%m%d_%H%M%S).log"
    
    $PYTHON_PATH evaluation/evaluate_with_thermalizer.py \
        --config "$CONFIG_FILE" \
        --mode full \
        --classifier_checkpoint "$CLASSIFIER_CHECKPOINT" \
        --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" \
        --output_dir "../$OUTPUT_DIR" \
        --num_sequences "$NUM_SEQUENCES" \
        --sequence_length "$SEQUENCE_LENGTH" \
        --save_videos \
        --compute_metrics \
        2>&1 | tee "$EVAL_LOG"
    
    return $?
}

# Function to run quick evaluation
run_quick_evaluation() {
    echo ""
    echo "=== Running Quick Evaluation ==="
    echo "This runs a subset of sequences for faster feedback"
    
    EVAL_LOG="logs/evaluation/quick_evaluation_$(date +%Y%m%d_%H%M%S).log"
    QUICK_NUM_SEQUENCES=$((NUM_SEQUENCES / 5))  # Use 1/5 of sequences for quick eval
    QUICK_SEQUENCE_LENGTH=$((SEQUENCE_LENGTH / 2))  # Use shorter sequences
    
    $PYTHON_PATH evaluation/evaluate_with_thermalizer.py \
        --config "$CONFIG_FILE" \
        --mode quick \
        --classifier_checkpoint "$CLASSIFIER_CHECKPOINT" \
        --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" \
        --output_dir "../$OUTPUT_DIR" \
        --num_sequences "$QUICK_NUM_SEQUENCES" \
        --sequence_length "$QUICK_SEQUENCE_LENGTH" \
        --compute_metrics \
        2>&1 | tee "$EVAL_LOG"
    
    return $?
}

# Function to run comparison evaluation
run_comparison_evaluation() {
    echo ""
    echo "=== Running Comparison Evaluation ==="
    echo "This compares generation with and without thermalization"
    
    EVAL_LOG="logs/evaluation/comparison_evaluation_$(date +%Y%m%d_%H%M%S).log"
    
    $PYTHON_PATH evaluation/evaluate_with_thermalizer.py \
        --config "$CONFIG_FILE" \
        --mode comparison \
        --classifier_checkpoint "$CLASSIFIER_CHECKPOINT" \
        --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" \
        --output_dir "../$OUTPUT_DIR" \
        --num_sequences "$NUM_SEQUENCES" \
        --sequence_length "$SEQUENCE_LENGTH" \
        --save_videos \
        --compute_metrics \
        --compare_baseline \
        2>&1 | tee "$EVAL_LOG"
    
    return $?
}

# Function to display system info
display_evaluation_info() {
    echo ""
    echo "=== Evaluation Information ==="
    echo "Date: $(date)"
    echo "Config file: $CONFIG_FILE"
    echo "Classifier checkpoint: $(basename "$CLASSIFIER_CHECKPOINT")"
    echo "Diffusion checkpoint: $(basename "$DIFFUSION_CHECKPOINT")"
    echo "GPU memory available:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | xargs -I {} echo "{}MB"
    else
        echo "nvidia-smi not available"
    fi
}

# Display evaluation information
display_evaluation_info

# Execute evaluation based on mode
case $EVAL_MODE in
    "full")
        run_full_evaluation
        EVAL_SUCCESS=$?
        ;;
    "quick")
        run_quick_evaluation
        EVAL_SUCCESS=$?
        ;;
    "comparison")
        run_comparison_evaluation
        EVAL_SUCCESS=$?
        ;;
esac

cd ..

# Process results and generate summary
if [ $EVAL_SUCCESS -eq 0 ]; then
    echo ""
    echo "=== Evaluation Completed Successfully ==="
    echo "Finished at $(date)"
    
    # Generate evaluation summary
    echo ""
    echo "=== Evaluation Results Summary ==="
    
    # Count generated files
    VIDEO_COUNT=$(find "$OUTPUT_DIR/videos" -name "*.mp4" -o -name "*.gif" 2>/dev/null | wc -l)
    PLOT_COUNT=$(find "$OUTPUT_DIR/plots" -name "*.png" -o -name "*.pdf" 2>/dev/null | wc -l)
    METRIC_COUNT=$(find "$OUTPUT_DIR/metrics" -name "*.json" -o -name "*.csv" 2>/dev/null | wc -l)
    
    echo "Generated files:"
    echo "  - Videos: $VIDEO_COUNT"
    echo "  - Plots: $PLOT_COUNT"
    echo "  - Metric files: $METRIC_COUNT"
    
    # Display key metrics if available
    if [ -f "$OUTPUT_DIR/metrics/summary.json" ]; then
        echo ""
        echo "Key metrics:"
        cat "$OUTPUT_DIR/metrics/summary.json" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'thermalized' in data and 'baseline' in data:
        print(f'  - Baseline MSE: {data[\"baseline\"].get(\"mse\", \"N/A\"):.4f}')
        print(f'  - Thermalized MSE: {data[\"thermalized\"].get(\"mse\", \"N/A\"):.4f}')
        print(f'  - Improvement: {data.get(\"improvement_percent\", \"N/A\")}%')
    else:
        print('  - Summary data structure not recognized')
except Exception as e:
    print(f'  - Error reading summary: {e}')
" 2>/dev/null || echo "  - Summary file found but could not parse"
    fi
    
    # Display output structure
    echo ""
    echo "=== Generated Output Structure ==="
    find "$OUTPUT_DIR" -type f -name "*" | head -20 | sed 's/^/  - /'
    if [ $(find "$OUTPUT_DIR" -type f | wc -l) -gt 20 ]; then
        echo "  - ... and $(expr $(find "$OUTPUT_DIR" -type f | wc -l) - 20) more files"
    fi
    
    echo ""
    echo "=== Next Steps ==="
    echo "1. Review evaluation results in: $OUTPUT_DIR"
    echo "2. Check generated videos for visual quality comparison"
    echo "3. Analyze metrics files for quantitative results"
    echo "4. Review evaluation logs in: embedding_thermalizer/logs/evaluation/"
    
    # Generate final report
    REPORT_FILE="$OUTPUT_DIR/evaluation_report.md"
    cat > "$REPORT_FILE" << EOF
# Embedding Thermalizer Evaluation Report

**Evaluation Date:** $(date)
**Evaluation Mode:** $EVAL_MODE
**Number of Sequences:** $NUM_SEQUENCES
**Sequence Length:** $SEQUENCE_LENGTH

## Model Information
- **Noise Classifier:** $(basename "$CLASSIFIER_CHECKPOINT")
- **Diffusion Model:** $(basename "$DIFFUSION_CHECKPOINT")

## Results Summary
- Videos Generated: $VIDEO_COUNT
- Plots Generated: $PLOT_COUNT
- Metric Files: $METRIC_COUNT

## Output Directory Structure
\`\`\`
$OUTPUT_DIR/
├── videos/     # Generated video comparisons
├── plots/      # Evaluation plots and visualizations
├── metrics/    # Quantitative evaluation results
└── logs/       # Detailed evaluation logs
\`\`\`

## Next Steps
1. Review video outputs for qualitative assessment
2. Analyze metric files for quantitative results
3. Compare baseline vs thermalized generation quality
4. Consider hyperparameter tuning based on results
EOF
    
    echo ""
    echo "📊 Evaluation report generated: $REPORT_FILE"
    echo "✅ Evaluation completed successfully!"
    
else
    echo ""
    echo "❌ Evaluation failed!"
    echo ""
    echo "=== Troubleshooting ==="
    echo "1. Check evaluation logs in embedding_thermalizer/logs/evaluation/"
    echo "2. Verify model checkpoints exist and are valid"
    echo "3. Check GPU memory usage"
    echo "4. Ensure pretrained models are accessible"
    echo ""
    echo "Common fixes:"
    echo "  - Reduce --num-sequences or --sequence-length"
    echo "  - Check checkpoint paths are correct"
    echo "  - Verify CUDA memory with: nvidia-smi"
    echo "  - Try --mode quick for faster debugging"
    
    exit 1
fi 