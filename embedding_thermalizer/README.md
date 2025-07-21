# Embedding-Space Thermalizer

This project implements a thermalizer that operates in the embedding space to reduce autoregression drift in GPT-generated sequences.

## Pipeline Overview

```
Images -> Tokenizer.encode() -> Tokens_t -> GPT Model -> Tokens_{t+1} 
    -> Embedding Lookup -> Raw_Embeddings_{t+1} -> THERMALIZER -> Corrected_Embeddings_{t+1} 
    -> Re-Quantization -> Corrected_Tokens_{t+1} -> Next Iteration
```

## Project Structure

```
embedding_thermalizer/
├── configs/
│   ├── embedding_noise_classifier.json     # Noise classifier config
│   ├── embedding_diffusion.json            # Diffusion model config
│   └── experiment_config.json              # Main experiment config
├── data/
│   ├── prepare_embedding_data.py           # Extract clean/drifted embeddings
│   └── embedding_dataset.py                # Dataset classes
├── models/
│   ├── embedding_noise_classifier.py       # Noise level predictor
│   ├── embedding_diffusion.py              # Diffusion model for embeddings
│   └── embedding_thermalizer.py            # Main thermalizer wrapper
├── training/
│   ├── train_embedding_thermalizer.py      # Training script
│   └── embedding_trainer.py                # Training logic
├── evaluation/
│   ├── evaluate_with_thermalizer.py        # Evaluation with thermalization
│   └── embedding_metrics.py                # Evaluation metrics
├── scripts/
│   ├── run_data_preparation.sh             # Data prep bash script
│   ├── run_training.sh                     # Training bash script
│   └── run_evaluation.sh                   # Evaluation bash script
└── utils/
    ├── model_loading.py                    # Model loading utilities
    ├── embedding_utils.py                  # Embedding conversion utilities
    └── logging_utils.py                    # Logging utilities
```

## Quick Start

1. **Prepare embedding data:**
   ```bash
   bash scripts/run_data_preparation.sh
   ```

2. **Train thermalizer:**
   ```bash
   bash scripts/run_training.sh
   ```

3. **Evaluate with thermalization:**
   ```bash
   bash scripts/run_evaluation.sh
   ```

## Configuration

Edit `configs/experiment_config.json` to set paths to pretrained models, datasets, and hyperparameters.

## Requirements

- PyTorch
- Existing pretrained GPT model
- Existing pretrained VQGAN/VAE tokenizer
- Thermalizer package (installed in development mode) 