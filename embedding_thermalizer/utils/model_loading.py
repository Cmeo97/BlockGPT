"""
Model loading utilities for embedding thermalizer experiment.
Adapted from the main train_gpt.py file.
"""

import json
import torch
import logging
import sys
import os

# Add parent directory to path to import from main codebase
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.blockGPT.model import GPTConfig, GPT, ContinuousGPTConfig, ContinuousGPT
from models.taming.vqgan import get_model as get_vqgan_model
from models.taming.vae import get_model as get_vae_model
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def load_gpt_model(config_path, checkpoint_path=None, predictor_name="blockGPT"):
    """
    Load GPT model from configuration and optional checkpoint.
    
    Args:
        config_path: Path to GPT config JSON file
        checkpoint_path: Path to GPT checkpoint (optional)
        predictor_name: Type of GPT model ("blockGPT" or "continuousGPT")
    
    Returns:
        Loaded GPT model
    """
    logger.info(f"Loading GPT model from config: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if predictor_name == 'blockGPT':
        config_gpt = GPTConfig(**config)
        model = GPT(config_gpt)
    elif predictor_name == 'continuousGPT':
        config_gpt = ContinuousGPTConfig(**config)
        model = ContinuousGPT(config_gpt)
    else:
        raise ValueError(f"Unknown predictor_name: {predictor_name}")
    
    if checkpoint_path is not None:
        logger.info(f"Loading GPT checkpoint from: {checkpoint_path}")
        if checkpoint_path.endswith('.safetensors'):
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
        
        model.load_state_dict(state_dict, strict=True)
        logger.info("GPT checkpoint loaded successfully")
    
    model.eval()
    return model


def load_tokenizer_model(tokenizer_config_path, checkpoint_path, vqgan_type="vqgan"):
    """
    Load VQGAN/VAE tokenizer model.
    
    Args:
        tokenizer_config_path: Path to tokenizer config JSON file
        checkpoint_path: Path to tokenizer checkpoint
        vqgan_type: Type of tokenizer ("vqgan" or "vae")
    
    Returns:
        Loaded tokenizer model and vocab size
    """
    logger.info(f"Loading {vqgan_type} tokenizer from config: {tokenizer_config_path}")
    
    with open(tokenizer_config_path, 'r') as f:
        config = json.load(f)
    
    # Load checkpoint data
    data = torch.load(checkpoint_path, map_location='cpu')
    
    if vqgan_type == 'vqgan':
        model = get_vqgan_model(**config)
        vocab_size = model.quantize.n_e
    elif vqgan_type == 'vae':
        model = get_vae_model(config)
        vocab_size = None
    else:
        raise ValueError(f"Unknown vqgan_type: {vqgan_type}")
    
    # Load state dict
    if 'model' in data:
        model.load_state_dict(data['model'])
    else:
        model.load_state_dict(data)
    
    model.eval()
    logger.info(f"Tokenizer loaded successfully with vocab_size: {vocab_size}")
    
    return model, vocab_size


def get_embedding_dim(tokenizer, vqgan_type="vqgan"):
    """
    Get the embedding dimension from the tokenizer.
    
    Args:
        tokenizer: Loaded tokenizer model
        vqgan_type: Type of tokenizer ("vqgan" or "vae")
    
    Returns:
        Embedding dimension
    """
    if vqgan_type == 'vqgan':
        return tokenizer.quantize.e_dim
    elif vqgan_type == 'vae':
        return tokenizer.embed_dim
    else:
        raise ValueError(f"Unknown vqgan_type: {vqgan_type}")


def get_codebook(tokenizer, vqgan_type="vqgan"):
    """
    Get the codebook from the tokenizer.
    
    Args:
        tokenizer: Loaded tokenizer model
        vqgan_type: Type of tokenizer ("vqgan" or "vae")
    
    Returns:
        Codebook tensor
    """
    if vqgan_type == 'vqgan':
        return tokenizer.quantize.embedding.weight
    else:
        raise ValueError("Codebook only available for VQGAN tokenizers")


def load_all_models(config):
    """
    Load all required models based on experiment configuration.
    
    Args:
        config: Experiment configuration dictionary
    
    Returns:
        Dictionary containing all loaded models
    """
    models = {}
    
    # Load GPT model
    models['gpt'] = load_gpt_model(
        config['pretrained_models']['gpt_config'],
        config['pretrained_models'].get('gpt_checkpoint'),
        predictor_name="blockGPT"  # Assuming blockGPT for now
    )
    
    # Load tokenizer
    models['tokenizer'], models['vocab_size'] = load_tokenizer_model(
        config['pretrained_models']['tokenizer_config'],
        config['pretrained_models']['tokenizer_checkpoint'],
        config['pretrained_models']['vqgan_type']
    )
    
    # Get embedding properties
    models['embed_dim'] = get_embedding_dim(
        models['tokenizer'],
        config['pretrained_models']['vqgan_type']
    )
    
    if config['pretrained_models']['vqgan_type'] == 'vqgan':
        models['codebook'] = get_codebook(
            models['tokenizer'],
            config['pretrained_models']['vqgan_type']
        )
    
    logger.info("All models loaded successfully")
    return models


if __name__ == "__main__":
    # Test model loading
    config = {
        'pretrained_models': {
            'gpt_config': '../configs/GPT/config_blockGPT_KNMI30.json',
            'gpt_checkpoint': None,
            'tokenizer_config': '../configs/Encoders/config_vqgan.json',
            'tokenizer_checkpoint': 'path/to/tokenizer/checkpoint.pt',
            'vqgan_type': 'vqgan'
        }
    }
    
    models = load_all_models(config)
    print(f"Loaded models: {list(models.keys())}")
    print(f"Embedding dimension: {models['embed_dim']}") 