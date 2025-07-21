"""
Data preparation script for embedding thermalizer.
Extracts clean embeddings from real data and generates drifted embeddings from GPT.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import logging

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.model_loading import load_all_models
from utils.embedding_utils import EmbeddingConverter, create_embedding_corruption
from utils.logging_utils import setup_logging
from dataset.get_datasets import get_dataset


def extract_clean_embeddings(tokenizer, dataloader, num_samples, embed_dim, vqgan_type, device, logger):
    """
    Extract clean embeddings from real data.
    
    Args:
        tokenizer: Loaded tokenizer model
        dataloader: Data loader for real images
        num_samples: Number of clean embeddings to extract
        embed_dim: Embedding dimension
        vqgan_type: Type of tokenizer
        device: Device to use
        logger: Logger instance
    
    Returns:
        clean_embeddings: Tensor of clean embeddings
    """
    logger.info(f"Extracting {num_samples} clean embeddings...")
    
    clean_embeddings_list = []
    total_extracted = 0
    
    converter = EmbeddingConverter(tokenizer, 
                                 tokenizer.quantize.embedding.weight if vqgan_type == 'vqgan' else None,
                                 vqgan_type, device)
    
    tokenizer.to(device)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting clean embeddings")):
            if total_extracted >= num_samples:
                break
            
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                pixel_values = batch[0].to(device)
            else:
                pixel_values = batch.to(device)
            
            batch_size = pixel_values.shape[0]
            
            try:
                if vqgan_type == 'vqgan':
                    # Encode images to get embeddings
                    # pixel_values shape: [B, T, C, H, W] or [B, C, H, W]
                    if pixel_values.dim() == 5:  # Video data
                        B, T, C, H, W = pixel_values.shape
                        pixel_values = pixel_values.view(B * T, C, H, W)
                    
                    # Get tokens first
                    with torch.no_grad():
                        h = tokenizer.encode(pixel_values)  # Get continuous representation
                        _, _, info = tokenizer.quantize(h)  # Quantize to get tokens
                        tokens = info[2]  # Get token indices
                    
                    # Convert tokens to embeddings
                    embeddings = converter.tokens_to_embeddings(tokens)
                    
                    # Flatten spatial dimensions to get individual embeddings
                    embeddings = embeddings.view(-1, embed_dim)
                    
                elif vqgan_type == 'vae':
                    # For VAE, embeddings are continuous
                    if pixel_values.dim() == 5:  # Video data
                        B, T, C, H, W = pixel_values.shape
                        pixel_values = pixel_values.view(B * T, C, H, W)
                    
                    posterior = tokenizer.encode(pixel_values)
                    embeddings = posterior.sample()
                    
                    # Flatten to get individual embeddings
                    embeddings = embeddings.view(-1, embed_dim)
                
                # Take only what we need
                needed = min(embeddings.shape[0], num_samples - total_extracted)
                clean_embeddings_list.append(embeddings[:needed].cpu())
                total_extracted += needed
                
                logger.info(f"Extracted {total_extracted}/{num_samples} clean embeddings")
                
            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue
    
    if len(clean_embeddings_list) == 0:
        raise ValueError("No clean embeddings extracted. Check data loading.")
    
    clean_embeddings = torch.cat(clean_embeddings_list, dim=0)[:num_samples]
    logger.info(f"Successfully extracted {clean_embeddings.shape[0]} clean embeddings")
    
    return clean_embeddings


def generate_drifted_embeddings(gpt_model, tokenizer, converter, num_samples, max_steps, embed_dim, device, logger):
    """
    Generate drifted embeddings using autoregressive GPT model.
    
    Args:
        gpt_model: Loaded GPT model
        tokenizer: Loaded tokenizer model
        converter: EmbeddingConverter instance
        num_samples: Number of drifted sequences to generate
        max_steps: Maximum generation steps per sequence
        embed_dim: Embedding dimension
        device: Device to use
        logger: Logger instance
    
    Returns:
        drifted_embeddings: Tensor of drifted embeddings
        drift_steps: Corresponding timesteps for each embedding
    """
    logger.info(f"Generating {num_samples} drifted embeddings with max {max_steps} steps...")
    
    drifted_embeddings_list = []
    drift_steps_list = []
    
    gpt_model.to(device)
    gpt_model.eval()
    
    # Generate sequences
    for seq_idx in tqdm(range(num_samples), desc="Generating drifted sequences"):
        try:
            # Start with random context tokens
            context_tokens = torch.randint(0, 1024, (1, 64), device=device)  # Assuming 64 tokens per frame
            
            sequence_embeddings = []
            sequence_steps = []
            
            current_tokens = context_tokens
            
            with torch.no_grad():
                for step in range(max_steps):
                    # Generate next tokens
                    if hasattr(gpt_model, 'generate'):
                        next_tokens = gpt_model.generate(current_tokens, max_new_tokens=64)
                        next_tokens = next_tokens[:, -64:]  # Take only the new tokens
                    else:
                        # Alternative generation method
                        logits, _ = gpt_model(current_tokens)
                        next_tokens = torch.multinomial(torch.softmax(logits[:, -1:, :], dim=-1), num_samples=64)
                    
                    # Convert tokens to embeddings
                    embeddings = converter.tokens_to_embeddings(next_tokens)
                    
                    # Flatten and store
                    embeddings_flat = embeddings.view(-1, embed_dim)
                    sequence_embeddings.append(embeddings_flat.cpu())
                    sequence_steps.extend([step + 1] * embeddings_flat.shape[0])
                    
                    # Update context for next step (rolling window)
                    current_tokens = next_tokens
            
            # Collect sequence embeddings
            if sequence_embeddings:
                seq_emb = torch.cat(sequence_embeddings, dim=0)
                drifted_embeddings_list.append(seq_emb)
                drift_steps_list.extend(sequence_steps)
                
                logger.info(f"Generated sequence {seq_idx + 1}/{num_samples} with {seq_emb.shape[0]} embeddings")
        
        except Exception as e:
            logger.warning(f"Error generating sequence {seq_idx}: {e}")
            continue
    
    if len(drifted_embeddings_list) == 0:
        raise ValueError("No drifted embeddings generated. Check GPT model.")
    
    drifted_embeddings = torch.cat(drifted_embeddings_list, dim=0)
    drift_steps = torch.tensor(drift_steps_list, dtype=torch.long)
    
    logger.info(f"Successfully generated {drifted_embeddings.shape[0]} drifted embeddings")
    
    return drifted_embeddings, drift_steps


def create_corrupted_embeddings(clean_embeddings, config, logger):
    """
    Create corrupted versions of clean embeddings for training.
    
    Args:
        clean_embeddings: Tensor of clean embeddings
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        corrupted_embeddings: Tensor of corrupted embeddings
        noise_levels: Corresponding noise levels
    """
    logger.info("Creating corrupted embeddings for training...")
    
    corrupted_embeddings_list = []
    noise_levels_list = []
    
    noise_levels = config['embedding_extraction']['noise_levels']
    corruption_methods = config['embedding_extraction']['corruption_methods']
    
    for noise_level in noise_levels:
        for method in corruption_methods:
            # Take a subset of clean embeddings
            num_per_level = len(clean_embeddings) // (len(noise_levels) * len(corruption_methods))
            subset_embeddings = clean_embeddings[:num_per_level]
            
            # Create corruption
            if method == "gaussian_noise":
                corrupted, _ = create_embedding_corruption(
                    subset_embeddings, 
                    corruption_type="gaussian", 
                    noise_level=noise_level / 1000.0  # Normalize noise level
                )
            elif method == "autoregressive_drift":
                # Simulate cumulative drift
                drift_factor = noise_level / 1000.0
                drift = torch.randn_like(subset_embeddings) * drift_factor
                drift = torch.cumsum(drift, dim=0)  # Cumulative drift
                corrupted = subset_embeddings + drift
            elif method == "distribution_shift":
                # Apply systematic shift
                shift_factor = noise_level / 1000.0
                shift = torch.randn(1, subset_embeddings.shape[-1]) * shift_factor
                corrupted = subset_embeddings + shift.expand_as(subset_embeddings)
            else:
                logger.warning(f"Unknown corruption method: {method}")
                continue
            
            corrupted_embeddings_list.append(corrupted)
            noise_levels_list.extend([noise_level] * corrupted.shape[0])
    
    corrupted_embeddings = torch.cat(corrupted_embeddings_list, dim=0)
    noise_levels = torch.tensor(noise_levels_list, dtype=torch.long)
    
    logger.info(f"Created {corrupted_embeddings.shape[0]} corrupted embeddings")
    
    return corrupted_embeddings, noise_levels


def main():
    """Main data preparation function."""
    # Load configuration
    config_path = "/home/cmeo/Cristian/models/WeatherForecast/ChroneCast/embedding_thermalizer/configs/experiment_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Setup logging
    logger = setup_logging({'log_dir': '/home/cmeo/Cristian/models/WeatherForecast/ChroneCast/embedding_thermalizer/logs'})
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create output directory
    save_path = Path(config['embedding_extraction']['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load models
        logger.info("Loading pretrained models...")
        models = load_all_models(config)
        
        # Create embedding converter
        converter = EmbeddingConverter(
            models['tokenizer'],
            models.get('codebook'),
            config['pretrained_models']['vqgan_type'],
            device
        )
        
        # Setup data loader for clean embeddings
        logger.info("Setting up data loader...")
        train_data, valid_data, test_data, _, _, _ = get_dataset(
            data_name=config['data']['dataset_name'],
            img_size=config['data']['resolution'],
            seq_len=config['data']['segment_length'],
            temp_res_sevir=5,  # Default value
            batch_size=config['data']['batch_size'],
            debug=config['data']['debug']
        )
        
        if hasattr(train_data, 'get_torch_dataloader'):
            train_dataloader = train_data.get_torch_dataloader(num_workers=config['data']['num_workers'])
        else:
            train_dataloader = torch.utils.data.DataLoader(
                train_data, 
                batch_size=config['data']['batch_size'], 
                shuffle=False, 
                num_workers=config['data']['num_workers']
            )
        
        # Extract clean embeddings
        clean_embeddings = extract_clean_embeddings(
            models['tokenizer'],
            train_dataloader,
            config['embedding_extraction']['num_clean_samples'],
            models['embed_dim'],
            config['pretrained_models']['vqgan_type'],
            device,
            logger
        )
        
        # Generate drifted embeddings (if GPT model is available)
        if models['gpt'] is not None:
            drifted_embeddings, drift_steps = generate_drifted_embeddings(
                models['gpt'],
                models['tokenizer'],
                converter,
                config['embedding_extraction']['num_drift_samples'],
                config['embedding_extraction']['max_generation_steps'],
                models['embed_dim'],
                device,
                logger
            )
        else:
            logger.warning("No GPT model available, skipping drift generation")
            drifted_embeddings, drift_steps = None, None
        
        # Create corrupted embeddings
        corrupted_embeddings, noise_levels = create_corrupted_embeddings(
            clean_embeddings,
            config,
            logger
        )
        
        # Save all data
        logger.info(f"Saving data to {save_path}")
        
        torch.save(clean_embeddings, save_path / "clean_embeddings.pt")
        torch.save(corrupted_embeddings, save_path / "corrupted_embeddings.pt")
        torch.save(noise_levels, save_path / "noise_levels.pt")
        
        if drifted_embeddings is not None:
            torch.save(drifted_embeddings, save_path / "drifted_embeddings.pt")
            torch.save(drift_steps, save_path / "drift_steps.pt")
        
        # Save metadata
        metadata = {
            'num_clean_samples': clean_embeddings.shape[0],
            'num_corrupted_samples': corrupted_embeddings.shape[0],
            'num_drifted_samples': drifted_embeddings.shape[0] if drifted_embeddings is not None else 0,
            'embed_dim': models['embed_dim'],
            'vqgan_type': config['pretrained_models']['vqgan_type'],
            'config': config
        }
        
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Data preparation completed successfully!")
        logger.info(f"Clean embeddings: {clean_embeddings.shape}")
        logger.info(f"Corrupted embeddings: {corrupted_embeddings.shape}")
        if drifted_embeddings is not None:
            logger.info(f"Drifted embeddings: {drifted_embeddings.shape}")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise


if __name__ == "__main__":
    main() 