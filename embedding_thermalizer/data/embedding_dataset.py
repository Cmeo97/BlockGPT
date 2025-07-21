"""
Dataset classes for embedding thermalizer training.
Handles clean, corrupted, and drifted embeddings.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path


class EmbeddingNoiseDataset(Dataset):
    """
    Dataset for training the noise classifier.
    Returns corrupted embeddings and their corresponding noise levels.
    """
    
    def __init__(self, corrupted_embeddings_path, noise_levels_path, transform=None):
        """
        Initialize noise classifier dataset.
        
        Args:
            corrupted_embeddings_path: Path to corrupted embeddings tensor
            noise_levels_path: Path to noise levels tensor
            transform: Optional transform to apply to embeddings
        """
        self.corrupted_embeddings = torch.load(corrupted_embeddings_path)
        self.noise_levels = torch.load(noise_levels_path)
        self.transform = transform
        
        assert len(self.corrupted_embeddings) == len(self.noise_levels), \
            "Corrupted embeddings and noise levels must have same length"
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded {len(self.corrupted_embeddings)} corrupted embeddings")
    
    def __len__(self):
        return len(self.corrupted_embeddings)
    
    def __getitem__(self, idx):
        embedding = self.corrupted_embeddings[idx]
        noise_level = self.noise_levels[idx]
        
        if self.transform:
            embedding = self.transform(embedding)
        
        return embedding.float(), noise_level.float()


class EmbeddingDiffusionDataset(Dataset):
    """
    Dataset for training the diffusion model.
    Returns clean embeddings for diffusion training.
    """
    
    def __init__(self, clean_embeddings_path, transform=None):
        """
        Initialize diffusion dataset.
        
        Args:
            clean_embeddings_path: Path to clean embeddings tensor
            transform: Optional transform to apply to embeddings
        """
        self.clean_embeddings = torch.load(clean_embeddings_path)
        self.transform = transform
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded {len(self.clean_embeddings)} clean embeddings")
    
    def __len__(self):
        return len(self.clean_embeddings)
    
    def __getitem__(self, idx):
        embedding = self.clean_embeddings[idx]
        
        if self.transform:
            embedding = self.transform(embedding)
        
        return embedding.float()


class CombinedEmbeddingDataset(Dataset):
    """
    Combined dataset with clean, corrupted, and drifted embeddings.
    Useful for comprehensive evaluation.
    """
    
    def __init__(self, data_dir, include_drifted=True, transform=None):
        """
        Initialize combined dataset.
        
        Args:
            data_dir: Directory containing all embedding data
            include_drifted: Whether to include drifted embeddings
            transform: Optional transform to apply to embeddings
        """
        data_dir = Path(data_dir)
        
        # Load clean embeddings
        self.clean_embeddings = torch.load(data_dir / "clean_embeddings.pt")
        
        # Load corrupted embeddings and noise levels
        self.corrupted_embeddings = torch.load(data_dir / "corrupted_embeddings.pt")
        self.noise_levels = torch.load(data_dir / "noise_levels.pt")
        
        # Load drifted embeddings if available
        self.drifted_embeddings = None
        self.drift_steps = None
        if include_drifted:
            try:
                self.drifted_embeddings = torch.load(data_dir / "drifted_embeddings.pt")
                self.drift_steps = torch.load(data_dir / "drift_steps.pt")
            except FileNotFoundError:
                self.logger.warning("Drifted embeddings not found, skipping")
        
        self.transform = transform
        
        # Create labels: 0=clean, 1=corrupted, 2=drifted
        self.labels = []
        self.embeddings = []
        self.metadata = []
        
        # Add clean embeddings
        self.embeddings.append(self.clean_embeddings)
        self.labels.extend([0] * len(self.clean_embeddings))
        self.metadata.extend([{"type": "clean", "noise_level": 0}] * len(self.clean_embeddings))
        
        # Add corrupted embeddings
        self.embeddings.append(self.corrupted_embeddings)
        self.labels.extend([1] * len(self.corrupted_embeddings))
        for noise_level in self.noise_levels:
            self.metadata.append({"type": "corrupted", "noise_level": noise_level.item()})
        
        # Add drifted embeddings
        if self.drifted_embeddings is not None:
            self.embeddings.append(self.drifted_embeddings)
            self.labels.extend([2] * len(self.drifted_embeddings))
            for drift_step in self.drift_steps:
                self.metadata.append({"type": "drifted", "drift_step": drift_step.item()})
        
        # Concatenate all embeddings
        self.all_embeddings = torch.cat(self.embeddings, dim=0)
        self.all_labels = torch.tensor(self.labels)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded combined dataset with {len(self.all_embeddings)} embeddings")
    
    def __len__(self):
        return len(self.all_embeddings)
    
    def __getitem__(self, idx):
        embedding = self.all_embeddings[idx]
        label = self.all_labels[idx]
        metadata = self.metadata[idx]
        
        if self.transform:
            embedding = self.transform(embedding)
        
        return embedding.float(), label, metadata


class EmbeddingTransforms:
    """
    Transform functions for embedding data augmentation.
    """
    
    @staticmethod
    def normalize(embedding):
        """Normalize embedding to unit norm."""
        return embedding / (torch.norm(embedding, dim=-1, keepdim=True) + 1e-8)
    
    @staticmethod
    def add_noise(embedding, noise_std=0.01):
        """Add Gaussian noise to embedding."""
        noise = torch.randn_like(embedding) * noise_std
        return embedding + noise
    
    @staticmethod
    def scale(embedding, scale_range=(0.9, 1.1)):
        """Scale embedding by random factor."""
        scale_factor = torch.rand(1) * (scale_range[1] - scale_range[0]) + scale_range[0]
        return embedding * scale_factor
    
    @staticmethod
    def dropout(embedding, dropout_rate=0.1):
        """Apply dropout to embedding."""
        mask = torch.rand_like(embedding) > dropout_rate
        return embedding * mask


def create_dataloaders(config, mode="noise_classifier"):
    """
    Create data loaders for different training modes.
    
    Args:
        config: Configuration dictionary
        mode: Training mode ("noise_classifier", "diffusion", or "combined")
    
    Returns:
        train_loader, val_loader: Training and validation data loaders
    """
    data_dir = Path(config['embedding_extraction']['save_path'])
    
    if mode == "noise_classifier":
        # Create noise classifier dataset
        dataset = EmbeddingNoiseDataset(
            data_dir / "corrupted_embeddings.pt",
            data_dir / "noise_levels.pt"
        )
        
        # Split into train/val
        train_size = int(config['training']['classifier']['train_split'] * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['classifier']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['classifier']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
    
    elif mode == "diffusion":
        # Create diffusion dataset
        dataset = EmbeddingDiffusionDataset(
            data_dir / "clean_embeddings.pt"
        )
        
        # Split into train/val
        train_size = int(config['training']['diffusion']['train_split'] * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['diffusion']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['diffusion']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
    
    elif mode == "combined":
        # Create combined dataset for evaluation
        dataset = CombinedEmbeddingDataset(data_dir)
        
        # For evaluation, we might not need train/val split
        data_loader = DataLoader(
            dataset,
            batch_size=config['evaluation']['batch_size'] if 'evaluation' in config else 32,
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        return data_loader, None
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return train_loader, val_loader


def analyze_embedding_statistics(data_dir):
    """
    Analyze statistics of embedding datasets.
    
    Args:
        data_dir: Directory containing embedding data
    
    Returns:
        stats: Dictionary of statistics
    """
    data_dir = Path(data_dir)
    
    stats = {}
    
    # Load and analyze clean embeddings
    clean_embeddings = torch.load(data_dir / "clean_embeddings.pt")
    stats['clean'] = {
        'count': len(clean_embeddings),
        'mean': clean_embeddings.mean().item(),
        'std': clean_embeddings.std().item(),
        'min': clean_embeddings.min().item(),
        'max': clean_embeddings.max().item(),
        'norm_mean': torch.norm(clean_embeddings, dim=-1).mean().item()
    }
    
    # Load and analyze corrupted embeddings
    corrupted_embeddings = torch.load(data_dir / "corrupted_embeddings.pt")
    noise_levels = torch.load(data_dir / "noise_levels.pt")
    
    stats['corrupted'] = {
        'count': len(corrupted_embeddings),
        'mean': corrupted_embeddings.mean().item(),
        'std': corrupted_embeddings.std().item(),
        'min': corrupted_embeddings.min().item(),
        'max': corrupted_embeddings.max().item(),
        'norm_mean': torch.norm(corrupted_embeddings, dim=-1).mean().item(),
        'noise_levels_unique': torch.unique(noise_levels).tolist()
    }
    
    # Load and analyze drifted embeddings if available
    try:
        drifted_embeddings = torch.load(data_dir / "drifted_embeddings.pt")
        drift_steps = torch.load(data_dir / "drift_steps.pt")
        
        stats['drifted'] = {
            'count': len(drifted_embeddings),
            'mean': drifted_embeddings.mean().item(),
            'std': drifted_embeddings.std().item(),
            'min': drifted_embeddings.min().item(),
            'max': drifted_embeddings.max().item(),
            'norm_mean': torch.norm(drifted_embeddings, dim=-1).mean().item(),
            'drift_steps_unique': torch.unique(drift_steps).tolist()
        }
    except FileNotFoundError:
        stats['drifted'] = None
    
    return stats


if __name__ == "__main__":
    # Test dataset creation
    import json
    from pathlib import Path
    
    # Load test config
    config_path = Path("embedding_thermalizer/configs/experiment_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Test noise classifier dataset
        try:
            train_loader, val_loader = create_dataloaders(config, mode="noise_classifier")
            print(f"Noise classifier - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            
            # Test one batch
            for embeddings, noise_levels in train_loader:
                print(f"Batch - Embeddings: {embeddings.shape}, Noise levels: {noise_levels.shape}")
                break
                
        except FileNotFoundError:
            print("Embedding data not found. Run data preparation first.")
    
    else:
        print("Config file not found. Please create the experiment configuration first.") 