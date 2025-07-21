"""
Logging utilities for the embedding thermalizer experiment.
Provides comprehensive logging, visualization, and progress tracking.
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm

# TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


class Logger:
    """
    Comprehensive logger for the embedding thermalizer experiment.
    Supports both Weights & Biases and TensorBoard logging.
    """
    
    def __init__(self, config, experiment_name=None):
        """
        Initialize logger with experiment configuration.
        
        Args:
            config: Experiment configuration dictionary
            experiment_name: Optional experiment name override
        """
        self.config = config
        self.experiment_name = experiment_name or config.get('experiment_name', 'embedding_thermalizer')
        
        # Setup paths from config
        self._setup_paths()
        
        # Setup file logging
        self._setup_file_logging()
        
        # Setup TensorBoard if available
        self.tb_writer = None
        if config['logging'].get('use_tensorboard', True) and TENSORBOARD_AVAILABLE:
            self._setup_tensorboard()
        
        # Setup wandb if enabled
        self.use_wandb = config['logging'].get('use_wandb', False)
        if self.use_wandb:
            self._setup_wandb()
        
        # Initialize metrics tracking
        self.metrics = {}
        self.step_count = 0
        
        # Logger is already set up in _setup_file_logging()
        self.logger.info(f"Initialized logger for experiment: {self.experiment_name}")
        
        # Log directory information
        if hasattr(self, '_directory_info'):
            self.logger.info("Created directories:")
            for name, path in self._directory_info.items():
                self.logger.info(f"  - {name}: {path}")
        
        # Log configuration
        self._log_system_info()
    
    def _setup_paths(self):
        """Setup all paths from configuration."""
        # Get base paths from config
        base_log_dir = self.config['logging'].get('log_dir', 'embedding_thermalizer/logs')
        base_save_dir = self.config['logging'].get('save_dir', 'embedding_thermalizer/checkpoints')
        
        # Create full paths
        self.log_dir = Path(base_log_dir)
        self.save_dir = Path(base_save_dir)
        self.tensorboard_dir = self.log_dir / 'tensorboard' / self.experiment_name
        self.plots_dir = self.log_dir / 'plots' / self.experiment_name
        self.results_dir = Path(self.config['logging'].get('results_dir', 'Results/embedding_thermalizer'))
        
        # Create directories
        for dir_path in [self.log_dir, self.save_dir, self.tensorboard_dir, self.plots_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Store directory info to log later (after logger is set up)
        self._directory_info = {
            "log_dir": self.log_dir,
            "save_dir": self.save_dir,
            "tensorboard_dir": self.tensorboard_dir,
            "results_dir": self.results_dir
        }
    
    def _setup_file_logging(self):
        """Setup file-based logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{self.experiment_name}_{timestamp}.log"
        
        # Create logger instance
        self.logger = logging.getLogger(f"embedding_thermalizer.{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Store references to handlers for later flushing
        self.file_handler = file_handler
        self.console_handler = console_handler
        
        # Save config to log directory
        config_file = self.log_dir / f"{self.experiment_name}_{timestamp}_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        try:
            self.tb_writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
            self.logger.info(f"TensorBoard initialized: {self.tensorboard_dir}")
            
            # Log configuration as text
            config_str = json.dumps(self.config, indent=2)
            self.tb_writer.add_text("Configuration", config_str, 0)
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.tb_writer = None
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            wandb.init(
                project=self.config['logging']['project'],
                entity=self.config['logging'].get('entity'),
                name=self.experiment_name,
                config=self.config,
                dir=str(self.log_dir)
            )
            self.logger.info("Wandb initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def _log_system_info(self):
        """Log system information."""
        system_info = {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            system_info["cuda_version"] = torch.version.cuda
            system_info["gpu_name"] = torch.cuda.get_device_name()
        
        self.logger.info(f"System info: {system_info}")
        
        if self.tb_writer:
            for key, value in system_info.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"System/{key}", value, 0)
                else:
                    self.tb_writer.add_text(f"System/{key}", str(value), 0)
    
    def log_step(self, metrics, step=None, prefix=""):
        """
        Log metrics for a single step.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (auto-increment if None)
            prefix: Prefix for metric names
        """
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Store metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Log to TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float, torch.Tensor)):
                    if isinstance(value, torch.Tensor):
                        value = value.item() if value.numel() == 1 else value.mean().item()
                    self.tb_writer.add_scalar(key, value, step)
        
        # Log to wandb if available
        if self.use_wandb and wandb.run is not None:
            wandb.log(metrics, step=step)
        
        # Log key metrics to console
        metric_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items() if isinstance(v, (int, float))])
        self.logger.info(f"Step {step} - {metric_str}")
    
    def log_model_info(self, model, model_name="model"):
        """Log model architecture and parameter information."""
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            f"{model_name}/total_params": num_params,
            f"{model_name}/trainable_params": num_trainable
        }
        
        self.logger.info(f"{model_name} - Total parameters: {num_params:,}")
        self.logger.info(f"{model_name} - Trainable parameters: {num_trainable:,}")
        
        # Log to both TensorBoard and wandb
        if self.tb_writer:
            for key, value in model_info.items():
                self.tb_writer.add_scalar(key, value, 0)
            
            # Log model architecture as text
            model_str = str(model)
            self.tb_writer.add_text(f"{model_name}/architecture", model_str, 0)
        
        if self.use_wandb and wandb.run is not None:
            wandb.log(model_info)
    
    def log_embeddings_info(self, embeddings, name="embeddings", step=None):
        """Log statistics about embeddings."""
        stats = {
            f"{name}/mean": embeddings.mean().item(),
            f"{name}/std": embeddings.std().item(),
            f"{name}/min": embeddings.min().item(),
            f"{name}/max": embeddings.max().item(),
            f"{name}/norm": torch.norm(embeddings).item()
        }
        
        self.log_step(stats, step)
        
        # Log histogram to TensorBoard
        if self.tb_writer and step is not None:
            self.tb_writer.add_histogram(f"{name}/values", embeddings, step)
        
        # Log histogram to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.log({f"{name}/histogram": wandb.Histogram(embeddings.cpu().numpy())}, step=step)
    
    def log_corruption_analysis(self, clean_embeddings, corrupted_embeddings, noise_levels, step=None):
        """Log analysis of embedding corruption."""
        mse_loss = torch.nn.functional.mse_loss(clean_embeddings, corrupted_embeddings)
        cosine_sim = torch.nn.functional.cosine_similarity(
            clean_embeddings.flatten(1), 
            corrupted_embeddings.flatten(1), 
            dim=1
        ).mean()
        
        stats = {
            "corruption/mse_loss": mse_loss.item(),
            "corruption/cosine_similarity": cosine_sim.item(),
            "corruption/mean_noise_level": noise_levels.float().mean().item(),
            "corruption/max_noise_level": noise_levels.float().max().item()
        }
        
        self.log_step(stats, step)
    
    def log_generation_metrics(self, generated_tokens, clean_tokens, step=None):
        """Log metrics comparing generated vs clean tokens."""
        # Token accuracy
        accuracy = (generated_tokens == clean_tokens).float().mean().item()
        
        # Token distribution comparison
        gen_unique = torch.unique(generated_tokens).numel()
        clean_unique = torch.unique(clean_tokens).numel()
        
        stats = {
            "generation/token_accuracy": accuracy,
            "generation/generated_unique_tokens": gen_unique,
            "generation/clean_unique_tokens": clean_unique,
            "generation/token_diversity_ratio": gen_unique / clean_unique if clean_unique > 0 else 0
        }
        
        self.log_step(stats, step)
    
    def log_learning_rate(self, optimizer, step):
        """Log current learning rate."""
        lr = optimizer.param_groups[0]['lr']
        self.log_step({"learning_rate": lr}, step)
    
    def log_gradient_norms(self, model, step, prefix=""):
        """Log gradient norms for monitoring training stability."""
        total_norm = 0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        gradient_stats = {
            f"{prefix}gradient_norm/total": total_norm,
            f"{prefix}gradient_norm/param_count": param_count
        }
        
        self.log_step(gradient_stats, step)
    
    def plot_loss_curves(self, save=True, show=False):
        """Plot training loss curves and save to configured directory."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Training Metrics - {self.experiment_name}')
        
        # Training losses
        if 'train/loss' in self.metrics:
            axes[0, 0].plot(self.metrics['train/loss'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
        
        # Validation losses
        if 'val/loss' in self.metrics:
            axes[0, 1].plot(self.metrics['val/loss'])
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
        
        # Noise level predictions
        if 'train/noise_accuracy' in self.metrics:
            axes[1, 0].plot(self.metrics['train/noise_accuracy'])
            axes[1, 0].set_title('Noise Level Prediction Accuracy')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Accuracy')
        
        # Thermalization metrics
        if 'thermalizer/correction_rate' in self.metrics:
            axes[1, 1].plot(self.metrics['thermalizer/correction_rate'])
            axes[1, 1].set_title('Thermalization Correction Rate')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Rate')
        
        plt.tight_layout()
        
        if save:
            plot_path = self.plots_dir / f"{self.experiment_name}_losses.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Loss curves saved: {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_figure("Plots/loss_curves", fig, self.step_count)
        
        # Log to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.log({"plots/loss_curves": wandb.Image(fig)})
    
    def plot_embedding_evolution(self, embeddings_history, save=True, show=False):
        """Plot evolution of embeddings during training/generation."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Embedding Evolution - {self.experiment_name}')
        
        # Extract statistics over time
        means = [emb.mean().item() for emb in embeddings_history]
        stds = [emb.std().item() for emb in embeddings_history]
        norms = [torch.norm(emb).item() for emb in embeddings_history]
        
        axes[0, 0].plot(means)
        axes[0, 0].set_title('Embedding Mean')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Mean')
        
        axes[0, 1].plot(stds)
        axes[0, 1].set_title('Embedding Std')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Std')
        
        axes[1, 0].plot(norms)
        axes[1, 0].set_title('Embedding Norm')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Norm')
        
        # PCA visualization of last embeddings
        if len(embeddings_history) > 0:
            last_emb = embeddings_history[-1]
            if last_emb.dim() > 2:
                last_emb = last_emb.view(-1, last_emb.shape[-1])
            
            # Simple PCA approximation
            U, S, V = torch.pca_lowrank(last_emb, q=2)
            pca_emb = torch.matmul(last_emb, V[:, :2])
            
            axes[1, 1].scatter(pca_emb[:, 0].cpu(), pca_emb[:, 1].cpu(), alpha=0.5)
            axes[1, 1].set_title('Embedding PCA (Last Step)')
            axes[1, 1].set_xlabel('PC1')
            axes[1, 1].set_ylabel('PC2')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.log_dir / f"{self.experiment_name}_embedding_evolution.png", dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        # Log to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.log({"plots/embedding_evolution": wandb.Image(fig)})
    
    def save_checkpoint(self, model, optimizer, epoch, step, additional_info=None, filename=None):
        """Save model checkpoint to configured directory."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        if filename is None:
            filename = f"{self.experiment_name}_checkpoint_step_{step}.pt"
        
        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def save_model_only(self, model, filename):
        """Save only the model state dict."""
        model_path = self.save_dir / filename
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"Model saved: {model_path}")
        return model_path
    
    def create_progress_bar(self, total, desc="Training"):
        """Create a progress bar with logging integration."""
        return tqdm(total=total, desc=f"{self.experiment_name} - {desc}")
    
    def flush(self):
        """Force flush all logging handlers to ensure messages are written."""
        # Flush the specific handlers we created
        if hasattr(self, 'file_handler'):
            self.file_handler.flush()
        if hasattr(self, 'console_handler'):
            self.console_handler.flush()
        
        # Flush all handlers for the logger as backup
        for handler in self.logger.handlers:
            handler.flush()
    
    def finish(self):
        """Finish logging and cleanup."""
        self.plot_loss_curves()
        
        # Flush all handlers before closing
        self.flush()
        
        # Close TensorBoard writer
        if self.tb_writer:
            self.tb_writer.close()
            self.logger.info("TensorBoard writer closed")
        
        # Close wandb
        if self.use_wandb and wandb.run is not None:
            wandb.finish()
        
        self.logger.info(f"Experiment {self.experiment_name} completed")
        self.logger.info(f"Results saved to: {self.results_dir}")
        
        # Final flush
        self.flush()


class MetricsTracker:
    """Simple metrics tracker for validation and testing."""
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_averages(self):
        """Get average values for all metrics."""
        return {key: self.metrics[key] / self.counts[key] for key in self.metrics}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counts.clear()


def setup_logging(config):
    """
    Setup logging for a simple script without the full Logger class.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configured logger
    """
    # Create log directory
    log_dir = Path(config.get('log_dir', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")
    
    return logger


if __name__ == "__main__":
    # Test logging setup
    config = {
        'experiment_name': 'test_experiment',
        'logging': {
            'log_dir': 'test_logs',
            'save_dir': 'test_checkpoints',
            'use_wandb': False,
            'project': 'test_project'
        }
    }
    
    logger = Logger(config)
    
    # Test logging
    logger.log_step({'loss': 1.5, 'accuracy': 0.8}, step=0)
    logger.log_step({'loss': 1.2, 'accuracy': 0.85}, step=1)
    
    # Test plotting
    logger.plot_loss_curves(save=True, show=False)
    
    logger.finish()
    print("Logging test completed") 