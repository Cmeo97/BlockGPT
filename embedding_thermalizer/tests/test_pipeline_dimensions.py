"""
Test script to validate pipeline dimensions and forward/backward propagation.
This script ensures all components of the embedding thermalizer work together correctly.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import warnings

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.model_loading import load_all_models, load_gpt_model, load_tokenizer_model
from utils.embedding_utils import EmbeddingConverter, create_embedding_corruption
from utils.logging_utils import Logger


class TestPipelineDimensions(unittest.TestCase):
    """Test class for validating pipeline dimensions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.config_path = "embedding_thermalizer/configs/experiment_config.json"
        cls.test_config_path = "embedding_thermalizer/tests/test_config.json"
        
        # Create test configuration
        cls._create_test_config()
        
        # Set device
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {cls.device}")
        
        # Mock models if needed
        cls.use_mock_models = True
        
    @classmethod
    def _create_test_config(cls):
        """Create a test configuration file."""
        test_config = {
            "experiment_name": "test_pipeline",
            "seed": 42,
            
            "pretrained_models": {
                "gpt_config": "configs/GPT/config_blockGPT_KNMI30.json",
                "gpt_checkpoint": None,
                "tokenizer_config": "configs/Encoders/config_vqgan.json", 
                "tokenizer_checkpoint": None,
                "vqgan_type": "vqgan"
            },
            
            "noise_classifier": {
                "embed_dim": 256,
                "hidden_dims": [512, 256, 128],
                "dropout": 0.1
            },
            
            "diffusion_model": {
                "embed_dim": 256,
                "model_channels": 128,
                "timesteps": 100  # Smaller for testing
            },
            
            "logging": {
                "use_wandb": False,
                "use_tensorboard": False,
                "log_dir": "embedding_thermalizer/tests/logs",
                "save_dir": "embedding_thermalizer/tests/checkpoints",
                "results_dir": "embedding_thermalizer/tests/results"
            }
        }
        
        # Create test directory and save config
        test_dir = Path("embedding_thermalizer/tests")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        with open(cls.test_config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
    
    def test_config_loading(self):
        """Test that configuration can be loaded correctly."""
        print("\n=== Testing Configuration Loading ===")
        
        with open(self.test_config_path, 'r') as f:
            config = json.load(f)
        
        # Check required keys
        required_keys = ['experiment_name', 'pretrained_models', 'noise_classifier', 'diffusion_model']
        for key in required_keys:
            self.assertIn(key, config, f"Missing required key: {key}")
        
        print("✓ Configuration loading test passed")
    
    def test_embedding_dimensions(self):
        """Test embedding dimension consistency."""
        print("\n=== Testing Embedding Dimensions ===")
        
        # Create mock codebook and converter
        vocab_size = 1024
        embed_dim = 256
        batch_size = 4
        seq_len = 64
        
        # Create mock codebook
        codebook = torch.randn(vocab_size, embed_dim)
        converter = EmbeddingConverter(None, codebook, "vqgan", self.device)
        
        # Test token to embedding conversion
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        embeddings = converter.tokens_to_embeddings(tokens)
        
        # Check dimensions
        expected_shape = (batch_size, seq_len, embed_dim)
        self.assertEqual(embeddings.shape, expected_shape, 
                        f"Expected shape {expected_shape}, got {embeddings.shape}")
        
        # Test embedding to token conversion
        reconstructed_tokens = converter.embeddings_to_tokens(embeddings)
        self.assertEqual(reconstructed_tokens.shape, tokens.shape,
                        f"Token reconstruction shape mismatch: {reconstructed_tokens.shape} vs {tokens.shape}")
        
        print(f"✓ Token-embedding conversion test passed")
        print(f"  - Tokens shape: {tokens.shape}")
        print(f"  - Embeddings shape: {embeddings.shape}")
        print(f"  - Reconstructed tokens shape: {reconstructed_tokens.shape}")
    
    def test_noise_classifier_dimensions(self):
        """Test noise classifier input/output dimensions."""
        print("\n=== Testing Noise Classifier Dimensions ===")
        
        # Create mock noise classifier
        embed_dim = 256
        batch_size = 32
        
        class MockNoiseClassifier(torch.nn.Module):
            def __init__(self, embed_dim, hidden_dims):
                super().__init__()
                layers = []
                in_dim = embed_dim
                for hidden_dim in hidden_dims:
                    layers.extend([
                        torch.nn.Linear(in_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.1)
                    ])
                    in_dim = hidden_dim
                layers.append(torch.nn.Linear(in_dim, 1))  # Output noise level
                self.network = torch.nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        # Create model
        classifier = MockNoiseClassifier(embed_dim, [512, 256, 128])
        classifier.to(self.device)
        
        # Test forward pass
        embeddings = torch.randn(batch_size, embed_dim).to(self.device)
        noise_predictions = classifier(embeddings)
        
        # Check output dimensions
        expected_shape = (batch_size, 1)
        self.assertEqual(noise_predictions.shape, expected_shape,
                        f"Expected noise prediction shape {expected_shape}, got {noise_predictions.shape}")
        
        print(f"✓ Noise classifier dimension test passed")
        print(f"  - Input shape: {embeddings.shape}")
        print(f"  - Output shape: {noise_predictions.shape}")
    
    def test_diffusion_model_dimensions(self):
        """Test diffusion model input/output dimensions."""
        print("\n=== Testing Diffusion Model Dimensions ===")
        
        # Create mock diffusion model
        embed_dim = 256
        batch_size = 16
        
        class MockDiffusionModel(torch.nn.Module):
            def __init__(self, embed_dim, model_channels):
                super().__init__()
                self.time_embedding = torch.nn.Sequential(
                    torch.nn.Linear(1, model_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(model_channels, model_channels)
                )
                self.denoiser = torch.nn.Sequential(
                    torch.nn.Linear(embed_dim + model_channels, model_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(model_channels, model_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(model_channels, embed_dim)
                )
            
            def forward(self, x, t):
                # x: [B, embed_dim], t: [B]
                t_embed = self.time_embedding(t.float().unsqueeze(-1))
                combined = torch.cat([x, t_embed], dim=-1)
                return self.denoiser(combined)
        
        # Create model
        diffusion_model = MockDiffusionModel(embed_dim, 128)
        diffusion_model.to(self.device)
        
        # Test forward pass
        embeddings = torch.randn(batch_size, embed_dim).to(self.device)
        timesteps = torch.randint(0, 100, (batch_size,)).to(self.device)
        
        denoised_embeddings = diffusion_model(embeddings, timesteps)
        
        # Check output dimensions
        expected_shape = embeddings.shape
        self.assertEqual(denoised_embeddings.shape, expected_shape,
                        f"Expected denoised embedding shape {expected_shape}, got {denoised_embeddings.shape}")
        
        print(f"✓ Diffusion model dimension test passed")
        print(f"  - Input embeddings shape: {embeddings.shape}")
        print(f"  - Input timesteps shape: {timesteps.shape}")
        print(f"  - Output shape: {denoised_embeddings.shape}")
    
    def test_gradient_flow(self):
        """Test that gradients flow through the pipeline correctly."""
        print("\n=== Testing Gradient Flow ===")
        
        embed_dim = 256
        batch_size = 8
        
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, embed_dim)
        ).to(self.device)
        
        # Create input and target on the same device
        input_embeddings = torch.randn(batch_size, embed_dim, device=self.device, requires_grad=True)
        target_embeddings = torch.randn(batch_size, embed_dim, device=self.device)
        
        # Clear any existing gradients
        model.zero_grad()
        if input_embeddings.grad is not None:
            input_embeddings.grad.zero_()
        
        # Forward pass
        output_embeddings = model(input_embeddings)
        loss = torch.nn.functional.mse_loss(output_embeddings, target_embeddings)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(input_embeddings.grad, "Input gradients are None")
        
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient is None for parameter: {name}")
            self.assertFalse(torch.all(param.grad == 0), f"Gradient is zero for parameter: {name}")
        
        print(f"✓ Gradient flow test passed")
        print(f"  - Loss: {loss.item():.6f}")
        print(f"  - Input gradient norm: {input_embeddings.grad.norm().item():.6f}")
        
        # Check gradient magnitudes are reasonable
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        self.assertGreater(total_grad_norm, 0, "Total gradient norm is zero")
        self.assertLess(total_grad_norm, 1000, f"Total gradient norm too large: {total_grad_norm}")
        
        print(f"  - Total gradient norm: {total_grad_norm:.6f}")
    
    def test_pipeline_integration(self):
        """Test full pipeline integration."""
        print("\n=== Testing Pipeline Integration ===")
        
        # Parameters
        vocab_size = 1024
        embed_dim = 256
        batch_size = 4
        seq_len = 32
        
        # Create components
        codebook = torch.randn(vocab_size, embed_dim)
        converter = EmbeddingConverter(None, codebook, "vqgan", self.device)
        
        # Mock noise classifier
        noise_classifier = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        ).to(self.device)
        
        # Mock diffusion model
        diffusion_model = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, embed_dim)
        ).to(self.device)
        
        # Test full pipeline
        print("  1. Generating mock tokens...")
        original_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        print("  2. Converting tokens to embeddings...")
        embeddings = converter.tokens_to_embeddings(original_tokens)
        # Move embeddings to the same device as models
        embeddings = embeddings.to(self.device)
        
        print("  3. Predicting noise levels...")
        noise_levels = noise_classifier(embeddings.view(-1, embed_dim))
        
        print("  4. Applying diffusion denoising...")
        denoised_embeddings = diffusion_model(embeddings.view(-1, embed_dim))
        denoised_embeddings = denoised_embeddings.view(batch_size, seq_len, embed_dim)
        
        print("  5. Converting back to tokens...")
        # Move back to CPU for token conversion if needed
        final_tokens = converter.embeddings_to_tokens(denoised_embeddings.cpu())
        
        # Check all dimensions
        self.assertEqual(original_tokens.shape, final_tokens.shape,
                        f"Token shape mismatch: {original_tokens.shape} vs {final_tokens.shape}")
        
        print(f"✓ Pipeline integration test passed")
        print(f"  - Original tokens shape: {original_tokens.shape}")
        print(f"  - Embeddings shape: {embeddings.shape}")
        print(f"  - Noise levels shape: {noise_levels.shape}")
        print(f"  - Final tokens shape: {final_tokens.shape}")
    
    def test_corruption_methods(self):
        """Test different corruption methods for training data."""
        print("\n=== Testing Corruption Methods ===")
        
        embed_dim = 256
        batch_size = 16
        
        clean_embeddings = torch.randn(batch_size, embed_dim)
        
        # Test different corruption types
        corruption_types = ["gaussian", "uniform", "dropout", "scale"]
        
        for corruption_type in corruption_types:
            print(f"  Testing {corruption_type} corruption...")
            
            corrupted, info = create_embedding_corruption(
                clean_embeddings, 
                corruption_type=corruption_type, 
                noise_level=0.1
            )
            
            # Check dimensions preserved
            self.assertEqual(corrupted.shape, clean_embeddings.shape,
                            f"Shape mismatch for {corruption_type}: {corrupted.shape} vs {clean_embeddings.shape}")
            
            # Check corruption actually happened (except for very small noise)
            if corruption_type != "dropout":
                mse_diff = torch.nn.functional.mse_loss(corrupted, clean_embeddings)
                self.assertGreater(mse_diff.item(), 0, f"No corruption detected for {corruption_type}")
            
            print(f"    ✓ {corruption_type} corruption test passed (MSE: {torch.nn.functional.mse_loss(corrupted, clean_embeddings).item():.6f})")
    
    def test_memory_usage(self):
        """Test memory usage and cleanup."""
        print("\n=== Testing Memory Usage ===")
        
        if not torch.cuda.is_available():
            print("  Skipping memory test (CUDA not available)")
            return
        
        initial_memory = torch.cuda.memory_allocated()
        print(f"  Initial GPU memory: {initial_memory / 1e6:.2f} MB")
        
        # Create large tensors
        large_embeddings = torch.randn(1000, 256, device=self.device)
        peak_memory = torch.cuda.memory_allocated()
        print(f"  Peak GPU memory: {peak_memory / 1e6:.2f} MB")
        
        # Clean up
        del large_embeddings
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        print(f"  Final GPU memory: {final_memory / 1e6:.2f} MB")
        
        # Check memory was freed
        memory_freed = peak_memory - final_memory
        self.assertGreater(memory_freed, 0, "Memory was not properly freed")
        
        print(f"  ✓ Memory cleanup test passed ({memory_freed / 1e6:.2f} MB freed)")


def run_dimension_tests():
    """Run all dimension tests."""
    print("=" * 60)
    print("EMBEDDING THERMALIZER PIPELINE DIMENSION TESTS")
    print("=" * 60)
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPipelineDimensions)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL DIMENSION TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Set up warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Run tests
    success = run_dimension_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 