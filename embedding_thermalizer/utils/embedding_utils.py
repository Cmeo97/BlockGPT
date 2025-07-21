"""
Embedding utilities for token-embedding conversions in the thermalizer.
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class EmbeddingConverter:
    """
    Handles conversion between tokens and embeddings for VQGAN/VAE tokenizers.
    """
    
    def __init__(self, tokenizer, codebook=None, vqgan_type="vqgan", device="cuda"):
        """
        Initialize embedding converter.
        
        Args:
            tokenizer: Loaded tokenizer model
            codebook: Codebook tensor (for VQGAN)
            vqgan_type: Type of tokenizer ("vqgan" or "vae")
            device: Device to use for computations
        """
        self.tokenizer = tokenizer
        self.codebook = codebook
        self.vqgan_type = vqgan_type
        self.device = device
        
        if vqgan_type == 'vqgan' and codebook is None:
            raise ValueError("Codebook required for VQGAN tokenizer")
        
        logger.info(f"Initialized EmbeddingConverter for {vqgan_type}")
    
    def tokens_to_embeddings(self, tokens):
        """
        Convert discrete tokens to continuous embeddings.
        
        Args:
            tokens: Tensor of token indices [B, T] or [B, T, spatial_dims]
        
        Returns:
            embeddings: Continuous embeddings [B, T, embed_dim] or [B, T, spatial_dims, embed_dim]
        """
        if self.vqgan_type == 'vqgan':
            # For VQGAN, lookup embeddings from codebook
            original_shape = tokens.shape
            tokens_flat = tokens.flatten()
            
            # Clamp tokens to valid range
            tokens_flat = torch.clamp(tokens_flat, 0, self.codebook.shape[0] - 1)
            
            # Ensure codebook is on same device as tokens
            codebook = self.codebook.to(tokens.device)
            
            # Lookup embeddings
            embeddings_flat = codebook[tokens_flat]  # [N, embed_dim]
            
            # Reshape back to original spatial structure plus embedding dimension
            embeddings = embeddings_flat.view(*original_shape, -1)
            
        elif self.vqgan_type == 'vae':
            # For VAE, tokens are already continuous embeddings
            embeddings = tokens
            
        else:
            raise ValueError(f"Unknown vqgan_type: {self.vqgan_type}")
        
        return embeddings
    
    def embeddings_to_tokens(self, embeddings):
        """
        Convert continuous embeddings back to discrete tokens.
        
        Args:
            embeddings: Continuous embeddings [..., embed_dim]
        
        Returns:
            tokens: Discrete token indices (same shape as input without embed_dim)
        """
        if self.vqgan_type == 'vqgan':
            # For VQGAN, find nearest codebook entries
            original_shape = embeddings.shape[:-1]  # Remove embed_dim
            embeddings_flat = embeddings.view(-1, embeddings.shape[-1])  # [N, embed_dim]
            
            # Ensure codebook is on same device as embeddings
            codebook = self.codebook.to(embeddings.device)
            
            # Calculate distances to all codebook vectors
            distances = torch.cdist(embeddings_flat, codebook)  # [N, codebook_size]
            
            # Find nearest codebook entries
            tokens_flat = torch.argmin(distances, dim=1)  # [N]
            
            # Reshape back to original spatial structure
            tokens = tokens_flat.view(*original_shape)
            
        elif self.vqgan_type == 'vae':
            # For VAE, embeddings are the tokens
            tokens = embeddings
            
        else:
            raise ValueError(f"Unknown vqgan_type: {self.vqgan_type}")
        
        return tokens
    
    def get_embedding_from_token_index(self, token_idx):
        """
        Get single embedding vector from token index.
        
        Args:
            token_idx: Single token index (scalar)
        
        Returns:
            embedding: Embedding vector [embed_dim]
        """
        if self.vqgan_type == 'vqgan':
            token_idx = torch.clamp(token_idx, 0, self.codebook.shape[0] - 1)
            # Ensure codebook is on same device as token_idx if tensor
            if isinstance(token_idx, torch.Tensor):
                codebook = self.codebook.to(token_idx.device)
            else:
                codebook = self.codebook
            return codebook[token_idx]
        else:
            raise ValueError("Single token lookup only supported for VQGAN")
    
    def quantize_embeddings(self, embeddings, temperature=1.0):
        """
        Soft quantization of embeddings using Gumbel softmax.
        
        Args:
            embeddings: Continuous embeddings [..., embed_dim]
            temperature: Temperature for Gumbel softmax
        
        Returns:
            quantized_embeddings: Soft-quantized embeddings
        """
        if self.vqgan_type != 'vqgan':
            raise ValueError("Quantization only supported for VQGAN")
        
        original_shape = embeddings.shape[:-1]
        embeddings_flat = embeddings.view(-1, embeddings.shape[-1])  # [N, embed_dim]
        
        # Calculate logits (negative distances to codebook)
        distances = torch.cdist(embeddings_flat, self.codebook)  # [N, codebook_size]
        logits = -distances
        
        # Apply Gumbel softmax
        soft_tokens = F.gumbel_softmax(logits, tau=temperature, hard=False)  # [N, codebook_size]
        
        # Weighted combination of codebook vectors
        quantized_flat = torch.matmul(soft_tokens, self.codebook)  # [N, embed_dim]
        
        # Reshape back
        quantized_embeddings = quantized_flat.view(*original_shape, -1)
        
        return quantized_embeddings


def batch_tokens_to_embeddings(tokens, converter, batch_size=64):
    """
    Convert tokens to embeddings in batches to save memory.
    
    Args:
        tokens: Token tensor
        converter: EmbeddingConverter instance
        batch_size: Batch size for processing
    
    Returns:
        embeddings: Converted embeddings
    """
    if tokens.numel() <= batch_size * 1000:  # Process small tensors directly
        return converter.tokens_to_embeddings(tokens)
    
    # Process in batches for large tensors
    original_shape = tokens.shape
    tokens_flat = tokens.view(original_shape[0], -1)  # [B, N]
    
    embeddings_list = []
    for i in range(0, tokens_flat.shape[0], batch_size):
        batch_tokens = tokens_flat[i:i+batch_size]
        batch_embeddings = converter.tokens_to_embeddings(batch_tokens)
        embeddings_list.append(batch_embeddings)
    
    embeddings_flat = torch.cat(embeddings_list, dim=0)
    
    # Reshape back to original structure plus embedding dimension
    embed_dim = embeddings_flat.shape[-1]
    embeddings = embeddings_flat.view(*original_shape, embed_dim)
    
    return embeddings


def batch_embeddings_to_tokens(embeddings, converter, batch_size=64):
    """
    Convert embeddings to tokens in batches to save memory.
    
    Args:
        embeddings: Embedding tensor
        converter: EmbeddingConverter instance
        batch_size: Batch size for processing
    
    Returns:
        tokens: Converted tokens
    """
    if embeddings.numel() <= batch_size * 1000:  # Process small tensors directly
        return converter.embeddings_to_tokens(embeddings)
    
    # Process in batches for large tensors
    original_shape = embeddings.shape[:-1]  # Remove embed_dim
    embeddings_flat = embeddings.view(embeddings.shape[0], -1, embeddings.shape[-1])  # [B, N, embed_dim]
    
    tokens_list = []
    for i in range(0, embeddings_flat.shape[0], batch_size):
        batch_embeddings = embeddings_flat[i:i+batch_size]
        batch_tokens = converter.embeddings_to_tokens(batch_embeddings)
        tokens_list.append(batch_tokens)
    
    tokens_flat = torch.cat(tokens_list, dim=0)
    
    # Reshape back to original structure
    tokens = tokens_flat.view(*original_shape)
    
    return tokens


def create_embedding_corruption(embeddings, corruption_type="gaussian", noise_level=0.1, **kwargs):
    """
    Create corrupted versions of embeddings for training data generation.
    
    Args:
        embeddings: Clean embeddings [B, ..., embed_dim]
        corruption_type: Type of corruption ("gaussian", "uniform", "dropout")
        noise_level: Intensity of corruption
        **kwargs: Additional parameters for corruption
    
    Returns:
        corrupted_embeddings: Corrupted embeddings
        corruption_info: Information about the corruption applied
    """
    if corruption_type == "gaussian":
        noise = torch.randn_like(embeddings) * noise_level
        corrupted = embeddings + noise
        corruption_info = {"type": "gaussian", "std": noise_level}
        
    elif corruption_type == "uniform":
        noise = (torch.rand_like(embeddings) - 0.5) * 2 * noise_level
        corrupted = embeddings + noise
        corruption_info = {"type": "uniform", "range": noise_level}
        
    elif corruption_type == "dropout":
        mask = torch.rand_like(embeddings) > noise_level
        corrupted = embeddings * mask
        corruption_info = {"type": "dropout", "rate": noise_level}
        
    elif corruption_type == "scale":
        scale = 1.0 + (torch.rand_like(embeddings) - 0.5) * 2 * noise_level
        corrupted = embeddings * scale
        corruption_info = {"type": "scale", "range": noise_level}
        
    else:
        raise ValueError(f"Unknown corruption_type: {corruption_type}")
    
    return corrupted, corruption_info


if __name__ == "__main__":
    # Test embedding converter
    torch.manual_seed(42)
    
    # Create dummy codebook and tokens for testing
    codebook = torch.randn(1024, 256)  # 1024 tokens, 256 dim
    converter = EmbeddingConverter(None, codebook, "vqgan")
    
    # Test token to embedding conversion
    tokens = torch.randint(0, 1024, (4, 64))  # Batch of 4, 64 tokens each
    embeddings = converter.tokens_to_embeddings(tokens)
    print(f"Tokens shape: {tokens.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test embedding to token conversion
    reconstructed_tokens = converter.embeddings_to_tokens(embeddings)
    print(f"Reconstructed tokens shape: {reconstructed_tokens.shape}")
    print(f"Reconstruction exact: {torch.equal(tokens, reconstructed_tokens)}")
    
    # Test corruption
    corrupted, info = create_embedding_corruption(embeddings, "gaussian", 0.1)
    print(f"Corruption info: {info}")
    print(f"Corruption MSE: {F.mse_loss(embeddings, corrupted).item()}") 