"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from models.blockGPT.diffloss import DiffLoss

#not used, but option
class ScalableSoftmax(nn.Module):
    """Scalable-Softmax (SSMax) implementation from the paper
    'Scalable-Softmax Is Superior for Attention'.

    This is a drop-in replacement for standard Softmax that helps prevent attention
    fading in transformers by incorporating input size scaling. The scaling helps
    maintain focused attention distributions even with large input sizes.

    Args:
        s (float, optional): Scaling parameter that controls attention focusing strength.
            Lower values (e.g. 0.1) produce sharper attention, higher values (e.g. 1.0)
            produce softer attention. Default: 0.43 as used in paper.
        learn_scaling (bool, optional): If True, make scaling parameter learnable.
            Default: True
        bias (bool, optional): If True, adds a learnable bias term. The paper found
            that while bias helps training, it can hurt length generalization.
            Default: False

    Shape:
        - Input: (*, N) where * is any number of dimensions and N is the sequence length
        - Output: Same shape as input
    """

    def __init__(self, s: float = 0.43, learn_scaling: bool = True, bias: bool = False):
        super().__init__()

        # Initialize scaling parameter
        if learn_scaling:
            self.s = nn.Parameter(torch.tensor(s, dtype=torch.float))
        else:
            self.register_buffer("s", torch.tensor(s, dtype=torch.float))

        # Optional bias parameter
        self.has_bias = bias
        if bias:
            self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor,dim: int = -1) -> torch.Tensor:
        """Forward pass applying SSMax along specified dimension.

        Args:
            x (torch.Tensor): Input tensor
            dim (int): Dimension along which to apply SSMax. Default: -1

        Returns:
            torch.Tensor: Output tensor with same shape as input
        """

        # Apply scaling factor based on input size
        
        scale = self.s * math.log(x.size(dim))
        if self.has_bias:
            scale += self.b

        #s = torch.clamp(self.s, min=1e-3, max=5.0)
        # s = 5* torch.sigmoid(self.s)
        # scale = s * math.log(x.size(dim))
        # if self.has_bias:
        #     scale += self.b
        att = x.mul(scale)
        att = att.masked_fill(mask==False, float('-inf')) 
        return F.softmax(att, dim=dim)

def _get_base_mask_generation(src_length, tgt_length, device, num_current_tokens, slot_based):
    src_mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=device)
    delta_lengths = src_length - tgt_length
    if slot_based:
        for tgt_index in range(tgt_length):
            complete_square = (num_current_tokens - tgt_index % num_current_tokens)% num_current_tokens if (tgt_index+1)%num_current_tokens==0 and tgt_index>0 else (num_current_tokens - tgt_index % num_current_tokens)
            src_index = delta_lengths + tgt_index + complete_square 
            src_mask[tgt_index, :src_index] = False  # rows are targets, columns are sources            for tgt_index in range(tgt_length):
    else:
        for tgt_index in range(tgt_length):
            src_index = delta_lengths + tgt_index 
            src_mask[tgt_index, :src_index + 1] = False  # rows are targets, columns are sources
    return src_mask


class SpatioTemporalPositionalEmbeddingWithStart(nn.Module):
    def __init__(self, num_frames: int, tokens_per_frame: int, d_model: int):
        """
        Args:
            num_frames (int): Total number of frames in the video.
            tokens_per_frame (int): Number of tokens per frame (excluding the start token).
                                    This should be a perfect square (e.g. 64 for an 8x8 grid).
            d_model (int): Embedding dimension.
        """
        super().__init__()
        self.num_frames = num_frames
        self.tokens_per_frame = tokens_per_frame
        self.d_model = d_model

        # Determine the grid size (assuming a perfect square)
        self.spatial_dim = int(math.sqrt(tokens_per_frame))
        if self.spatial_dim * self.spatial_dim != tokens_per_frame:
            raise ValueError("tokens_per_frame must be a perfect square (e.g. 64, 256, etc.).")

        # Embedding layers for temporal and spatial positions.
        self.temporal_emb = nn.Embedding(num_frames, d_model)
        self.row_emb = nn.Embedding(self.spatial_dim, d_model)
        self.col_emb = nn.Embedding(self.spatial_dim, d_model)
        # Dedicated embedding for the start-of-frame token.
        self.start_emb = nn.Parameter(torch.zeros(d_model))

        # Precompute the indices for each token position in a video.
        # Layout:
        # - Frame 0: Only normal tokens (tokens_per_frame tokens).
        # - Frames 1 ... num_frames-1: A start token followed by tokens_per_frame normal tokens.
        #
        # Total tokens = tokens_per_frame + (num_frames - 1) * (tokens_per_frame + 1)
        total_tokens = tokens_per_frame + (num_frames - 1) * (tokens_per_frame + 1)
        frame_indices = []
        row_indices = []
        col_indices = []
        is_start = []  # Boolean flag: True if token is a start-of-frame token, else False.

        # Frame 0 (no start token)
        for i in range(tokens_per_frame):
            frame_indices.append(0)
            is_start.append(False)
            row_indices.append(i // self.spatial_dim)
            col_indices.append(i % self.spatial_dim)

        # Frames 1 to num_frames-1: add a start token then the normal tokens.
        for f in range(1, num_frames):
            # Start-of-frame token
            frame_indices.append(f)
            is_start.append(True)
            # Dummy spatial positions for start token (they won’t be used)
            row_indices.append(0)
            col_indices.append(0)
            # Normal tokens for the frame
            for i in range(tokens_per_frame):
                frame_indices.append(f)
                is_start.append(False)
                row_indices.append(i // self.spatial_dim)
                col_indices.append(i % self.spatial_dim)

        # Convert lists to tensors and register as buffers.
        self.register_buffer('frame_indices', torch.tensor(frame_indices, dtype=torch.long))
        self.register_buffer('row_indices', torch.tensor(row_indices, dtype=torch.long))
        self.register_buffer('col_indices', torch.tensor(col_indices, dtype=torch.long))
        self.register_buffer('is_start', torch.tensor(is_start, dtype=torch.bool))
        self.total_tokens = total_tokens

    def forward(self,pos) -> torch.Tensor:
        """
        Returns the positional embeddings for a batch of videos.

        Args:
            batch_size (int): The batch size.
            device (torch.device): The device on which the embeddings should reside.

        Returns:
            pos_emb (Tensor): A tensor of shape 
                (batch_size, total_tokens, d_model),
                where total_tokens = tokens_per_frame + (num_frames - 1) * (tokens_per_frame + 1).
        """
        # Ensure buffers are on the proper device.
        frame_indices = self.frame_indices#.to(device)
        row_indices = self.row_indices#.to(device)
        col_indices = self.col_indices#.to(device)
        is_start = self.is_start#.to(device)

        # Look up the temporal embeddings based on the frame index.
        temp_emb = self.temporal_emb(frame_indices)  # Shape: (total_tokens, d_model)

        # Compute the spatial embeddings for all tokens.
        # For "normal" tokens, this is the sum of row and column embeddings.
        normal_spatial = self.row_emb(row_indices) + self.col_emb(col_indices)  # (total_tokens, d_model)

        # For start tokens, we want to override the spatial embedding with self.start_emb.
        # Using torch.where to select per token:
        spatial_emb = torch.where(
            is_start.unsqueeze(1),  # (total_tokens, 1) mask
            self.start_emb.unsqueeze(0).expand_as(normal_spatial),
            normal_spatial
        )

        # The final positional embedding is the sum of the temporal and spatial embeddings.
        pos_emb = temp_emb + spatial_emb  # Shape: (total_tokens, d_model)
        if pos is not None:
            pos_emb = pos_emb[pos]
        # Expand to include the batch dimension.
       # pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        return pos_emb

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.block_size
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.use_ssmax = config.use_ssmax
        if self.use_ssmax:
            self.flash = False
            self.ssmax = ScalableSoftmax(s=0.43,learn_scaling=True)
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        self.is_block = config.is_block

        
        if self.is_block:
            self.num_block_tokens = config.num_block_tokens
           
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        if self.is_block:
            if T < self.block_size:
                attn_mask = ~_get_base_mask_generation(T,T , x.device, self.num_block_tokens, True)

            else:
                
                 attn_mask = ~_get_base_mask_generation(self.block_size,self.block_size, x.device, self.num_block_tokens, True)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            if self.is_block:
             y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
            else:
             y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)   
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            if not self.use_ssmax:
                if self.is_block:
     
                    att = att.masked_fill(attn_mask.view(1,1,T,T)==False, float('-inf')) 
                else:
                    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
       
            if self.use_ssmax:
                if self.is_block:
               
                    mask = attn_mask.view(1,1,T,T)
                else:
                    mask = self.bias[:,:,:T,:T].bool() 
                att = self.ssmax(att,mask) # default dim = -1
            else:
                att = F.softmax(att, dim=-1)
            print("scaling parameter is",self.ssmax.s)
           # print("att is ", att)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 1024
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    num_frames: int = 9
    tokens_per_frame: int = 64
    pos_emb_type: str = 'temporal'
    is_block: bool = False
    num_block_tokens: int = 64
    use_ssmax: bool =False
    #generation args
    temperature: float = 1.0
    top_k: int = None
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.pos_emb_type = config.pos_emb_type
        self.is_block = config.is_block
        if self.is_block:
            self.num_block_tokens = config.num_block_tokens
            assert self.pos_emb_type== 'temporal', "block attention is only supported with temporal positional embeddings"
        if config.pos_emb_type == 'temporal':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
            
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))
        elif config.pos_emb_type == 'spatiotemporal':            
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))

        self.lm_head = nn.Linear(config.n_embd,config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        if config.pos_emb_type == 'spatiotemporal':
            self.pos_emb_module = SpatioTemporalPositionalEmbeddingWithStart(config.num_frames, config.tokens_per_frame, config.n_embd)
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        #generation args:
        self.temperature = config.temperature
        self.top_k = config.top_k
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
       # if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
       # print("idx shape is ",idx.shape)
       # print("tok_emb shape is ",tok_emb.shape)
        if self.pos_emb_type == 'temporal':
         pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        elif self.pos_emb_type == 'spatiotemporal':
         pos_emb = self.pos_emb_module(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # if targets is not None:
        #     # if we are given some desired targets also calculate the loss
        #     logits = self.lm_head(x)
        #     logits = logits.reshape(-1, self.config.block_size, self.config.vocab_size)
            
        #     shifted_logits = logits[..., :-1, :]
        #     shifted_targets = targets[..., 1:]
        #     loss = F.cross_entropy(shifted_logits.reshape(-1, shifted_logits.size(-1)), shifted_targets.reshape(-1))
        # else:
        #     # inference-time mini-optimization: only forward the lm_head on the very last position
        #    # logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        #     logits = self.lm_head(x)
        #  #   logits = logits.reshape(-1, self.config.block_size, self.config.vocab_size)
        #     loss = None
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            if self.is_block:
                shifted_logits = logits[..., :-self.num_block_tokens, :]
                shifted_targets = targets[..., self.num_block_tokens:]
            else:
                shifted_logits = logits[..., :-1, :]
                shifted_targets = targets[..., 1:]
            loss = F.cross_entropy(shifted_logits.reshape(-1, shifted_logits.size(-1)), shifted_targets.reshape(-1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            if self.is_block:
                logits = self.lm_head(x[:, -self.num_block_tokens:, :])
            else:
                logits = self.lm_head(x[:, [-1], :])
             # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

      #  return logits, loss

   
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if self.is_block:
            max_new_tokens = max_new_tokens//self.num_block_tokens
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            if self.is_block:
                logits = logits[:, -self.num_block_tokens:, :] / self.temperature
            else:
                logits = logits[:, -1, :] / self.temperature
            # optionally crop the logits to only the top k options
            
            if self.top_k is not None:
         
                if self.is_block:
                        B, T, V = logits.shape  # B = batch, T = block_size, V = vocab size

                        # Get top_k logits at each position (B × T × top_k)
                        v, _ = torch.topk(logits, min(self.top_k, V), dim=-1)

                        # Get the k-th largest logit at each position → shape (B, T, 1)
                        kth_value = v[:, :, -1].unsqueeze(-1)

                        # Mask out everything below the k-th largest logit
                        logits = logits.masked_fill(logits < kth_value, float('-inf'))
                else:
                        v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            if self.is_block:
                idx_next = torch.multinomial(probs.view(-1,probs.size(-1)), num_samples=1).view(probs.shape[0],probs.shape[1])
            else:
                idx_next = torch.multinomial(probs,num_samples=1)

            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
@dataclass
class ContinuousGPTConfig:
    block_size: int = 576
   # vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 1024 #linear transformation from the vae_embd_dim
    vae_embd_dim: int = 16
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    num_frames: int = 9
    tokens_per_frame: int = 64
    pos_emb_type: str = 'temporal'
    is_block: bool = False
    num_block_tokens: int = 64
    patch_size: int = 1
    diffloss_w: int = 1024
    diffloss_d: int = 3
    num_sampling_steps: str = '100'
    grad_checkpointing: bool = False
    diffusion_batch_mul: int = 4
    generation_temp: int = 0.9
    use_ssmax: bool =False


class ContinuousGPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.block_size is not None
        self.config = config
        self.pos_emb_type = config.pos_emb_type
        self.is_block = config.is_block
        print("Continuous model block:",self.is_block)
        if self.is_block:
            self.num_block_tokens = config.num_block_tokens
            assert self.pos_emb_type== 'temporal', "block attention is only supported with temporal positional embeddings"
        if config.pos_emb_type == 'temporal':
            self.transformer = nn.ModuleDict(dict(   
                enc_lin = nn.Linear(config.vae_embd_dim, config.n_embd),       
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))
        elif config.pos_emb_type == 'spatiotemporal':            
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))
        print("sampling steps ",config.num_sampling_steps)

       # self.lm_head = nn.Linear(config.n_embd,config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
       # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        self.diffloss = DiffLoss(
            target_channels=config.vae_embd_dim,
            z_channels=config.n_embd,
            width=config.diffloss_w,
            depth=config.diffloss_d,
            num_sampling_steps=config.num_sampling_steps,
            grad_checkpointing=config.grad_checkpointing
        )
        self.diffusion_batch_mul = config.diffusion_batch_mul    
        if config.pos_emb_type == 'spatiotemporal':
            self.pos_emb_module = SpatioTemporalPositionalEmbeddingWithStart(config.num_frames, config.tokens_per_frame, config.n_embd)
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    def patchify(self,x):
        bsz, c, h, w = x.shape
        p = self.config.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
       # if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, mask=None,generate_temp=None,cfg=1.0):
        device = idx.device
        b, t, c = idx.size() # b,t,vae_embd_dim
         # b,t,n_embd
        if mask is not None:
            targets = idx.clone().detach() # targets should be the original vae embeddings

            mask = mask.expand(b, -1)
        idx = self.transformer.enc_lin(idx) # B,T,n_embd
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        if self.pos_emb_type == 'temporal':
         pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        elif self.pos_emb_type == 'spatiotemporal':
         pos_emb = self.pos_emb_module(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(idx + pos_emb) # adds pos emb, now shape is (b,t,n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # shape is (b,t,n_embd)

        if mask is not None:
            # if we are given some desired targets also calculate the loss

            if self.is_block:
                shifted_logits = x[..., :-self.num_block_tokens, :] # logits are of size (b,t,n_embd). truncate the last self.num_block_tokens
                shifted_targets = targets[..., self.num_block_tokens:,:] # targets are of size (b,t,vae_embd_dim). truncate the first self.num_block_tokens
                mask = mask[...,self.num_block_tokens:]


                shifted_logits = shifted_logits.reshape(b*(t-self.num_block_tokens),-1).repeat(self.diffusion_batch_mul,1)
                shifted_targets = shifted_targets.reshape(b*(t-self.num_block_tokens),-1).repeat(self.diffusion_batch_mul,1)
                mask = mask.reshape(b*(t-self.num_block_tokens)).repeat(self.diffusion_batch_mul)
            else:
                shifted_logits = x[..., :-1, :] 
                shifted_targets = targets[..., 1:,:]
                mask = mask[..., 1:]
           
  
                shifted_logits = shifted_logits.reshape(b*(t-1),-1).repeat(self.diffusion_batch_mul,1)
                shifted_targets = shifted_targets.reshape(b*(t-1),-1).repeat(self.diffusion_batch_mul,1)
                mask = mask.reshape(b*(t-1)).repeat(self.diffusion_batch_mul)
                

             #targets,z, mask
            loss = self.diffloss(target=shifted_targets, z=shifted_logits,mask=mask)
        
        else:
            if generate_temp is None:
                generate_temp = self.config.generation_temp
            if self.is_block:
                
                z = x[:,-self.num_block_tokens:,:]
                z = z.reshape(b*self.num_block_tokens,-1)
                next_batch_tokens = self.diffloss.sample(z,generate_temp,cfg)
                next_batch_tokens= next_batch_tokens.reshape(b,self.num_block_tokens,-1)
                return next_batch_tokens,None
            else:
               
                z = x[:,-1,:]
                next_token = self.diffloss.sample(z,generate_temp,cfg)
                next_token = next_token.reshape(b,1,-1)
                return next_token,None

        return None,loss



    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.5,cfg=1.0):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if self.is_block:
            max_new_tokens = max_new_tokens//self.num_block_tokens
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:,:]
            # forward the model to get the logits for the index in the sequence
            idx_next, _ = self(idx_cond,generate_temp=temperature,cfg=cfg)
            # pluck the logits at the final step and scale by desired temperature


            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx