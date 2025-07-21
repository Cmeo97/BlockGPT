import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

# Add the parent directory of `taming` to sys.path
#sys.path.append(os.path.abspath(".."))
#from main import instantiate_from_config
#instantiate the loss on your own

from .modules.diffusionmodules.model import Encoder, Decoder
from .modules.diffusionmodules.model import EncoderCond
from .modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from .modules.vqvae.quantize import GumbelQuantize
from .modules.vqvae.quantize import EMAVectorQuantizer
from .modules.losses.vqperceptual import VQLPIPSWithDiscriminator as VQLoss
# from taming.modules.diffusionmodules.model import Encoder, Decoder
# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
# from taming.modules.vqvae.quantize import GumbelQuantize
# from taming.modules.vqvae.quantize import EMAVectorQuantizer
def add_perturbation(z, z_q, z_channels, codebook_norm, codebook, alpha, beta, delta):
    # reshape z -> (batch, height * width, channel) and flatten
    z = torch.einsum('b c h w -> b h w c', z).contiguous()
    z_flattened = z.view(-1, z_channels)

    if codebook_norm:
        z = F.normalize(z, p=2, dim=-1)
        z_flattened = F.normalize(z_flattened, p=2, dim=-1)
        embedding = F.normalize(codebook.weight, p=2, dim=-1)
    else:
        embedding = codebook.weight

    d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        torch.sum(embedding ** 2, dim=1) - 2 * \
        torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

    _, min_encoding_indices = torch.topk(d, delta, dim=1, largest=False)
    random_prob = torch.rand(min_encoding_indices.shape[0], device=d.device)
    random_idx = torch.randint(0, delta, random_prob.shape, device=d.device)
    random_idx = torch.where(random_prob > alpha, 0, random_idx)
    min_encoding_indices = min_encoding_indices[torch.arange(min_encoding_indices.size(0)), random_idx]

    perturbed_z_q = codebook(min_encoding_indices).view(z.shape)
    if codebook_norm:
        perturbed_z_q = F.normalize(perturbed_z_q, p=2, dim=-1)
    perturbed_z_q = z + (perturbed_z_q - z).detach()
    perturbed_z_q = torch.einsum('b h w c -> b c h w', perturbed_z_q)

    mask = torch.arange(z.shape[0], device=perturbed_z_q.device) < int(z.shape[0] * beta)
    mask = mask[:, None, None, None]

    return torch.where(mask, perturbed_z_q, z_q)
class VQModel(nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 perturb,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        #self.loss = instantiate_from_config(lossconfig)
        self.loss = VQLoss(**lossconfig['params'])
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # if ckpt_path is not None:
        #     self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        #checkpointing is done differently in diffcast, consult training script. 
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.perturb= perturb
        print("perturbation is ",self.perturb)
    # def init_from_ckpt(self, path, ignore_keys=list()):
    #     sd = torch.load(path, map_location="cpu")["state_dict"]
    #     keys = list(sd.keys())
    #     for k in keys:
    #         for ik in ignore_keys:
    #             if k.startswith(ik):
    #                 print("Deleting key {} from state_dict.".format(k))
    #                 del sd[k]
    #     self.load_state_dict(sd, strict=False)
    #     print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
       # quant, emb_loss, info = self.quantize(h)
        #return quant, emb_loss, info
        return h
    # def encode(self, x):
    #     h = self.encoder(x)
    #     h = self.quant_conv(h)
    #     quant, emb_loss, info = self.quantize(h)
    #     return quant, emb_loss, info
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    # def forward(self, input):
    #     #call forward with batchsizex1x256x256
    #     quant, diff, _ = self.encode(input)
    #     dec = self.decode(quant)
    #     return dec, diff
    # def forward(self,input,optimizer_idx,global_step):
    #     quant, qloss, _ = self.encode(input)
    #     print("quant shape is ",quant.shape)
    #     recons = self.decode(quant)
    #     loss,log_dict = self.compute_loss(qloss,input,recons,optimizer_idx,global_step,last_layer=self.get_last_layer(),split="train")
    #     #optionally consider returning log_dict
    #     return recons, loss,log_dict
    def forward(self,input,optimizer_idx,global_step,alpha,beta,delta):
        h = self.encode(input)
        quant, qloss, _ = self.quantize(h)
        if self.perturb:
            print("perturbing the quantized output")
            quant = add_perturbation(h, quant, self.quantize.e_dim, False, self.quantize.embedding, alpha, beta, delta)
        recons = self.decode(quant)
        loss,log_dict = self.compute_loss(qloss,input,recons,optimizer_idx,global_step,last_layer=self.get_last_layer(),split="train")
        #optionally consider returning log_dict
        return recons, loss,log_dict
    def inference(self,input):
        with torch.no_grad():
            #quant, diff, _ = self.encode(input)
            h = self.encode(input)
            quant, _, _ = self.quantize(h)
            #quant,_,_ = self.encode(input)
            recons = self.decode(quant)
        return recons#.reshape(-1,24,1,128,128)
    def tokenize(self,x,ctx_len,n_tokens,include_sos=False,include_special_toks=True):
            """
            this function is specifically written for the Llama model in ivideogpt. x is a video, and the output is a set of tokens for the entire video.
            we also add start of frame tokens at the beginning of every frame (not the first)

            ctx_len is the number of frames given as conditioning.

            n_tokens is the number of tokens per frame in the embedded space. 
            """
        
            print("special token",include_special_toks)
            n_embd = self.quantize.n_e
            B,T,C,H,W = x.shape
            x = x.reshape(B*T,C,H,W)
            #_,_,info=self.encode(x)
            h= self.encode(x)
            _,_,info=self.quantize(h)
            indices = info[2].view(B,T,-1)
            sf_token = n_embd 
            sf_tokens = torch.ones(B,T,1).to(indices.device,indices.dtype)*sf_token
            if include_special_toks:
                print("entered special token block")
                if include_sos:
                    indices = torch.cat((sf_tokens,indices),dim=2).reshape(B,-1)
                    ctx_indices = ctx_len*n_tokens + ctx_len #3 probably now
                    labels = torch.cat([
                                torch.ones(B,ctx_indices +1 ).to(indices.device, indices.dtype) * -100,  # -100 for no loss
                                indices[:,  (n_tokens+(ctx_len-1)*(n_tokens+1)+2):]], dim=1) #maybe 196 now?

                else:
                    indices = torch.cat((sf_tokens,indices),dim=2).reshape(B,-1)[:,1:] #do not remove the first
                    ctx_indices = ctx_len*n_tokens + (ctx_len-1) #3 probably now
                    labels = torch.cat([
                                torch.ones(B,ctx_indices +1 ).to(indices.device, indices.dtype) * -100,  # -100 for no loss
                                indices[:, (n_tokens+(ctx_len-1)*(n_tokens+1)+1) :]], dim=1) #maybe 196 now?
                
            else:
                    print("entered no special token block")
                    indices = indices.reshape(B,-1)
                    ctx_indices = ctx_len*n_tokens 
                    labels = torch.cat([
                                torch.ones(B,ctx_indices).to(indices.device, indices.dtype) * -100,  # -100 for no loss
                                indices[:, (n_tokens*ctx_len) :]], dim=1) 
                    print("indices shape inside tokenizer", indices.shape)
                    print("labels shape inside tokenizer", labels.shape)

            
            return indices,labels
    
    def detokenize(self,indices,ctx_len,ctx_res,tot_time_steps,include_sos=False,include_special_toks=True):
            """
            this function is also specifically written for iVideoGPT. Indices are B,tokens.
            ctx_len is the number of frames given as conditioning.
            ctx_res is resolution of context frames (same as the resolution of the video) in the embedding space (8 in this case)
            tot_time_steps is the total number of time steps in the video.
            """
            #ctx_res = 8 # TODO: magic number
            dyn_res = ctx_res
            emb_dim = self.quantize.e_dim
            context_length = ctx_len
            if include_special_toks:
                if include_sos:
                    indices = indices[:,1:]
                B = indices.shape[0]
                T=tot_time_steps
                print("indices shape", indices.shape)
                print("future length",((indices.shape[1] + 1 - (1 + ctx_res * ctx_res) * context_length) // (1 + dyn_res * dyn_res)))
                # extract embeddings
                assert (indices.shape[1] + 1 - (1 + ctx_res * ctx_res) * context_length) % (1 + dyn_res * dyn_res) == 0
                future_length = (indices.shape[1] + 1 - (1 + ctx_res * ctx_res) * context_length) // (1 + dyn_res * dyn_res)

                indices = torch.cat([torch.ones(B, 1).to(indices.device, indices.dtype), indices], dim=1)  # concat dummy tokens

                indices = indices.view(B,T,-1)[:,:,1:]
                indices = indices.clamp(min=0, max=self.quantize.n_e - 1)
                print("largest index inside detokenize is ",torch.max(indices))
                quant= self.quantize.embedding(indices.reshape(B,-1))
                quant = quant.view(B*T,ctx_res,ctx_res,emb_dim).permute(0,3,1,2)
                decode = self.decode(quant).view(B,T,-1,128,128)
            else:
                                
                B = indices.shape[0]
                T=tot_time_steps
                print("indices shape", indices.shape)
                print("future length",((indices.shape[1] - (ctx_res * ctx_res) * context_length) // (dyn_res * dyn_res)))
                # extract embeddings
                assert (indices.shape[1] - (ctx_res * ctx_res) * context_length) % (dyn_res * dyn_res) == 0
                indices = indices.view(B,T,-1)
                indices = indices.clamp(min=0, max=self.quantize.n_e - 1)
                print("largest index inside detokenize is ",torch.max(indices))
                quant= self.quantize.embedding(indices.reshape(B,-1))
                quant = quant.view(B*T,ctx_res,ctx_res,emb_dim).permute(0,3,1,2)
                decode = self.decode(quant).view(B,T,-1,128,128)


            return decode

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()
    #(qloss,dummy_input,recons,0,1,last_layer=model.get_last_layer(),split="train")
    def compute_loss(self, qloss, x, xrec, optimizer_idx, global_step, last_layer, split):
        return self.loss(qloss, x,xrec,optimizer_idx, global_step, last_layer=self.get_last_layer(), split=split)
    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        print('input shape before calling forward is ',x.shape)
        #batchsize,1,256,256
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss
            #encoding loss and generator loss (perceptual loss, discriminator loss, codebook loss)
        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
            #only discriminator loss
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelCond(nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = EncoderCond(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        #self.loss = instantiate_from_config(lossconfig)
        self.loss = VQLoss(**lossconfig['params'])
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        #takes the z_chaneels to embedding dim. 
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"]*2, embed_dim, 1)
        #make the decoder equally powerful? by making it 2*z but there is really no need for that. 
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # if ckpt_path is not None:
        #     self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        #checkpointing is done differently in diffcast, consult training script. 
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    # def init_from_ckpt(self, path, ignore_keys=list()):
    #     sd = torch.load(path, map_location="cpu")["state_dict"]
    #     keys = list(sd.keys())
    #     for k in keys:
    #         for ik in ignore_keys:
    #             if k.startswith(ik):
    #                 print("Deleting key {} from state_dict.".format(k))
    #                 del sd[k]
    #     self.load_state_dict(sd, strict=False)
    #     print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    # def forward(self, input):
    #     #call forward with batchsizex1x256x256
    #     quant, diff, _ = self.encode(input)
    #     dec = self.decode(quant)
    #     return dec, diff
    def forward(self,input,optimizer_idx,global_step):
        quant, qloss, _ = self.encode(input)
        recons = self.decode(quant)
        loss,log_dict = self.compute_loss(qloss,input,recons,optimizer_idx,global_step,last_layer=self.get_last_layer(),split="train")
        #optionally consider returning log_dict
        return recons, loss,log_dict
    def inference(self,input):
        with torch.no_grad():
            quant, diff, _ = self.encode(input)
            recons = self.decode(quant)
        return recons#.reshape(-1,24,1,128,128)
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()
    #(qloss,dummy_input,recons,0,1,last_layer=model.get_last_layer(),split="train")
    def compute_loss(self, qloss, x, xrec, optimizer_idx, global_step, last_layer, split):
        return self.loss(qloss, x,xrec,optimizer_idx, global_step, last_layer=self.get_last_layer(), split=split)
    def tokenize(self,x,ctx_len,n_tokens,include_sos=False):
            """
            this function is specifically written for the Llama model in ivideogpt. x is a video, and the output is a set of tokens for the entire video.
            we also add start of frame tokens at the beginning of every frame (not the first)

            ctx_len is the number of frames given as conditioning.

            n_tokens is the number of tokens per frame in the embedded space. 
            """
        

            n_embd = self.quantize.n_e
            B,T,C,H,W = x.shape
            x = x.view(B*T,C,H,W)
            _,_,info=self.encode(x)
            indices = info[2].view(B,T,-1)
            sf_token = n_embd 
            sf_tokens = torch.ones(B,T,1).to(indices.device,indices.dtype)*sf_token
            if include_sos:
                indices = torch.cat((sf_tokens,indices),dim=2).reshape(B,-1)
                ctx_indices = ctx_len*n_tokens + 3 #3 probably now
                labels = torch.cat([
                            torch.ones(B,ctx_indices +1 ).to(indices.device, indices.dtype) * -100,  # -100 for no loss
                            indices[:, 196:]], dim=1) #maybe 196 now?

            else:
                indices = torch.cat((sf_tokens,indices),dim=2).reshape(B,-1)[:,1:] #do not remove the first
                ctx_indices = ctx_len*n_tokens + 2 #3 probably now
                labels = torch.cat([
                            torch.ones(B,ctx_indices +1 ).to(indices.device, indices.dtype) * -100,  # -100 for no loss
                            indices[:, 195:]], dim=1) #maybe 196 now?
            
            
            return indices,labels
    
    def detokenize(self,indices,ctx_len,ctx_res,tot_time_steps,include_sos=False):
            """
            this function is also specifically written for iVideoGPT. Indices are B,tokens.
            ctx_len is the number of frames given as conditioning.
            ctx_res is resolution of context frames (same as the resolution of the video) in the embedding space (8 in this case)
            tot_time_steps is the total number of time steps in the video.
            """
            #ctx_res = 8 # TODO: magic number
            dyn_res = ctx_res
            emb_dim = self.quantize.e_dim
            context_length = ctx_len
            if include_sos:
                indices = indices[:,1:]
            B = indices.shape[0]
            T=tot_time_steps
            print("indices shape", indices.shape)
            print("future length",((indices.shape[1] + 1 - (1 + ctx_res * ctx_res) * context_length) // (1 + dyn_res * dyn_res)))
            # extract embeddings
            assert (indices.shape[1] + 1 - (1 + ctx_res * ctx_res) * context_length) % (1 + dyn_res * dyn_res) == 0
            future_length = (indices.shape[1] + 1 - (1 + ctx_res * ctx_res) * context_length) // (1 + dyn_res * dyn_res)
            indices = torch.cat([torch.ones(B, 1).to(indices.device, indices.dtype), indices], dim=1)  # concat dummy tokens

            indices = indices.view(B,T,-1)[:,:,1:]
            indices = indices.clamp(min=0, max=self.quantize.n_e - 1)
            print("largest index inside detokenize is ",torch.max(indices))
            quant= self.quantize.embedding(indices.reshape(B,-1))
            quant = quant.view(B*T,ctx_res,ctx_res,emb_dim).permute(0,3,1,2)
            decode = self.decode(quant).view(B,T,-1,128,128)
            return decode
    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        print('input shape before calling forward is ',x.shape)
        #batchsize,1,256,256
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss
            #encoding loss and generator loss (perceptual loss, discriminator loss, codebook loss)
        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
            #only discriminator loss
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class EMAVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )
        self.quantize = EMAVectorQuantizer(n_embed=n_embed,
                                           embedding_dim=embed_dim,
                                           beta=0.25,
                                           remap=remap)
    def configure_optimizers(self):
        lr = self.learning_rate
        #Remove self.quantize from parameter list since it is updated via EMA
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []                                           
    

def get_model(**kwargs):
    return VQModel(**kwargs['params'])
def get_cond_model(**kwargs):
    return VQModelCond(**kwargs['params'])
def main():
    torch.manual_seed(42)
    dummy_input = torch.randn(8, 1, 128, 128)
    config = {
    "params": {
        "embed_dim": 256,
        "n_embed": 1024,
        "ddconfig": {
            "double_z": False,
            "z_channels": 128,
            "resolution": 128,
            "in_channels": 1,
            "out_ch": 1,
            "ch": 128,
            "ch_mult": [1, 1, 2, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [16],
            "dropout": 0.0
        },
         "lossconfig" :{
        "params":{
        "disc_conditional": False,
       "disc_in_channels": 1,
        "disc_start": 10000,
        "disc_weight": 0.8,
        "codebook_weight": 1.0,

        }


    }
        
   
    }
    }
  
    model = VQModel(**config['params'])
    recons,qloss = model(dummy_input)
    

    loss= model.compute_loss(qloss,dummy_input,recons,0,1,last_layer=model.get_last_layer(),split="train")
    print(recons)
    print(loss)
    # for name, param in model.named_parameters():
    #     print(f"Parameter: {name}")
    #     #print(param) 
if __name__ == '__main__':
    main()

#next very important step to fix is to see what the dataloader in run.py gives you.

#next step is to basically use the model forward in the run.py, without the discriminator loss. remember to comment out the discriminator loss in the forward function
    #even with optimizer_step=0, the discriminator loss is still computed, but the optimzer step for that is not taken since it is not seen by the optimzer. 
#think about the chain in detail- first the model is instaitatied(the loss is not part of the model), 
#then instatiate the optimzer like run.py does (no complications) since there is no discriminator loss
#then compute the loss function and backpropagate. 
    

#configure the optimizer correctly (check if the optimizer is taking the correct parameters)
#then make sure the structure of the vqgan is same as phydnet 

#currently, optimizer 0 is also backprop through the discriminator, which is not what we want.-solved
#the loss right now still uses the discriminator which is not being trained. 
#structure in inference
# multiGPU
#dataloader
#second optimizer
    
#got the second optimzer to train, then converted to newer pytorch, used compile and flash attention, then got multi gpu training working
#now we need to just see which to train..
#next step is to also try and mimic the same training conditions. 
   
#same model size of diffcast
#contact the TU Delft authors 
    
#run diffcast on knmi data- make sure to log results for whole diffcast and deterministic backbone- try both backbones
#think about one optimizer for both
#earthformer...?
#phydnet
    

#twice the size of the model
#try multi gpu training- chrck batch size- power of 2
    #talk to carlo

#end of nov: get the datasets ready with preprocessing etc, grid and image based
#our model working on the datasets
#two different encoding schemes for grid and encoding
#use 0,1,5,6- to max usage of the gpu

#**** must make sure that when backpropogating the disc loss, it does not flow through the ae_loss. 