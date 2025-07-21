
import torch
from torch import nn

import json
from dataclasses import dataclass
from models.blockGPT.model import GPT, GPTConfig
import safetensors.torch as sf

def get_model(config):
    return blockGPTBackbone(config).eval()


@dataclass
class BlockGPTBackboneConfig:
     vqgan_type: str = 'vqgan' 
     vqgan_ckpt: str = None
     vqgan_config: str = 'configs/config_vqgan.json'
     predictor_type: str = 'blockGPT'
     predictor_ckpt: str = None
     predictor_config: str = None
     device: str = 'cuda'
     segment_length: int = 9
     context_length: int = 3
     n_tokens: int = 64
     context_res: int = 8
class blockGPTBackbone(nn.Module):
    def __init__(self,config):
        super(blockGPTBackbone, self).__init__()

        #nothing should be None
        assert config.vqgan_ckpt is not None, "vqgan checkpoint should not be None"
        assert config.predictor_ckpt is not None, "predictor checkpoint should not be None"
        assert config.predictor_config is not None, "predictor config should not be None"
        self.vqgan_type = config.vqgan_type
        self.vqgan_ckpt = config.vqgan_ckpt
        self.vqgan_config = config.vqgan_config
        self.predictor_type = config.predictor_type
        self.predictor_ckpt = config.predictor_ckpt
        self.predictor_config = config.predictor_config
        self.device = config.device
        self.context_length = config.context_length
        self.segment_length = config.segment_length
        self.n_tokens = config.n_tokens
        self.context_res= config.context_res
        self.special_token = False
        
        self.tokenizer,self.predictor = self.get_tokenizer_and_model_pair(self.predictor_type,self.predictor_config,self.predictor_ckpt,self.vqgan_type,self.vqgan_config,self.vqgan_ckpt)

    def load_vqgan_models(self,vqgan_type,configs,checkpoints):
                loaded_models = []
                from models.taming.vqgan import get_model, get_cond_model
                
                for i in range(len(configs)):
                    #instantiate the model:
                    
                    data = torch.load(checkpoints[i])
                    if vqgan_type == 'vqgan':
                        model = get_model(**configs[i])
                    elif vqgan_type == 'cond_vqgan':
                        model = get_cond_model(**configs[i])
                # model = get_model(**configs[i])
                    model.load_state_dict(data['model'])
                    loaded_models.append(model)
                    print("loaded model number",i)
                return loaded_models

    def get_tokenizer(self,vqgan_name,vqgan_config,vqgan_checkpoint):
            if vqgan_name == 'vqgan':
                
                checkpoints = [vqgan_checkpoint]
                if vqgan_name == 'vqgan':
                    with open(vqgan_config) as f:
                        config_vqgan = json.load(f)
                        config = [config_vqgan]
                    vq_model = self.load_vqgan_models('vqgan',config,checkpoints)[0]

                vq_model = vq_model.eval()

        
            else:
                raise NotImplementedError
            return vq_model


    def get_tokenizer_and_model_pair(self,predictor_name,config_name,predictor_dict,vqgan_name,vqgan_config,vqgan_checkpoint):
            tokenizer = self.get_tokenizer(vqgan_name,vqgan_config,vqgan_checkpoint)

            if predictor_name == 'blockGPT':
                from models.blockGPT.model import GPT, GPTConfig
                import safetensors.torch as sf
                with open(config_name) as f:
                    config = json.load(f)
                config_gpt = GPTConfig(**config)
                model = GPT(config_gpt)
                sf.load_model(model,predictor_dict)

            tokenizer =tokenizer.to(self.device) 
            model=model.to(self.device)
            return tokenizer,model

    
       
    

    def generate(self,batch,predictor,tokenizer):

        pixel_values = batch.to(self.device, non_blocking=True)
    

        tokens, labels = tokenizer.tokenize(pixel_values,self.context_length,self.n_tokens,include_sos=False,include_special_toks=False)
        gen_input = tokens[:, :self.context_length * self.n_tokens]
        max_new_tokens = self.n_tokens * (self.segment_length - self.context_length)

            
        gen_kwargs={
        
                            'max_new_tokens': max_new_tokens,
                        }       
        
        generated_tokens = predictor.generate(
                            gen_input, gen_kwargs['max_new_tokens'])


        recon_output =tokenizer.detokenize(generated_tokens, self.context_length,self.context_res,self.segment_length,include_sos=False,include_special_toks=False)
        

        recon_output = recon_output.clamp(0.0, 1.0)
        return recon_output

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        assert compute_loss == False, "Compute loss is not supported for blockGPTBackbone. The model is kept frozen"
        #dummy ground truth frames required during inference. It will be replaced by the generated frames
        if frames_gt is None:
            frames_gt = torch.zeros(frames_in.shape[0],self.segment_length-self.context_length, frames_in.shape[2], frames_in.shape[3], frames_in.shape[4], device=frames_in.device)
        batch = torch.cat([frames_in, frames_gt], dim=1) 
        preds = self.generate(batch,self.predictor,self.tokenizer)
        preds = preds[:,3:]

        return preds,None
        
