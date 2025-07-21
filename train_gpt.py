import copy
import os
import warnings

current_path = os.getcwd()
print("current_path is: ", current_path)

import numpy as np
import torch
from tqdm import tqdm, trange
import time
import sys

import argparse
import json
import logging
import math
import os
from pathlib import Path
import imageio

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from tqdm.auto import tqdm

from safetensors.torch import load_file
import transformers
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


from models.blockGPT.model import GPTConfig,GPT,ContinuousGPTConfig,ContinuousGPT


from peft import LoraConfig, TaskType, get_peft_model
from dataset.get_datasets import get_dataset
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.39.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
def get_dataloaders(args):
    # DataLoaders creation:
    if args.segment_length_sevir is None:
        args.segment_length_sevir = args.segment_length 
    train_data, valid_data, test_data, color_save_fn, PIXEL_SCALE, THRESHOLDS = get_dataset(
            data_name=args.dataset_name,
            # data_path=self.args.data_path,
            img_size=args.resolution,#128
            seq_len=args.segment_length_sevir,#25
            temp_res_sevir = args.temp_res_sevir,
            batch_size=args.per_device_train_batch_size,
            debug=args.debug
        )
    if args.dataset_name == "knmi" or args.dataset_name == "sevir" or args.dataset_name == 'knmi_5mins':


        train_dataloader = train_data.get_torch_dataloader(num_workers=args.dataloader_num_workers)
        eval_dataloader = valid_data.get_torch_dataloader(num_workers=args.dataloader_num_workers)


    else:
        train_dataloader = torch.utils.data.DataLoader(
                train_data, batch_size=args.per_device_train_batch_size*args.gradient_accumulation_steps, shuffle=True, num_workers=args.dataloader_num_workers, drop_last=True
            )
        eval_dataloader = torch.utils.data.DataLoader(
                valid_data, batch_size=args.per_device_train_batch_size, shuffle=False, num_workers=args.dataloader_num_workers, drop_last=True
            )
 
        print(f"Length of train_dataloader: {len(train_dataloader)}")

        print(f"Length of eval_dataloader: {len(eval_dataloader)}")

    return train_dataloader, eval_dataloader


def load_vqgan_models(vqgan_type,configs,checkpoints):
    
    loaded_models = []
    for i in range(len(configs)):
        #instantiate the model:
        data = torch.load(checkpoints[i])
        if vqgan_type == 'vqgan':
            from models.taming.vqgan import get_model
        #    print("config is: ",**configs[i])
            model = get_model(**configs[i])

        elif vqgan_type == 'vae':

            from models.taming.vae import get_model
            model = get_model(configs[i])
      
       # model = get_model(**configs[i])
        model.load_state_dict(data['model'])
        loaded_models.append(model)
        print("loaded model number",i)
    return loaded_models

def get_tokenizer(args):
    if args.vqgan_type == 'vqgan' or args.vqgan_type == 'vae' :
      
        with open(args.encoder_config) as f:
                    config = json.load(f)
        print("config is: ",config)
        checkpoints = [args.pretrained_model_name_or_path]
        

        vq_model = load_vqgan_models(args.vqgan_type,[config],checkpoints)[0]
        if args.vqgan_type == 'vqgan':
             vocab_size = vq_model.quantize.n_e
       
        elif args.vqgan_type == 'vae':
            vocab_size = None
            assert args.special_token == False
        vq_model = vq_model.eval()



        if args.special_token:
            vocab_size += 1
        print("vq vocab size is: ",vocab_size)

    else:
        raise NotImplementedError
    return vq_model, vocab_size


def generate_multiple_times(predictor_name,
    gen_times,
    accelerator,
    model,
    gen_input,
    actions,
    gen_kwargs,
    max_batch_size=None,
    verbose=False,
    reward_prediction=False,
):
    max_batch_size = max_batch_size or gen_input.shape[0]
    assert max_batch_size % gen_input.shape[0] == 0
    repeat_times = max_batch_size // gen_input.shape[0]
    assert gen_times % (max_batch_size // gen_input.shape[0]) == 0
    repeat_iters = gen_times // (max_batch_size // gen_input.shape[0])
    results = []
    rewards = []
    print("gen kwargs: ", gen_kwargs)
    print("repeat iters", repeat_iters)
    for i in trange(repeat_iters, disable=not verbose):
            print('iter number',i)
            print('gen input shape is', gen_input.shape)
            if predictor_name == 'blockGPT':
                generated_tokens = accelerator.unwrap_model(model).generate(
                    gen_input.repeat(repeat_times, 1), gen_kwargs['max_new_tokens']
                   # this is meaningless but supressing warning
                )
            elif predictor_name == 'continuousGPT':
                print("gen input shape is: ", gen_input.shape)
                generated_tokens = accelerator.unwrap_model(model).generate(
                   gen_input.repeat(repeat_times, 1,1), gen_kwargs['max_new_tokens']
                   # this is meaningless but supressing warning
                )
            else:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    gen_input.repeat(repeat_times, 1),
                    **gen_kwargs,
                    **({'action': actions.repeat(repeat_times, 1, 1)} if actions is not None else {}),
                    pad_token_id=50256,  # this is meaningless but supressing warning
                )
            print("generated tokens shape in each generate iter is: ", generated_tokens.shape)
            results.append(generated_tokens)
 
    results = torch.cat(results, dim=0)  
    print("results shape is ", results.shape)
    return results


def batch_forward(batch_size, input, forward, verbose=False):
    return torch.cat([forward(input[i: i + batch_size]) for i in trange(0, input.shape[0], batch_size, disable=not verbose)], dim=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--config_name", type=str, default="configs/GPT/config_blockGPT_KNMI30.json",
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument('--llama_attn_drop', default=None, type=float)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=2000000,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--dataset_name", type=str, default="knmi",
                        help=(
                            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
                            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
                            " or to a folder containing files that 🤗 Datasets can understand."
                        ),
                        )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="constant_with_warmup",
                        help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=5000,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, require=True, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--vqgan_type", type=str, default="vqgan",
                        choices=['vqgan','vae'], help="VQGAN model type to use.")
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True)
    parser.add_argument('--n_tokens_per_frame', type=int,default=64, required=False,help="number of tokens per frame defined in the encoder")
    parser.add_argument('--encoder_config', type=str, help = 'path to the encoder config file. Not required for ctx_vqgan')
    parser.add_argument('--pretrained_transformer_path', type=str, default=None)
    parser.add_argument('--load_internal_llm', default=False, action='store_true')
    parser.add_argument("--trust_remote_code", type=bool, default=False,
                        help=(
                            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                            "execute code present on the Hub on your local machine."
                        ),
                        )
    parser.add_argument("--checkpointing_steps", type=int, default=5000,
                        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="If the training should continue from a checkpoint folder.")
    parser.add_argument("--with_tracking", type=bool, default=True,
                        help="Whether to enable experiment trackers for logging.")
    parser.add_argument("--report_to", type=str, default="wandb",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
                            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
                            "Only applicable when `--with_tracking` is passed."
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                        help=(
                            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    ),
    )
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--lora', default=False, action='store_true')
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=32, type=float)
    parser.add_argument('--lora_dropout', default=0.0, type=float)
    parser.add_argument('--gradient_checkpointing', default=False, action='store_true')
    parser.add_argument("--max_grad_norm", default=None, type=float, help="Max gradient norm.")

    parser.add_argument('--reward_prediction', default=False, action='store_true')
    parser.add_argument('--start_completed_steps', default=None, type=int)
    parser.add_argument('--action_recon', default=None, type=float)
    parser.add_argument('--predictor_name', default=None, type=str)
    # datasets
    parser.add_argument("--segment_length", type=int, default=2,
                        help="The length of the segmented trajectories to use for the training.")
    parser.add_argument("--segment_length_sevir", type=int, default=None,
                        help="to be used only for 30 min temp res sevir data, else leave None")
    parser.add_argument("--temp_res_sevir", type=int, default=5,
                        help="Temporal Resolution between frames (Currently only for SEVIR)")          
    parser.add_argument("--context_length", type=int, default=1)
    parser.add_argument('--video_stepsize', default=1, type=int)
    parser.add_argument('--dataset_path', default='/data2/frame_datasets',
                        type=str, help='Path to the tensorflow datasets')
    parser.add_argument('--dataset_size', default=None, type=int)
    parser.add_argument('--sthsth_root_path',
                        default='/data/something-something-v2/20bn-something-something-v2-frames-64', type=str)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help=(
                            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
                        ),
                        )
    parser.add_argument('--strong_aug', default=False, action='store_true')
    parser.add_argument('--no_aug', default=False, action='store_true')
    parser.add_argument('--oxe_data_mixes_type', default='select', type=str)

    parser.add_argument("--log_steps", type=int, default=100, help=("Print logs every X steps."))
    parser.add_argument("--validation_steps", type=int, default=2000)
    parser.add_argument('--skip_first_val', default=False, action='store_true')
    parser.add_argument('--latest_checkpoint_only', default=False, action='store_true')
    parser.add_argument('--special_token', type=bool,default=False)
    parser.add_argument('--action_conditioned', default=False, action='store_true')
    parser.add_argument('--action_dim', default=4, type=int, help='action dimension for the task')
    parser.add_argument('--embed_no_wd', default=False, action='store_true')
    parser.add_argument('--goal_conditioned', default=False, action='store_true')

    # evaluation
    parser.add_argument('--max_eval_iters', default=100, type=int)
    parser.add_argument('--use_eval_dataset', default=False, action='store_true')
    parser.add_argument('--eval_generate_times', default=1, type=int, help='for eval, fvd')
    parser.add_argument('--max_generate_batchsize', default=None, type=int)
    parser.add_argument('--max_decode_batchsize', default=None, type=int)
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--log_gif_interval', default=10, type=int)
    parser.add_argument("--debug", type=bool, default=False,)
    parser.add_argument("--include_sos",type=bool,default=False)
    args = parser.parse_args()
    args.model_type = args.config_name.split('/')[1]


    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = args.per_device_train_batch_size


    return args


@torch.no_grad
def evaluate(args, accelerator, tokenizer, model, eval_dataloader, completed_steps):
    model.eval()
    losses = []
    mse_values, psnr_values, ssim_values = [], [], []
    eval_iters = min(len(eval_dataloader), args.max_eval_iters)
    bar = tqdm(range(eval_iters), desc="validation", disable=not accelerator.is_local_main_process)

    for i, batch in enumerate(eval_dataloader):
        if i == args.max_eval_iters:
            break

        pixel_values = batch.to(accelerator.device, non_blocking=True)
        batch_size = pixel_values.shape[0]


        if args.vqgan_type == 'vqgan':
         n_tokens=args.n_tokens_per_frame
         tokens, labels = accelerator.unwrap_model(tokenizer).tokenize(pixel_values,
                                                                      args.context_length,
                                                                      n_tokens,include_sos=args.include_sos,include_special_toks=args.special_token
                                                                    
                                                     )
  
        elif args.vqgan_type == 'vae':
            tokens,mask = accelerator.unwrap_model(tokenizer).tokenize(pixel_values,args.context_length)

        else:
            tokens, labels = accelerator.unwrap_model(tokenizer).tokenize(pixel_values,
                                                                      args.context_length,
                                                                   
                                                                      )
            

        if args.vqgan_type != 'vae':
          model_input = {'input_ids': tokens, 'labels': labels}
  

        if args.reward_prediction:
            if accelerator.num_processes > 1:
                outputs, rewards = model.module(**model_input)
            else:
                outputs, rewards = model(**model_input)
        else:
            if accelerator.num_processes > 1:
                if args.predictor_name == 'blockGPT':
                    input_ids =model_input['input_ids']
                    labels = model_input['labels']
                    logits,loss = model.module(input_ids,labels)
                elif args.predictor_name == 'continuousGPT':
                    _,loss = model.module(tokens,mask)
                else:
                 outputs = model.module(**model_input)
            else:
                if args.predictor_name == 'blockGPT':
                    input_ids = model_input['input_ids']
                    labels =  model_input['labels']
                    logits,loss = model(input_ids,labels)
                elif args.predictor_name == 'continuousGPT':
                    _,loss = model(tokens,mask)
                else: 
                  outputs = model(**model_input)
       

        if args.predictor_name != 'blockGPT' and args.predictor_name != 'continuousGPT':
            loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

        # predict next frames
        if (i % args.log_gif_interval == 0 and accelerator.is_main_process):
            if args.special_token:
                if args.vqgan_type == 'vqgan' :
                    gen_input = tokens[:, :args.context_length * (args.n_tokens_per_frame + 1)]  # TODO: magic number #shape is [batch_size, 1025]
               
                    max_new_tokens = (1 + args.n_tokens_per_frame) * (args.segment_length - args.context_length) - 1   
                    if args.include_sos:
                         gen_input = tokens[:, :args.context_length * (args.n_tokens_per_frame + 1)] # note that we do not give the start of frame token, because max_new_tokens should be divisible by 65
                         max_new_tokens = (1 + args.n_tokens_per_frame) * (args.segment_length - args.context_length)  
                else:    
                    gen_input = tokens[:, :args.context_length * (1024 + 1)]  # TODO: magic number #shape is [batch_size, 1025]
                    
                    max_new_tokens = (1 + args.n_tokens_per_frame) * (args.segment_length - args.context_length) - 1
            else:
                if args.vqgan_type == 'vqgan' :
                    gen_input = tokens[:, :args.context_length * args.n_tokens_per_frame]
                    max_new_tokens = args.n_tokens_per_frame * (args.segment_length - args.context_length)
                elif args.vqgan_type == 'vae':
                    gen_input = tokens[:,:args.context_length*args.n_tokens_per_frame,:]
                    max_new_tokens = args.n_tokens_per_frame * (args.segment_length - args.context_length)

            if args.reward_prediction:
                generated_tokens, rewards = generate_multiple_times(args.predictor_name,
                    args.eval_generate_times,
                    accelerator, model, gen_input, actions if args.action_conditioned else None,
                    gen_kwargs={
                        'do_sample': True,
                        'temperature': 0.4,
                        'top_k': 70,
                        'max_new_tokens': max_new_tokens,
                    },
                    max_batch_size=args.max_generate_batchsize,
                    verbose=False,
                    # verbose=True,
                    reward_prediction=True,
                )
            else:
                generated_tokens = generate_multiple_times(args.predictor_name,
                    args.eval_generate_times,
                    accelerator, model, gen_input, actions if args.action_conditioned else None,
                    gen_kwargs={
                        'do_sample': True,
                        'temperature': 0.4,
                        'top_k': 70,
                        'max_new_tokens': max_new_tokens,
                    },
                    max_batch_size=args.max_generate_batchsize,
                    verbose=False,
                    # verbose=True,
                    reward_prediction=False,
                )

            context_res = int(np.sqrt(args.n_tokens_per_frame))
            if args.max_decode_batchsize is not None and generated_tokens.shape[0] > args.max_decode_batchsize:
                if args.vqgan_type == 'vqgan' :
                    recon_output = batch_forward(
                    args.max_decode_batchsize,
                    generated_tokens,
                    lambda x: accelerator.unwrap_model(tokenizer).detokenize(
                        x, args.context_length,context_res,args.segment_length,include_sos=args.include_sos,include_special_toks=args.special_token
                        # special_token=args.special_token
                    )
                    )  
                elif args.vqgan_type == 'vae':
                    recon_output = batch_forward(
                    args.max_decode_batchsize,
                    generated_tokens,
                    lambda x: accelerator.unwrap_model(tokenizer).detokenize(
                        x, args.segment_length,
                        # special_token=args.special_token
                    )
                    )
                else:
                    recon_output = batch_forward(
                        args.max_decode_batchsize,
                        generated_tokens,
                        lambda x: accelerator.unwrap_model(tokenizer).detokenize(
                            x, args.context_length,
                            # special_token=args.special_token
                        )
                    )
            else:
                if args.vqgan_type == 'vqgan':
                    print("largest token in generated tokens ",torch.max(generated_tokens))
                    recon_output = accelerator.unwrap_model(tokenizer).detokenize(
                    generated_tokens, args.context_length,context_res,args.segment_length,include_sos=args.include_sos,include_special_toks=args.special_token
                    # special_token=args.special_token
                    ) 
                elif args.vqgan_type == 'vae':
                    recon_output = accelerator.unwrap_model(tokenizer).detokenize(
                    generated_tokens, args.segment_length,
                    )
                else:   
                    recon_output = accelerator.unwrap_model(tokenizer).detokenize(
                        generated_tokens, args.context_length,
                      
                    )  # generated_tokens will include gen_input
            recon_output = recon_output.clamp(0.0, 1.0)

        # save predicted video
        if i % args.log_gif_interval == 0 and accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, "images", f"val-samples-{completed_steps}")
            os.makedirs(save_path, exist_ok=True)
            gt_frames = [(pixel_values[0, i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                         for i in range(pixel_values.shape[1])]
            recon_frames = [(recon_output[0, i].permute(1, 2, 0).detach().cpu().numpy() *
                             255).astype(np.uint8) for i in range(recon_output.shape[1])]
            frames = [np.concatenate([gt_frames[i], recon_frames[i], np.abs(gt_frames[i].astype(
                float) - recon_frames[i].astype(float)).astype(np.uint8)]) for i in range(len(gt_frames))]
         #   imageio.mimsave(f"{save_path}/val-samples-{completed_steps}-{i}.gif", frames, fps=4, loop=0)
            
            assert pixel_values.shape[0] == recon_output.shape[0]
            mse_values.append(torch.mean((pixel_values - recon_output) ** 2).repeat(batch_size))



        bar.update(1)

    if accelerator.is_main_process:
        try:
            eval_loss = torch.cat(losses, 0).mean().item()
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        eval_logs = {
            'eval/eval_loss': eval_loss,
            'eval/perplexity': perplexity,
            'eval/mse': torch.cat(mse_values, 0).mean().item(),
        }



    
        accelerator.log(eval_logs, step=completed_steps)

    model.train()

    if accelerator.is_main_process:
        return eval_logs
    else:
        return None


def plot_gif(x, postfix=''):
    # [B, T, C, H, W]
    frames = [(x[0, i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8) for i in range(x.shape[1])]
    imageio.mimsave(f"tmp{postfix}.gif", frames, fps=4, loop=0)


def start_train():

    args = parse_args()
    print("special token is ",args.special_token)
    args.output_dir = os.path.join(args.output_dir, time.strftime("%Y-%m-%d-%X", time.localtime()) + (
        "" if args.exp_name is None else f"-{args.exp_name}"))
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment

    logging_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logging_dir, exist_ok=True)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

            with open(os.path.join(args.output_dir, "cmd.sh"), "w") as f:
                f.write("python " + " ".join(sys.argv))

            src_path = os.path.join(args.output_dir, 'src')
            os.makedirs(src_path, exist_ok=True)
          #  os.system(f"rsync -rv --exclude-from=.gitignore . {src_path}")

    accelerator.wait_for_everyone()

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    train_dataloader, eval_dataloader = get_dataloaders(args)
    tokenizer, vocab_size = get_tokenizer(args)

    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("gradient checkpointing enabled")


    if args.pretrained_transformer_path is not None:
        state_dict = load_file(os.path.join(args.pretrained_transformer_path, 'model.safetensors'))
        if args.load_internal_llm:
            model.llm.load_state_dict(state_dict, strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)
        logger.info("Finetuning the model from " + args.pretrained_transformer_path)
    else:
        logger.info("Training new model from scratch")

    
    if args.predictor_name == 'blockGPT':

        with open(args.config_name) as f:
            config = json.load(f)
        config_gpt = GPTConfig(**config)
        model = GPT(config_gpt)
    elif args.predictor_name =='continuousGPT':

        with open(args.config_name) as f:
            config = json.load(f)
        config_gpt = ContinuousGPTConfig(**config)
        model = ContinuousGPT(config_gpt)

    no_decay = []
    if args.embed_no_wd:
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                if pn.endswith('bias') or \
                    (pn.endswith('weight') and isinstance(m, torch.nn.Embedding)) or \
                    (pn.endswith('weight') and isinstance(m, torch.nn.LayerNorm)):
                    fpn = '%s.%s' % (mn, pn) if mn else pn
                    no_decay.append(fpn)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )



 
    model, tokenizer, optimizer, lr_scheduler, eval_dataloader = accelerator.prepare(
        model, tokenizer, optimizer, lr_scheduler, eval_dataloader
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    # if accelerator.distributed_type == DistributedType.TPU:
    #     model.tie_weights()

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps

    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    if args.start_completed_steps is not None:
        completed_steps = args.start_completed_steps
        progress_bar.update(completed_steps)
    starting_epoch = 0
    end = time.time()

    lastest_output_dir, lastest_completed_steps = None, None
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            raise NotImplementedError
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            # resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            resume_step = int(training_difference.replace("checkpoint_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

        lastest_output_dir, lastest_completed_steps = args.resume_from_checkpoint, completed_steps

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    avg_loss = None

    if args.eval_only:
        eval_logs = evaluate(args, accelerator, tokenizer, model, eval_dataloader, completed_steps)
        if eval_logs is not None:
            print(args.pretrained_model_name_or_path)
            print(args.pretrained_transformer_path)
            print(eval_logs)
        return
    print("model is: ", model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            print("skip first batches", resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            
            pixel_values = batch.to(accelerator.device, non_blocking=True)

            optimizer.zero_grad()
       


            with torch.no_grad():
                if args.vqgan_type == "vqgan" :
                    n_tokens = args.n_tokens_per_frame
                    tokens, labels = accelerator.unwrap_model(tokenizer).tokenize(pixel_values, args.context_length,n_tokens,include_sos=args.include_sos,include_special_toks=args.special_token)

                elif args.vqgan_type == 'vae':
                    tokens,mask = accelerator.unwrap_model(tokenizer).tokenize(pixel_values,args.context_length)
                
                

                else: #ctx_vqgan
                    raise NotImplementedError("the model is not implemented yet")                                                               
               
                
                if args.vqgan_type !='vae':
                    model_input = {
                        'input_ids': tokens,
                        'labels': labels,
                    }
           


            with accelerator.accumulate(model):
                if args.predictor_name == 'blockGPT':
                    input_ids = model_input['input_ids']
                    labels = model_input['labels']
                    _,loss = model(input_ids,labels)
                elif args.predictor_name == 'continuousGPT':
                    print("mask device is ",mask.device)
                    _,loss = model(tokens,mask)
                else:
                    outputs = model(**model_input)
                    loss = outputs.loss                  
                avg_loss = accelerator.gather(loss.repeat(args.per_device_train_batch_size)).float().mean()
               
                accelerator.backward(loss)

                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if accelerator.sync_gradients and accelerator.is_main_process:
                batch_time = time.time() - end
                progress_bar.set_postfix(batch_time=batch_time)
                end = time.time()
                # Log metrics
                if completed_steps % args.log_steps == 0:
                    logs = {
                        "batch_time": batch_time,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "loss": avg_loss.item(),
                    }
             
                    accelerator.log(logs, step=completed_steps)

                # Save model checkpoint
                if completed_steps % checkpointing_steps == 0 and avg_loss < 4.0:
                    output_dir = f"checkpoints/checkpoint_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    lastest_output_dir = output_dir
                    lastest_completed_steps = completed_steps
                    if args.latest_checkpoint_only:
                        latest_checkpoint_path = os.path.join(args.output_dir,
                                                              f"checkpoints/checkpoint_{completed_steps - checkpointing_steps}")
                        if os.path.exists(latest_checkpoint_path):
                            os.system(f"rm -rf {latest_checkpoint_path}")

            if accelerator.sync_gradients:
                # Validation
                if completed_steps == args.max_train_steps or (completed_steps % args.validation_steps == 1 and (completed_steps > 1 or not args.skip_first_val)):
                    evaluate(args, accelerator, tokenizer, model, eval_dataloader, completed_steps)


            if completed_steps >= args.max_train_steps:
                break

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save_state(output_dir)


if __name__ == "__main__":
    start_train()
