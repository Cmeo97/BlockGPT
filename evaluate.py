from safetensors.torch import load_file
from train_gpt import *
import random
random.seed(0)
import os
import matplotlib.pyplot as plt
import numpy as np
from pysteps.visualization.precipfields import plot_precip_field
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from plotnine import ggplot, aes, geom_line, facet_wrap, labs, theme, element_text
import os.path as osp
import json
import cv2
from matplotlib import colors
from moviepy import ImageSequenceClip
import yaml
import datetime
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")


    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--dataset_name", type=str, default="knmi",
                        help=(
                            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
                            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
                            " or to a folder containing files that 🤗 Datasets can understand."
                        ),
                        )
    
    
    parser.add_argument('--n_tokens', type=int,default=64, required=False,help="number of tokens per frame defined in the encoder")
    
    # datasets
    parser.add_argument("--segment_length", type=int, default=9,
                        help="The length of the segmented trajectories to use for the training.")
    parser.add_argument("--seq_len_sevir", type=int, default=None,
                        help="to be used only for 30 min temp res sevir data, else leave None")
    parser.add_argument("--context_length", type=int, default=3)
    parser.add_argument("--context_res", type=int, default=8,
                        help="The resolution of the frames.")    
    parser.add_argument("--gif", action="store_true", default=False,   
                        help="Whether to save the output as gif or not. Default is False. KNMI currently does not support gif")
    parser.add_argument("--time_resolution", type=int, default=30,
                        help="Time resolution of the dataset")
    
    parser.add_argument("--SCALE_FACTOR", type=float, default=40)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--device", default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
    parser.add_argument("--eval_checkpoints", type=str, default="evaluate_config.yaml",)

    args = parser.parse_args()

    return args


#first pasting all common functions 


def get_eval_dataloader(args):

    if args.seq_len_sevir is None:
       args.seq_len_sevir = args.segment_length
    if args.dataset_name=='knmi' or args.dataset_name=='sevir' or args.dataset_name=='knmi_5mins':
        train_data, valid_data, test_data, color_save_fn, PIXEL_SCALE, THRESHOLDS = get_dataset(
            data_name=args.dataset_name,
            # data_path=self.args.data_path,
            img_size=128,#128
            seq_len=args.seq_len_sevir,#25
            temp_res_sevir = args.time_resolution,
            batch_size=args.batch_size,
            debug=args.debug
        )
        eval_dataloader = test_data.get_torch_dataloader(num_workers=4)
        print("length of test data is ",len(eval_dataloader))
        return eval_dataloader
    

def load_vqgan_models(vqgan_type,configs,checkpoints):
    
    loaded_models = []
    for i in range(len(configs)):
        #instantiate the model:
        data = torch.load(checkpoints[i])
        if vqgan_type == 'vqgan':
            from models.taming.vqgan import get_model
        #    print("config is: ",**configs[i])
            model = get_model(**configs[i])
        elif vqgan_type == 'cond_vqgan':
            from models.taming.vqgan import get_cond_model
           # print("config is: ",configs[i])
            model = get_cond_model(**configs[i])
        elif vqgan_type == 'vit_vqgan':
            from models.enhancing.modules.stage1.vitvqgan import get_model
            model = get_model(configs[i])
        elif vqgan_type == 'vae':

            from models.taming.vae import get_model
            model = get_model(configs[i])

    
       # model = get_model(**configs[i])
        model.load_state_dict(data['model'])
        loaded_models.append(model)
        print("loaded model")
    return loaded_models


def get_tokenizer_and_model_pair(predictor_name,config_name,predictor_dict,vqgan_name,vqgan_config,vqgan_checkpoint,args):
    tokenizer = get_tokenizer(vqgan_name,vqgan_config,vqgan_checkpoint)

    if predictor_name == 'blockGPT':
        from models.blockGPT.model import GPT, GPTConfig
        import safetensors.torch as sf
        with open(config_name) as f:
            config = json.load(f)
        config_gpt = GPTConfig(**config)
        model = GPT(config_gpt)
        sf.load_model(model,predictor_dict)

    tokenizer =tokenizer.to(args.device) 
    model=model.to(args.device)
    return tokenizer,model


def get_tokenizer(vqgan_name,vqgan_config,vqgan_checkpoint):
    if vqgan_name == 'vqgan':
        
        checkpoints = [vqgan_checkpoint]
        if vqgan_name == 'vqgan':
            with open(vqgan_config) as f:
                config_vqgan = json.load(f)
                config = [config_vqgan]
            vq_model = load_vqgan_models('vqgan',config,checkpoints)[0]

        vq_model = vq_model.eval()

   
    else:
        raise NotImplementedError
    return vq_model

def generate(batch,predictor,tokenizer,args):

    pixel_values = batch.to(args.device, non_blocking=True)
   

    tokens, labels = tokenizer.tokenize(pixel_values,args.context_length,args.n_tokens,include_sos=False,include_special_toks=False)
    gen_input = tokens[:, :args.context_length * args.n_tokens]
    max_new_tokens = args.n_tokens * (args.segment_length - args.context_length)

        
    gen_kwargs={
      
                        'max_new_tokens': max_new_tokens,
                    }       
    
    generated_tokens = predictor.generate(
                        gen_input, gen_kwargs['max_new_tokens'])


    recon_output =tokenizer.detokenize(generated_tokens, args.context_length,args.context_res,args.segment_length,include_sos=False,include_special_toks=False)
    

    recon_output = recon_output.clamp(0.0, 1.0)
    return recon_output


def get_average_precipitations(test_loader):
    # Get average precipitations
    avg_precip_test = []
    for i, batch in enumerate(test_loader):
        #print(batch.shape)
        avg_precip_test.append(batch.mean(axis=(1,2,3,4)))
        if i %160 == 0:
            
            print(f"loaded {i} batches")
            #break
    return torch.cat(avg_precip_test, axis=0)


def get_categorized_indices(levels,avg_precip_test):
    #levels = [20,40,60,80,95]
    thresholds = np.percentile(avg_precip_test.numpy(), levels)
    categorized_indices = {}
    avg_precipitation = avg_precip_test
    for i, (low, high) in enumerate(zip([0] + thresholds.tolist(), thresholds.tolist() + [float('inf')])):
            categorized_indices[f"level_{i}"] = np.where((avg_precipitation >= low) & (avg_precipitation < high))[0]
    return categorized_indices

def global_to_batch_index(global_index, batch_size):
    """
    Converts a global index into a (batch_idx, sub_index) pair.

    Args:
        global_index (int): The global index of the event.
        batch_size (int): The batch size of the DataLoader.

    Returns:
        tuple: (batch_idx, sub_index) where:
            - batch_idx: The index of the batch in the DataLoader.
            - sub_index: The index within the batch.
    """
    batch_idx = global_index // batch_size
    sub_index = global_index % batch_size
    return batch_idx, sub_index
def subsample_and_store_events(test_loader, categorized_indices, batch_size, num_samples=10):
    """
    Subsamples events for each level and stores the corresponding tensors.

    Args:
        test_loader (DataLoader): Test DataLoader containing events.
        categorized_indices (dict): Dictionary with levels as keys and global indices as values.
        batch_size (int): Batch size of the DataLoader.
        num_samples (int): Number of events to sample per level.

    Returns:
        dict: Dictionary with levels as keys and tensors of sampled events as values.
    """
    # Step 1: Subsample global indices for each level
    subsampled_indices = {
        level: random.sample(list(global_indices), min(num_samples, len(global_indices)))
        for level, global_indices in categorized_indices.items()
    }

    # Step 2: Convert subsampled global indices to (batch_idx, sub_index) pairs
    batch_indices = {
        level: [global_to_batch_index(idx, batch_size) for idx in indices]
        for level, indices in subsampled_indices.items()
    }

    # Step 3: Iterate through the test loader and store tensors for each level
    stored_events = {level: [] for level in categorized_indices.keys()}

    for batch_idx, batch in enumerate(test_loader):
        for level, indices in batch_indices.items():
            for b_idx, sub_idx in indices:
                if b_idx == batch_idx:
                    stored_events[level].append(batch[sub_idx])

    # Convert lists to tensors for consistency
    stored_events = {level: torch.stack(events) for level, events in stored_events.items()}

    return stored_events

def store_all_events(test_loader, categorized_indices, batch_size):
    """
    Stores all events for each level without subsampling. Enumerates through the 
    test data loader and collects all events referenced by the categorized indices.

    Args:
        test_loader (DataLoader): Test DataLoader containing events.
        categorized_indices (dict): Dictionary with levels as keys and global indices as values.
        batch_size (int): Batch size of the DataLoader.

    Returns:
        dict: Dictionary with levels as keys and tensors of events as values.
    """

    # Use all global indices without subsampling
    all_indices = {
        level: list(global_indices)
        for level, global_indices in categorized_indices.items()
    }

    # Convert all global indices to (batch_idx, sub_index) pairs
    batch_indices = {
        level: [global_to_batch_index(idx, batch_size) for idx in indices]
        for level, indices in all_indices.items()
    }

    # Initialize a dictionary to store events for each level
    stored_events = {level: [] for level in categorized_indices.keys()}

    # Iterate through the test loader and store tensors for each referenced event
    for batch_idx, batch in enumerate(test_loader):
        for level, indices in batch_indices.items():
            for b_idx, sub_idx in indices:
                if b_idx == batch_idx:
                    stored_events[level].append(batch[sub_idx])
    
    # Convert lists to tensors for consistency
    
    stored_events = {level: torch.stack(events) for level, events in stored_events.items()}

    return stored_events

def plot_predictions_per_level_sampled(
    stored_events, 
    loaded_models, 
    output_folder, 
    num_samples=4, 
    device="cuda", 
    chunk_size=2,
    args=None
):
    import os
    import random
    import torch

    os.makedirs(output_folder, exist_ok=True)

    for level, events in stored_events.items():
        print(f"Processing level: {level}")
        B, T, C, H, W = events.shape
        assert C == 1 and H == 128 and W == 128, f"Unexpected event shape: {events.shape}"

        # Sample only required indices
        sampled_indices = random.sample(range(B), min(num_samples, B))
        sampled_events = events[sampled_indices].to(device)

        # Prepare per-model predictions
        predictions_by_model = {model_name: [] for model_name in loaded_models}

        for start in range(0, len(sampled_events), chunk_size):
            end = min(start + chunk_size, len(sampled_events))
            chunk = sampled_events[start:end]

            for model_name, components in loaded_models.items():
                print(f"Inferencing {model_name} on {end-start} samples...")
                if model_name.startswith("diffcast"):
                    model = components["model"].to(device)
                    pred = model.sample(chunk[:, :args.context_length], args.segment_length - args.context_length)[0]
                    pred = torch.cat([chunk[:, :args.context_length], pred], dim=1)
                else:
                    tokenizer = components["tokenizer"].to(device)
                    predictor = components["predictor"].to(device)
                    pred = generate(chunk, predictor, tokenizer,args)

                pred = torch.clamp(pred, 0, 1).cpu()
                pred = pred.view(-1, args.segment_length, 1, 128, 128).detach()
                predictions_by_model[model_name].append(pred)

        for model_name in predictions_by_model:
            predictions_by_model[model_name] = torch.cat(predictions_by_model[model_name], dim=0)

        # Save plots/gifs
        level_folder = os.path.join(output_folder, level)
        os.makedirs(level_folder, exist_ok=True)
        sampled_events_cpu = sampled_events.cpu()

        for i, idx in enumerate(sampled_indices):
            gt = sampled_events_cpu[i]
            preds = {k: v[i] for k, v in predictions_by_model.items()}
            print("prediction shapes ",{k: v[i].shape for k, v in predictions_by_model.items()})
            print("ground truth shape",gt.shape)
            path = os.path.join(level_folder, f"sample_event_{idx}")
  
            if args.dataset_name == 'knmi' or args.dataset_name == 'knmi_5mins':
        
                        display_knmi(gt.numpy(), preds, path)
            else:
                        display_sevir(gt.numpy(), preds, output_path=path, skip_after_5=False,args=args)
                        display_sevir_gifs(gt.numpy(), preds, output_path=path)   
                

        print(f"Saved samples for level: {level}")

def display_sevir_gifs(ground_truth, predictions_dict, output_path=None):
    """
    Generate GIFs comparing model predictions with ground truth for a single event.

    Parameters
    ----------
    ground_truth : ndarray
        NumPy array of ground truth images. Shape: [T, 1, H, W].
    predictions_dict : dict
        Dictionary of model predictions. Keys are model names (str),
        and values are NumPy arrays of shape [T, 1, H, W].
    output_path : str, optional
        Directory where the GIFs will be saved.

    Returns
    -------
    None
    """
    assert ground_truth.ndim == 4, "Ground truth must have shape [T, 1, H, W]"
    for model_name, prediction in predictions_dict.items():
        assert prediction.shape == ground_truth.shape, f"Mismatch with {model_name}"

    # Create output path if needed
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    # Loop through each model and save gif using `vis_res`
    for model_name, pred_seq in predictions_dict.items():
        save_path = os.path.join(output_path, f"{model_name}.gif") if output_path else f"{model_name}.gif"
        vis_res(pred_seq, ground_truth, save_path,model_name)
        print(f"Saved GIF for {model_name} at {save_path}")

def gray2color(image):
    COLOR_MAP = [[0, 0, 0],
              [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
              [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
              [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
              [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
              [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
              [0.9607843137254902, 0.9607843137254902, 0.0],
              [0.9294117647058824, 0.6745098039215687, 0.0],
              [0.9411764705882353, 0.43137254901960786, 0.0],
              [0.6274509803921569, 0.0, 0.0],
              [0.9058823529411765, 0.0, 1.0]]


    PIXEL_SCALE = 255.0
    BOUNDS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, PIXEL_SCALE]
    cmap = colors.ListedColormap(COLOR_MAP )
    bounds = BOUNDS
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    colored_image = cmap(norm(image))

    return colored_image



def vis_res(pred_seq, gt_seq, save_path,model_name, save_grays=False, save_colored=False):
    # pred_seq: ndarray, [T, C, H, W], value range: [0, 1] float
    if isinstance(pred_seq, torch.Tensor):
        pred_seq = pred_seq.detach().cpu().numpy()
       # gt_seq = gt_seq.cpu().numpy()
    pred_seq = pred_seq.squeeze()
    gt_seq = gt_seq.squeeze()
    os.makedirs(save_path, exist_ok=True)

    if save_grays:
        os.makedirs(osp.join(save_path, 'pred'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'targets'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(pred_seq, gt_seq)):

            plt.imsave(osp.join(save_path, 'pred', f'{i}.png'), pred, cmap='gray', vmax=1.0, vmin=0.0)
            plt.imsave(osp.join(save_path, 'targets', f'{i}.png'), gt, cmap='gray', vmax=1.0, vmin=0.0)

    pred_seq = pred_seq * 255
    pred_seq = pred_seq.astype(np.int16)
    gt_seq = gt_seq * 255
    gt_seq = gt_seq.astype(np.int16)
    

    colored_pred = np.array([gray2color(pred_seq[i]) for i in range(len(pred_seq))], dtype=np.float64)
    colored_gt =  np.array([gray2color(gt_seq[i]) for i in range(len(gt_seq))],dtype=np.float64)
    
    if save_colored:
        os.makedirs(osp.join(save_path, 'pred_colored'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'targets_colored'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(colored_pred, colored_gt)):
            plt.imsave(osp.join(save_path, 'pred_colored', f'{i}.png'), pred)
            plt.imsave(osp.join(save_path, 'targets_colored', f'{i}.png'), gt)


    clip = ImageSequenceClip(list(colored_pred * 255), fps=4)
    clip.write_gif(osp.join(save_path, f"pred_{model_name}.gif"), fps=4)

    clip = ImageSequenceClip(list(colored_gt * 255), fps=4)
    clip.write_gif(osp.join(save_path, f"gt_{model_name}.gif"), fps=4)

def display_sevir(
    ground_truth,
    predictions_dict,
    output_path=None,
    skip_after_5=False,
    args=None
):
    """
    Display colored visualizations using gray2color and save as a grid of images.
    Ground truth and model predictions are shown with per-row labels centered.

    Args:
        ground_truth (ndarray): [T, 1, H, W] or [T, H, W] ground truth.
        predictions_dict (dict): {model_name: prediction}, each shape [T, 1, H, W] or [T, H, W].
        output_path (str): Optional path to save the figure.
        skip_after_5 (bool): Skip alternate frames after the 5th.
        gray2color (function): Function to convert a grayscale frame to RGB.
    """
    assert gray2color is not None, "You must pass a gray2color function!"

    PIXEL_SCALE = 255.0

    def prepare_colored_seq(seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.detach().cpu().numpy()
        if seq.ndim == 4:  # [T, 1, H, W]
            seq = seq.squeeze(1)
        seq = (seq * PIXEL_SCALE).astype(np.int16)
        return np.array([gray2color(seq[i]) for i in range(seq.shape[0])], dtype=np.float64)
    if ground_truth.shape[0] > 10:
        skip_after_5 = True
    total_frames = ground_truth.shape[0]
    if skip_after_5:
        frame_indices = list(range(5)) + list(range(6, total_frames, 2))
    else:
        frame_indices = list(range(total_frames))
    time_labels = [f"{i * args.time_resolution} min" for i in frame_indices]

    # Prepare rows: first is ground truth
    rows = [prepare_colored_seq(ground_truth)[frame_indices]]
    row_labels = ["Ground Truth"]
    for model_name, pred in predictions_dict.items():
        rows.append(prepare_colored_seq(pred)[frame_indices])
        row_labels.append(model_name)

    n_rows = len(rows)
    n_cols = len(frame_indices)

    # Add one column at start for row titles
    fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=((n_cols + 1) * 2, n_rows * 2))

    for row_idx, row_images in enumerate(rows):
        for col_idx in range(n_cols + 1):
            ax = axes[row_idx, col_idx]
            ax.axis('off')

            if col_idx == 0:
                # Leftmost cell: write model name
                ax.text(0.5, 0.5, row_labels[row_idx],
                        fontsize=12, fontweight='bold', color='black',
                        ha='center', va='center', transform=ax.transAxes)
            else:
                img = row_images[col_idx - 1]
                ax.imshow(img)

                # Add time label on first row
                if row_idx == 0:
                    ax.set_title(time_labels[col_idx - 1], fontsize=10)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    plt.show()

from pysteps.verification.detcatscores import det_cat_fct
from pysteps.verification.detcontscores import det_cont_fct
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
def compute_all_metrics(stored_events, loaded_models, thresholds, device="cuda", chunk_size=2,args=None):
    """
    Computes MSE, MAE, PCC, CSI, and FAR for each model across precipitation levels and time steps.
    Performs inference once and calculates all metrics afterward.

    Args:
        stored_events (dict): Dictionary of precipitation events at different percentile levels.
        loaded_models (dict): Dictionary of models where keys are model names and values are dictionaries 
                              containing 'tokenizer' and 'predictor'.
        thresholds (list): List of thresholds (e.g., [1, 2, 8]) for which to calculate CSI and FAR.
        device (str): Device to perform computations (default: 'cuda').
        chunk_size (int): Number of events to process at once for inference.

    Returns:
        tuple:
            dict: MSE, MAE, PCC per level for each model, per time step.
            dict: Aggregated MSE, MAE, PCC across all levels for each model, per time step.
            dict: CSI and FAR for each model at specified thresholds.
    """
    per_level_results = {}
    aggregated_results = {}
    csi_far_results = {}

    for model_name, model_components in loaded_models.items():
        print("processing model ",model_name)
        if model_name.startswith("diffcast"):
            model = model_components.get("model")
            model.to(device)
        else:
            tokenizer = model_components.get("tokenizer")
            predictor = model_components.get("predictor")
        
        # Move models to device
            if tokenizer:
                tokenizer.to(device)
            if predictor:
                predictor.to(device)

        model_per_level = {}
        aggregated_mse = []
        aggregated_mae = []
        aggregated_pcc = []

        combined_events = []
        combined_predictions = []

        for level, events in stored_events.items():
            events = events.to(device)
            print("processing level",level)
            B, T, C, H, W = events.shape
            assert C == 1 and H == 128 and W == 128, "Event dimensions must match (B, T, 1, 128, 128)."

            num_events = events.shape[0]
            mse_per_time_step = np.zeros(T)
            mae_per_time_step = np.zeros(T)
            pcc_per_time_step = np.zeros(T)

            all_predictions = []
            all_events = []

            # Process in chunks
            with torch.no_grad():
                for start_idx in range(0, num_events, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_events)
                    events_chunk = events[start_idx:end_idx]  # Shape: [chunk_size, T, 1, 128, 128]
            
                    if model_name.startswith("diffcast"):
                        print("events chunk shape is ",events_chunk.shape)
                        preds_chunk = model.sample(events_chunk[:, :args.context_length], args.segment_length-args.context_length)[0]
                        preds_chunk = torch.cat([events_chunk[:,:args.context_length],preds_chunk],dim=1)


                    else:
    
                        preds_chunk = generate(events_chunk, predictor, tokenizer,args)


                    # Clamp predictions for consistency
                    preds_chunk = torch.clamp(preds_chunk, 0, 1)

                    # Move predictions and events to CPU
                    all_predictions.append(preds_chunk.cpu().detach().numpy()*args.SCALE_FACTOR)
                    all_events.append(events_chunk.cpu().detach().numpy()*args.SCALE_FACTOR)

            # Concatenate all predictions and events for the level
            all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: [B, T, 1, 128, 128]
            all_events = np.concatenate(all_events, axis=0)  # Shape: [B, T, 1, 128, 128]

            combined_events.append(all_events)
            combined_predictions.append(all_predictions)

            # Compute MSE, MAE, and PCC per time step for the level
            for t in range(T):
                mse = mean_squared_error(all_events[:, t].flatten(), all_predictions[:, t].flatten())
                mae = mean_absolute_error(all_events[:, t].flatten(), all_predictions[:, t].flatten())
                pcc = det_cont_fct(all_events[:, t], all_predictions[:, t], scores=["corr_p"], thr=0.1)["corr_p"]
                mse_per_time_step[t] = mse
                mae_per_time_step[t] = mae
                pcc_per_time_step[t] = pcc

            # Store per-level results
            model_per_level[level] = {
                "MSE": mse_per_time_step.tolist(),
                "MAE": mae_per_time_step.tolist(),
                "PCC": pcc_per_time_step.tolist(),
            }

            # Accumulate for overall aggregation
            aggregated_mse.append(mse_per_time_step)
            aggregated_mae.append(mae_per_time_step)
            aggregated_pcc.append(pcc_per_time_step)

        # Aggregate across all levels
        aggregated_mse = np.mean(aggregated_mse, axis=0)
        aggregated_mae = np.mean(aggregated_mae, axis=0)
        aggregated_pcc = np.mean(aggregated_pcc, axis=0)
        aggregated_results[model_name] = {
            "MSE": aggregated_mse.tolist(),
            "MAE": aggregated_mae.tolist(),
            "PCC": aggregated_pcc.tolist(),
        }

        # Store per-level results for this model
        per_level_results[model_name] = model_per_level
        print("computing csi and far")
        # Combine events and predictions for CSI and FAR calculation
        combined_events = np.concatenate(combined_events, axis=0)  # Combine all events across levels
        combined_predictions = np.concatenate(combined_predictions, axis=0)  # Combine all predictions
        print(combined_events.shape)
        # Compute CSI and FAR for the combined data
        T = combined_events.shape[1]
        csi_far_results[model_name] = {f"Threshold {threshold}": {"CSI": [], "FAR": []} for threshold in thresholds}

        for t in range(T):
            for threshold in thresholds:
                scores_cat = det_cat_fct(
                    combined_events[:, t], combined_predictions[:, t], threshold
                )
                csi_far_results[model_name][f"Threshold {threshold}"]["CSI"].append(scores_cat["CSI"])
                csi_far_results[model_name][f"Threshold {threshold}"]["FAR"].append(scores_cat["FAR"])

    return per_level_results, aggregated_results, csi_far_results

def get_diff_model(path,kwargs,use_BlockGPT=False,args=None):

  
        combined_state_dict = torch.load(path)
        combined_state_dict = combined_state_dict['model']
        backbone_state_dict = {k.replace("backbone_net.", ""): v for k, v in combined_state_dict.items() if k.startswith("backbone_net.")}
        main_model_state_dict = {k: v for k, v in combined_state_dict.items() if not k.startswith("backbone_net.")}
        if use_BlockGPT:
            from models.blockGPTBackbone import get_model
            from models.blockGPTBackbone import BlockGPTBackboneConfig
            config_path = kwargs.get('config_path', None)
            with open(config_path) as f:
                kwargs = json.load(f)
                config = BlockGPTBackboneConfig(**kwargs)
                backbone = get_model(config)        
        else:
            from models.phydnet import get_model
            backbone= get_model(**kwargs)

        from models.diffcast import get_model
        kwargs = {
                'img_channels' :1,
                'dim' : 64,
                'dim_mults' : (1,2,4,8),
                'T_in': args.context_length,
                'T_out':  args.segment_length-args.context_length,
                'sampling_timesteps': 250,
            }
        diff_model = get_model(**kwargs)
        backbone.load_state_dict(backbone_state_dict)
        diff_model.load_state_dict(main_model_state_dict)
        diff_model.load_backbone(backbone)
        return diff_model


#specififc functions 
def display_knmi(ground_truth, predictions_dict, output_path=None, skip_alternate=False):
    """
    Display the full event (time frames 0 to 24) with ground truth and predictions from multiple models.
    Optionally skip alternate frames.

    Parameters
    ----------
    ground_truth : ndarray
        NumPy array of ground truth images for the full time frames. Shape: [T, 1, H, W].
    predictions_dict : dict
        Dictionary of model predictions. Keys are model names (str),
        values are NumPy arrays with shape [T, 1, H, W].
    output_path : str, optional
        Full path to save the plot. If None, the plot will not be saved.
    skip_alternate : bool, optional
        If True, skips every other frame after the first 5. Default is False.
    """
    # Validate input dimensions
    for model_name, pred in predictions_dict.items():
        assert pred.shape == ground_truth.shape, f"Prediction for model '{model_name}' must match ground truth shape."
        assert pred.ndim == 4, f"Prediction for model '{model_name}' must have shape [T, 1, H, W]."

    assert ground_truth.ndim == 4, "Ground truth must have shape [T, 1, H, W]."
    #turn on skip_alternate if T dimension is greater than 10
    if ground_truth.shape[0] > 10:
        skip_alternate = True
    # Apply skip logic
    T = ground_truth.shape[0]
    if skip_alternate:
        indices = list(range(5)) + list(range(5, T, 2))
    else:
        indices = list(range(T))

    ground_truth = ground_truth[indices].squeeze(1)  # [N, H, W]
    predictions_dict = {k: v[indices].squeeze(1) for k, v in predictions_dict.items()}  # [N, H, W] for each model

    n_steps = len(indices)
    n_models = len(predictions_dict)
    n_rows = 1 + n_models

    fig, axes = plt.subplots(n_rows, n_steps, figsize=(2.5 * n_steps, 3 * n_rows), constrained_layout=True)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)


    # Plot ground truth
    for i, ax in enumerate(axes[0]):
        plot_precip_field(ground_truth[i] * 40, ax=ax, title=f"GT t={indices[i]}", colorbar=False)
        ax.set_title(f"GT t={indices[i]}", fontsize=10, wrap=True)

    # Plot predictions
    for row, (model_name, prediction) in enumerate(predictions_dict.items(), start=1):
       
        for i, ax in enumerate(axes[row]):
            plot_precip_field(prediction[i] * 40, ax=ax, colorbar=False)
            ax.set_title(f"{model_name} t={indices[i]}", fontsize=10, wrap=True)

    # Super title
    plt.suptitle("Ground Truth and Model Predictions", fontsize=16)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not output_path.lower().endswith(".png"):
            output_path += ".png"
        plt.savefig(output_path, format="png", bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    plt.show()

import yaml

def load_model_config(args):
    with open(args.eval_checkpoints, "r") as f:
        config = yaml.safe_load(f)

    loaded_models = {}

    # Load BlockGPT models
    for model_info in config.get("blockGPT_models", []):
        tokenizer, predictor = get_tokenizer_and_model_pair(
            predictor_name="blockGPT",
            config_name=model_info["config"],
            predictor_dict=model_info["checkpoint"],
            vqgan_name=model_info["vqgan_type"],
            vqgan_config=config["vqgan"]["config"],
            vqgan_checkpoint=config["vqgan"]["checkpoint"],
            args=args
        )
        loaded_models[model_info["name"]] = {
            "tokenizer": tokenizer,
            "predictor": predictor
        }

    # Load Diffcast models
    if "diffcast_models" in config:
        for model_info in config.get("diffcast_models", []):
            if model_info["use_BlockGPT"]:
                kwargs = {
             "config_path" : model_info["config_path"]
        
                }
            else:
                kwargs = {
                    "in_shape": tuple(config["kwargs"]["in_shape"]),
                    "T_in": args.context_length,
                    "T_out": args.segment_length - args.context_length,
                    "device": args.device
                }
            diff_model = get_diff_model(
                path=model_info["checkpoint"],
                kwargs=kwargs,
                use_BlockGPT=model_info.get("use_BlockGPT", False),
                args=args
            )
            loaded_models[model_info["name"]] = {
                "model": diff_model
            }

    return loaded_models

def main():
    args = parse_args()
    eval_dataloader = get_eval_dataloader(args)
    if args.dataset_name == 'knmi' or args.dataset_name == 'knmi_5mins':
       thresholds = [1, 2, 8]
    elif args.dataset_name == 'sevir':
       thresholds = [16, 74, 133, 160, 181, 219]
    else:
         raise ValueError("Unsupported dataset name. Use 'knmi', 'knmi_5mins', or 'sevir'.")
    output_folder = "Results/Evaluations" 
    output_folder = os.path.join(output_folder, args.dataset_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"{output_folder}_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)

    loaded_models = load_model_config(args)

    avg_precip_test = get_average_precipitations(eval_dataloader)
    categorized_indices = get_categorized_indices([20,40,60,80,95], avg_precip_test)
    stored_events = store_all_events(eval_dataloader,categorized_indices,args.batch_size)
 

    output_folder = "Results/Evaluations" 
    output_folder = os.path.join(output_folder, args.dataset_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"{output_folder}_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)

    # Save model names to evaluated_models.txt
    model_list_path = os.path.join(output_folder, "evaluated_models.txt")
    with open(model_list_path, "w") as f:
        for model_name in loaded_models.keys():
            f.write(model_name + "\n")

    output_folder = os.path.join(output_folder, f"temp_res_{args.time_resolution}")
    plots_path = os.path.join(output_folder, "predictions")
    plot_predictions_per_level_sampled(stored_events, loaded_models, plots_path, num_samples=args.batch_size, chunk_size=args.batch_size,args=args)

    results = compute_all_metrics(stored_events, loaded_models, thresholds, device=args.device, chunk_size=args.batch_size,args=args)
    import pickle

   #Save to file
    picklefolder = os.path.join(output_folder, "PickledResults")
    os.makedirs(picklefolder, exist_ok=True)
    pickle_path = os.path.join(picklefolder, "results_summary.pkl")
    with open(pickle_path, "wb") as file:
      pickle.dump(results, file)
    
if __name__ == "__main__":
    main()