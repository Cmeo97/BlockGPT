import os
import os.path as osp
from typing import List, Union, Dict, Sequence
from math import ceil
import numpy as np
import numpy.random as nprand
import datetime
import pandas as pd
import h5py 
import cv2

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.nn.functional import avg_pool2d
from torchvision import transforms 

from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
#from moviepy.editor import ImageSequenceClip


#TODO: must understand what color, gray2color are doing, right now they are copies of SEVIR.

def change_layout_np(data,
                     in_layout='NHWT', out_layout='NHWT',
                     ret_contiguous=False):
    # first convert to 'NHWT'
    if in_layout == 'NHWT':
        pass
    elif in_layout == 'NTHW':
        data = np.transpose(data,
                            axes=(0, 2, 3, 1))
    elif in_layout == 'NWHT':
        data = np.transpose(data,
                            axes=(0, 2, 1, 3))
    elif in_layout == 'NTCHW':
        data = data[:, :, 0, :, :]
        data = np.transpose(data,
                            axes=(0, 2, 3, 1))
    elif in_layout == 'NTHWC':
        data = data[:, :, :, :, 0]
        data = np.transpose(data,
                            axes=(0, 2, 3, 1))
    elif in_layout == 'NTWHC':
        data = data[:, :, :, :, 0]
        data = np.transpose(data,
                            axes=(0, 3, 2, 1))
    elif in_layout == 'TNHW':
        data = np.transpose(data,
                            axes=(1, 2, 3, 0))
    elif in_layout == 'TNCHW':
        data = data[:, :, 0, :, :]
        data = np.transpose(data,
                            axes=(1, 2, 3, 0))
    else:
        raise NotImplementedError

    if out_layout == 'NHWT':
        pass
    elif out_layout == 'NTHW':
        data = np.transpose(data,
                            axes=(0, 3, 1, 2))
    elif out_layout == 'NWHT':
        data = np.transpose(data,
                            axes=(0, 2, 1, 3))
    elif out_layout == 'NTCHW':
        data = np.transpose(data,
                            axes=(0, 3, 1, 2))
        data = np.expand_dims(data, axis=2)
    elif out_layout == 'NTHWC':
        data = np.transpose(data,
                            axes=(0, 3, 1, 2))
        data = np.expand_dims(data, axis=-1)
    elif out_layout == 'NTWHC':
        data = np.transpose(data,
                            axes=(0, 3, 2, 1))
        data = np.expand_dims(data, axis=-1)
    elif out_layout == 'TNHW':
        data = np.transpose(data,
                            axes=(3, 0, 1, 2))
    elif out_layout == 'TNCHW':
        data = np.transpose(data,
                            axes=(3, 0, 1, 2))
        data = np.expand_dims(data, axis=2)
    else:
        raise NotImplementedError
    if ret_contiguous:
        data = data.ascontiguousarray()
    return data

def change_layout_torch(data,
                        in_layout='NHWT', out_layout='NHWT',
                        ret_contiguous=False):
    # first convert to 'NHWT'
    if in_layout == 'NHWT':
        pass
    elif in_layout == 'NTHW':
        data = data.permute(0, 2, 3, 1)
    elif in_layout == 'NTCHW':
        data = data[:, :, 0, :, :]
        data = data.permute(0, 2, 3, 1)
    elif in_layout == 'NTHWC':
        data = data[:, :, :, :, 0]
        data = data.permute(0, 2, 3, 1)
    elif in_layout == 'TNHW':
        data = data.permute(1, 2, 3, 0)
    elif in_layout == 'TNCHW':
        data = data[:, :, 0, :, :]
        data = data.permute(1, 2, 3, 0)
    else:
        raise NotImplementedError

    if out_layout == 'NHWT':
        pass
    elif out_layout == 'NTHW':
        data = data.permute(0, 3, 1, 2)
    elif out_layout == 'NTCHW':
        data = data.permute(0, 3, 1, 2)
        data = torch.unsqueeze(data, dim=2)
    elif out_layout == 'NTHWC':
        data = data.permute(0, 3, 1, 2)
        data = torch.unsqueeze(data, dim=-1)
    elif out_layout == 'TNHW':
        data = data.permute(3, 0, 1, 2)
    elif out_layout == 'TNCHW':
        data = data.permute(3, 0, 1, 2)
        data = torch.unsqueeze(data, dim=2)
    else:
        raise NotImplementedError
    if ret_contiguous:
        data = data.contiguous()
    return data



class KNMIDataloader:

    def __init__(self,
                knmi_data_dir: str,
                seq_len:int =9,
                raw_seq_len:int = 9,
                sample_mode:str='sequent',
                stride:int=1,
                batch_size:int=1,
                layout:str='NTHW',
                num_shard:int=1,
                rank:int=0,
                split_mode:str='uneven',
                output_type=np.float32,
                ):
        self._hdf_files = None
        self.knmi_data_dir = knmi_data_dir
        self.data_shape = (128,128)

        self.raw_seq_len = raw_seq_len
        assert seq_len <= self.raw_seq_len, f'seq_len must not be larger than raw_seq_len = {raw_seq_len}, got {seq_len}.'
        self.seq_len = seq_len
        assert sample_mode in ['random', 'sequent'], f'Invalid sample_mode = {sample_mode}, must be \'random\' or \'sequent\'.'
        self.sample_mode = sample_mode
        self.stride = stride
        self.batch_size = batch_size
        valid_layout = ('NHWT', 'NTHW', 'NTCHW', 'NTHWC', 'TNHW', 'TNCHW')
        if layout not in valid_layout:
            raise ValueError(f'Invalid layout = {layout}! Must be one of {valid_layout}.')        
        self.layout = layout
        self.num_shard = num_shard
        self.rank = rank
        valid_split_mode = ('ceil', 'floor', 'uneven')
        if split_mode not in valid_split_mode:
            raise ValueError(f'Invalid split_mode: {split_mode}! Must be one of {valid_split_mode}.')
        self.split_mode = split_mode
        self._samples = None
        self._hdf_files = {}
        self.output_type = output_type

        #TODO: implement self._compute_samples(), and reset
        self._compute_samples()
        self._open_files()
        #self.reset()
        


        #self._open_files()
    def _compute_samples(self):
        """
        Computes the list of samples in the KNMI dataset to be used. This sets self._samples.
        Each sample is a video from a specific HDF5 file with a given index.
        """
        # List all HDF5 files in the directory (assuming they follow 'videos_batch_{i}.h5' naming)
        h5_filenames = sorted([f for f in os.listdir(self.knmi_data_dir) if f.startswith('videos_batch_') and f.endswith('.h5')])

        # Initialize a list to store sample information
        sample_list = []

        # Iterate over each HDF5 file and record the index of each video within the file
        for fname in h5_filenames:
            file_path = os.path.join(self.knmi_data_dir, fname)
            with h5py.File(file_path, 'r') as h5_file:
                num_videos = h5_file['videos'].shape[0]  # Number of videos in the batch

                # Create an entry for each video in this file
                for idx in range(num_videos):
                    sample_list.append({'filename': fname, 'index': idx})

        # Convert the list of samples into a DataFrame for consistency and store in self._samples
        self._samples = pd.DataFrame(sample_list)

    def _open_files(self, verbose=False):
        """
        Opens HDF5 files in the KNMI dataset directory
        """
        import os
        import h5py
        import numpy as np

        # Assuming self.knmi_data_dir is the directory containing all the KNMI HDF5 files
        hdf_filenames = [f for f in os.listdir(self.knmi_data_dir) if f.endswith('.h5') or f.endswith('.hdf5')]
        hdf_filenames = sorted(hdf_filenames)  # Optional: sort the filenames if order matters

        self._hdf_files = {}
        for f in hdf_filenames:
            if verbose:
                print('Opening HDF5 file for reading', f)
            file_path = os.path.join(self.knmi_data_dir, f)
            self._hdf_files[f] = h5py.File(file_path, 'r')
    def close(self):
        """
        Closes all open file handles
        """
        for f in self._hdf_files:
            self._hdf_files[f].close()
        self._hdf_files = {}
    
    @property
    def num_seq_per_event(self):
        return 1 + (self.raw_seq_len - self.seq_len) // self.stride
    
    @property
    def total_num_seq(self):
        """
        The total number of sequences within each shard.
        Notice that it is not the product of `self.num_seq_per_event` and `self.total_num_event`.
        """
        return int(self.num_seq_per_event * self.num_event)
    
    @property
    def total_num_event(self):
        """
        The total number of events in the whole dataset, before split into different shards.
        """
        return int(self._samples.shape[0])
    
    @property
    def start_event_idx(self):
        """
        The event idx used in certain rank should satisfy event_idx >= start_event_idx
        """
        return self.total_num_event // self.num_shard * self.rank
    
    @property
    def end_event_idx(self):
        """
        The event idx used in certain rank should satisfy event_idx < end_event_idx

        """
        if self.split_mode == 'ceil':
            _last_start_event_idx = self.total_num_event // self.num_shard * (self.num_shard - 1)
            _num_event = self.total_num_event - _last_start_event_idx
            return self.start_event_idx + _num_event
        elif self.split_mode == 'floor':
            return self.total_num_event // self.num_shard * (self.rank + 1)
        else:  # self.split_mode == 'uneven':
            if self.rank == self.num_shard - 1:  # the last process
                return self.total_num_event
            else:
                return self.total_num_event // self.num_shard * (self.rank + 1)
            
    @property
    def num_event(self):
        """
        The number of events split into each rank
        """
        return self.end_event_idx - self.start_event_idx
    def _read_data(self, row, data):
        """
        Reads data from a batch of videos in KNMI HDF5 files into the data dictionary.
        Each entry in the data dictionary corresponds to a batch of videos, updated iteratively.

        Parameters
        ----------
        row : pandas.Series
            Contains the filename and index for the HDF5 file.
            Expected fields: 'filename', 'index'.
        data : dict
            A dictionary to store the video data with shape (batch_size, time, 128, 128).

        Returns
        -------
        data : dict
            Updated data dictionary with new video data appended. Final shape will be
            (batch_size+1, time, 128, 128).
        """
        # Extract filename and index from the row
        fname = row['filename']
        idx = row['index']
        
        # Access the already-opened HDF5 file from self._hdf_files and read the video at the specified index
        video_data = self._hdf_files[fname]['videos'][idx:idx + 1]  # Shape will be (1, time, 128, 128)
        #print(video_data.shape)
        # Append this batch to the data dictionary
        data['videos'] = np.concatenate((data['videos'], video_data), axis=0) if 'videos' in data else video_data

        return data
    @property
    def sample_count(self):
        """
        Record how many times self.__next__() is called.
        """
        return self._sample_count

    def inc_sample_count(self):
        self._sample_count += 1

    @property
    def curr_event_idx(self):
        return self._curr_event_idx

    @property
    def curr_seq_idx(self):
        """
        Used only when self.sample_mode == 'sequent'
        """
        return self._curr_seq_idx

    def set_curr_event_idx(self, val):
        self._curr_event_idx = val

    def set_curr_seq_idx(self, val):
        """
        Used only when self.sample_mode == 'sequent'
        """
        self._curr_seq_idx = val

    def reset(self):
        self.set_curr_event_idx(val=self.start_event_idx)
        self.set_curr_seq_idx(0)
        self._sample_count = 0
        shuffle = None

    def __len__(self):
        """
        Used only when self.sample_mode == 'sequent'
        """
        return self.total_num_seq // self.batch_size
    
    @property
    def use_up(self):
        """
        Check if dataset is used up in 'sequent' mode.
        """
        if self.sample_mode == 'random':
            return False
        else:   # self.sample_mode == 'sequent'
            # compute the remaining number of sequences in current event
            curr_event_remain_seq = self.num_seq_per_event - self.curr_seq_idx
            all_remain_seq = curr_event_remain_seq + (
                        self.end_event_idx - self.curr_event_idx - 1) * self.num_seq_per_event
            if self.split_mode == "floor":
                # This approach does not cover all available data, but avoid dealing with masks
                return all_remain_seq < self.batch_size
            else:
                return all_remain_seq <= 0
    def _load_event_batch(self, event_idx, event_batch_size):
        """
        Loads a selected batch of events (videos) into memory for the KNMI dataset.

        Parameters
        ----------
        event_idx : int
            Start index for loading the batch.
        event_batch_size : int
            Number of events (videos) to load in the batch.

        Returns
        -------
        event_batch : np.ndarray
            Batch of video data with shape (batch_size, time, height, width).
            If padding is applied, the batch is padded with zeros to maintain consistent size.
        """
        event_idx_slice_end = event_idx + event_batch_size
        pad_size = 0

        # Adjust slice end and calculate padding if necessary
        if event_idx_slice_end > len(self._samples):
            pad_size = event_idx_slice_end - len(self._samples)
            event_idx_slice_end = len(self._samples)

        # Get the batch of samples
        pd_batch = self._samples.iloc[event_idx:event_idx_slice_end]

        # Initialize data dictionary
        data = {}
        for _, row in pd_batch.iterrows():
            data = self._read_data(row, data)

        # Handle padding if required
        if pad_size > 0:
            pad_shape = [pad_size, ] + list(data['videos'].shape[1:])
            data_padded = np.concatenate(
                (data['videos'].astype(self.output_type), np.zeros(pad_shape, dtype=self.output_type)),
                axis=0
            )
            return data_padded
        else:
            return data['videos'].astype(self.output_type)
        


    def __iter__(self):
        return self
    


    def __next__(self):
        if self.sample_mode == 'random':
            self.inc_sample_count()
            ret_dict = self._random_sample()
        else:
            if self.use_up:
                raise StopIteration
            else:
                self.inc_sample_count()
                ret_dict = self._sequent_sample()
        ret_dict = self.data_dict_to_tensor(data_dict=ret_dict,
                                            data_types=self.data_types)

        return ret_dict


    @staticmethod
    def data_dict_to_tensor(data_dict):
        """
        Convert each element in data_dict to torch.Tensor (copy without grad).

        Parameters
        ----------
        data_dict : dict
            A dictionary where values are either numpy arrays or torch tensors.

        Returns
        -------
        ret_dict : dict
            A dictionary with all elements converted to torch tensors.
        """
        ret_dict = {}
        for key, data in data_dict.items():
            if isinstance(data, torch.Tensor):
                ret_dict[key] = data.detach().clone()
            elif isinstance(data, np.ndarray):
                ret_dict[key] = torch.from_numpy(data)
            else:
                raise ValueError(f"Invalid data type: {type(data)}. Should be torch.Tensor or np.ndarray")
        return ret_dict
    




    def _random_sample(self):
        """
        Randomly samples a batch of sequences from the KNMI dataset.

        Returns
        -------
        ret_dict : dict
            A dictionary with a single key 'videos'.
            If self.preprocess == False:
                ret_dict['videos'].shape == (batch_size, height, width, seq_len)
        """
        num_sampled = 0
        event_idx_list = nprand.randint(low=self.start_event_idx,
                                        high=self.end_event_idx,
                                        size=self.batch_size)
        seq_idx_list = nprand.randint(low=0,
                                    high=self.num_seq_per_event,
                                    size=self.batch_size)
        seq_slice_list = [slice(seq_idx * self.stride,
                                seq_idx * self.stride + self.seq_len)
                        for seq_idx in seq_idx_list]
        ret_dict = {'videos': None}

        while num_sampled < self.batch_size:
            # Load a single event (video)
            event = self._load_event_batch(event_idx=event_idx_list[num_sampled],
                                        event_batch_size=1)

            # Extract the desired sequence slice
            sampled_seq = event[:, :, :, seq_slice_list[num_sampled]]  # Keep batch dimension for concatenation

            # Append to the batch
            if ret_dict['videos'] is None:
                ret_dict['videos'] = sampled_seq
            else:
                ret_dict['videos'] = np.concatenate((ret_dict['videos'], sampled_seq), axis=0)

            num_sampled += 1

        return ret_dict

    def _sequent_sample(self):
        """
        Sequentially samples a batch of sequences from the KNMI dataset.

        Returns
        -------
        ret_dict : dict
            - Contains a single key, 'videos', with shape (batch_size, height, width, seq_len).
            - Contains a key, 'mask', which is a list of bools indicating whether the data is real or padded.
        """
        assert not self.use_up, 'Data loader used up! Reset it to reuse.'

        event_idx = self.curr_event_idx
        seq_idx = self.curr_seq_idx
        num_sampled = 0
        sampled_idx_list = []  # List of (event_idx, seq_idx) records

        # Gather indices for the samples
        while num_sampled < self.batch_size:
            sampled_idx_list.append({'event_idx': event_idx, 'seq_idx': seq_idx})
            seq_idx += 1
            if seq_idx >= self.num_seq_per_event:
                event_idx += 1
                seq_idx = 0
            num_sampled += 1

        # Determine the range of events to load
        start_event_idx = sampled_idx_list[0]['event_idx']
        event_batch_size = sampled_idx_list[-1]['event_idx'] - start_event_idx + 1

        # Load the batch of events
        event_batch = self._load_event_batch(event_idx=start_event_idx, event_batch_size=event_batch_size)

        # Initialize return dictionary
        ret_dict = {"videos": None, "mask": []}
        all_no_pad_flag = True

        # Iterate through sampled indices to extract the sequences
        for sampled_idx in sampled_idx_list:
            batch_slice = [sampled_idx['event_idx'] - start_event_idx]  # Keep batch dimension
            seq_slice = slice(sampled_idx['seq_idx'] * self.stride,
                            sampled_idx['seq_idx'] * self.stride + self.seq_len)

            # Extract the sequence
            sampled_seq = event_batch[batch_slice, :, :, seq_slice]
            if ret_dict["videos"] is None:
                ret_dict["videos"] = sampled_seq
            else:
                ret_dict["videos"] = np.concatenate((ret_dict["videos"], sampled_seq), axis=0)

            # Add mask
            no_pad_flag = sampled_idx['event_idx'] < self.end_event_idx
            if not no_pad_flag:
                all_no_pad_flag = False
            ret_dict["mask"].append(no_pad_flag)

        # Set mask to None if no padded data items
        if all_no_pad_flag:
            ret_dict["mask"] = None

        # Update current indices
        self.set_curr_event_idx(event_idx)
        self.set_curr_seq_idx(seq_idx)

        return ret_dict
    
    def _idx_sample(self, index):
        """
        Samples a batch of sequences by index for the KNMI dataset.

        Parameters
        ----------
        index : int
            The index of the batch to sample.

        Returns
        -------
        ret_dict : dict
            A dictionary containing:
            - 'videos': torch.Tensor with shape (batch_size, height, width, seq_len)
            - Other preprocessed or downsampled data if applicable.
        """
        # Calculate event and sequence indices
        event_idx = (index * self.batch_size) // self.num_seq_per_event
        seq_idx = (index * self.batch_size) % self.num_seq_per_event
        num_sampled = 0
        sampled_idx_list = []  # List of (event_idx, seq_idx) records

        # Collect indices for sampling
        while num_sampled < self.batch_size:
            sampled_idx_list.append({'event_idx': event_idx, 'seq_idx': seq_idx})
            seq_idx += 1
            if seq_idx >= self.num_seq_per_event:
                event_idx += 1
                seq_idx = 0
            num_sampled += 1

        # Determine the range of events to load
        start_event_idx = sampled_idx_list[0]['event_idx']
        event_batch_size = sampled_idx_list[-1]['event_idx'] - start_event_idx + 1

        # Load the batch of events
        event_batch = self._load_event_batch(event_idx=start_event_idx, event_batch_size=event_batch_size)
      #  print(event_batch.shape)
        # Initialize return dictionary
        ret_dict = {"videos": None}

        # Extract sequences based on sampled indices
        for sampled_idx in sampled_idx_list:
            batch_slice = [sampled_idx['event_idx'] - start_event_idx]  # Keep batch dimension
            seq_slice = slice(sampled_idx['seq_idx'] * self.stride,
                            sampled_idx['seq_idx'] * self.stride + self.seq_len)
            # Correct slicing: batch, time, height, width
            sampled_seq = event_batch[batch_slice, seq_slice, :, :]
           # print(f"Sampled sequence shape: {sampled_seq.shape}")  # Should be [1, 9, 128, 128]

            if ret_dict["videos"] is None:
                ret_dict["videos"] = sampled_seq
            else:
                ret_dict["videos"] = np.concatenate((ret_dict["videos"], sampled_seq), axis=0)

        # Convert data to torch tensors
        ret_dict = self.data_dict_to_tensor(data_dict=ret_dict)
       # print(ret_dict["videos"].shape)

        return ret_dict

class KNMITorchDataset(TorchDataset):

    def __init__(self,
                 dataset_dir: str,
                 seq_len: int = 9,
                 img_size: int = 128,
                 raw_seq_len: int = 9,
                 sample_mode: str = "sequent",
                 stride: int = 1,
                 batch_size: int = 1,
                 layout: str = "NTHW",
                 num_shard: int = 1,
                 rank: int = 0,
                 split_mode: str = "uneven",
                 output_type = np.float32,
                 verbose: bool = False,
                 debug: bool = False):
        

        super(KNMITorchDataset, self).__init__()
        self.layout = layout
        self.img_size = img_size
        self.debug = debug
        self.knmi_dataloader = KNMIDataloader(
            knmi_data_dir=dataset_dir,
            seq_len=seq_len,
            raw_seq_len=raw_seq_len,
            sample_mode=sample_mode,
            stride=stride,
            batch_size=batch_size,
            layout=layout,
            num_shard=num_shard,
            rank=rank,
            split_mode=split_mode,
            output_type=output_type)
        
        self.transform = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        # transforms.ToTensor(),
                        # trans.Lambda(lambda x: x/255.0),
                        # transforms.Normalize(mean=[0.5], std=[0.5]),
                        # trans.RandomCrop(data_config["img_size"]),

                    ])
        
    def __getitem__(self,index):
            data_dict = self.knmi_dataloader._idx_sample(index)
            data = data_dict['videos']
            data = self.transform(data)
            return data
        
    def __len__(self):
            if self.debug:
                return min(25, self.knmi_dataloader.__len__())
            else:
                return self.knmi_dataloader.__len__()

        #TODO: not implementing the collate funciton since it is not required in outer batch size = 1
            

        

    def collate_fn(self, batch):
        """
        Since each item from `__getitem__` is already a batch,
        and the outer batch size is > 1, we need to concatenate the batches.
        """
        # When outer_batch_size > 1, batch is a list of tensors.
        # Each tensor is a batch from __getitem__.
        return torch.utils.data.dataloader.default_collate(batch)
        
    def get_torch_dataloader(self,
                             outer_batch_size=1,
                             collate_fn=None,
                             num_workers=1):
        # TODO: num_workers > 1
        
        if collate_fn is None:
            collate_fn=self.collate_fn
       
        dataloader = DataLoader(
            dataset=self,
            batch_size=outer_batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=num_workers)
        return dataloader

"""
COPIED FROM SEVIR!
"""
PIXEL_SCALE = 255.0
BOUNDS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, PIXEL_SCALE]
THRESHOLDS = (16, 74, 133, 160, 181, 219)


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

HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255
PIXEL_SCALE = 255.0
BOUNDS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, PIXEL_SCALE]
THRESHOLDS = (16, 74, 133, 160, 181, 219)


def gray2color(image, **kwargs):

    # 定义颜色映射和边界
    cmap = colors.ListedColormap(COLOR_MAP )
    bounds = BOUNDS
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # 将图像进行染色
    colored_image = cmap(norm(image))

    return colored_image