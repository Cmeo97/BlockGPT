import os
from typing import Iterable, List, Optional, Tuple

import numpy as np


def _get_available_cpus() -> int:
    """Return the number of CPUs available to this process.

    Prefers Linux affinity when available; falls back to os.cpu_count().
    """
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        return os.cpu_count() or 1


def _list_h5_files(dataset_dir: str) -> List[str]:
    """List all .h5/.hdf5 files in a directory, sorted by name."""
    entries = []
    for name in os.listdir(dataset_dir):
        if name.endswith(".h5") or name.endswith(".hdf5"):
            entries.append(os.path.join(dataset_dir, name))
    entries.sort()
    return entries


def _read_h5_videos_file(path: str, dataset_key: str = "videos", dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Read the full dataset from a single HDF5 file.

    Parameters
    - path: absolute path to the HDF5 file
    - dataset_key: dataset name inside the HDF5 (default: 'videos')
    - dtype: optionally cast to a numpy dtype (e.g., np.float32)
    """
    import h5py  # local import to avoid importing when unused

    with h5py.File(path, "r") as f:
        ds = f[dataset_key]
        arr = ds[...]
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def iter_h5_videos(
    dataset_dir: str,
    dataset_key: str = "videos",
    dtype: Optional[np.dtype] = np.float32,
) -> Iterable[Tuple[str, np.ndarray]]:
    """Yield (filename, array) for each HDF5 file in dataset_dir.

    This is memory-efficient: it reads one file at a time.
    """
    for path in _list_h5_files(dataset_dir):
        yield (path, _read_h5_videos_file(path, dataset_key=dataset_key, dtype=dtype))


def load_all_h5_videos(
    dataset_dir: str,
    dataset_key: str = "videos",
    dtype: Optional[np.dtype] = np.float32,
    as_torch: bool = False,
    workers: Optional[int] = None,
    max_files: Optional[int] = None,
    verbose: bool = True,
):
    """Load and concatenate all 'videos' datasets across HDF5 files in dataset_dir.

    Parameters
    - dataset_dir: directory containing .h5/.hdf5 files
    - dataset_key: dataset name inside HDF5 files (default: 'videos')
    - dtype: optional numpy dtype to cast arrays to (e.g., np.float32)
    - as_torch: if True, return a torch.Tensor; else return np.ndarray
    - workers: number of parallel workers (per-file). Defaults to min(num_files, available_cpus).
    - max_files: limit number of files read (useful for smoke tests)
    - verbose: print simple progress information

    Returns
    - concatenated array of shape (sum_i Ni, T, H, W) or torch.Tensor if as_torch=True
    """
    file_paths = _list_h5_files(dataset_dir)
    if max_files is not None:
        file_paths = file_paths[:max_files]

    if not file_paths:
        raise FileNotFoundError(f"No HDF5 files found in directory: {dataset_dir}")

    # Decide workers
    if workers is None:
        workers = min(len(file_paths), _get_available_cpus())
    workers = max(1, workers)

    arrays: List[np.ndarray] = []

    if workers == 1:
        for idx, path in enumerate(file_paths):
            if verbose:
                print(f"[{idx+1}/{len(file_paths)}] Reading {os.path.basename(path)}")
            arrays.append(_read_h5_videos_file(path, dataset_key=dataset_key, dtype=dtype))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if verbose:
            print(f"Reading {len(file_paths)} files with {workers} workers...")

        def task(p: str) -> Tuple[str, np.ndarray]:
            return (p, _read_h5_videos_file(p, dataset_key=dataset_key, dtype=dtype))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(task, p): p for p in file_paths}
            for i, fut in enumerate(as_completed(futures), 1):
                path, arr = fut.result()
                if verbose:
                    print(f"[{i}/{len(file_paths)}] Read {os.path.basename(path)} shape={arr.shape}")
                arrays.append(arr)

        # Maintain original order by re-reading sequentially if necessary
        # We sort arrays by the order of file_paths using an index map
        order = {p: i for i, p in enumerate(file_paths)}
        arrays = [x for _, x in sorted(zip([order[fp] for fp in file_paths], arrays), key=lambda t: t[0])]

    # Validate consistent shapes except the first dimension
    base_shape = arrays[0].shape[1:]
    for a in arrays:
        if a.shape[1:] != base_shape:
            raise ValueError(f"Inconsistent array shapes after axis 0: {a.shape} vs {arrays[0].shape}")

    concatenated = np.concatenate(arrays, axis=0)

    if as_torch:
        import torch  # local import
        return torch.from_numpy(concatenated)
    return concatenated


if __name__ == "__main__":
    # Default directory from the user request
    DEFAULT_DATASET_DIR = "/projects/0/prjs0951/Varun/KNMI/5mins/train"

    print(f"Dataset dir: {DEFAULT_DATASET_DIR}")
    try:
        data = load_all_h5_videos(
            dataset_dir=DEFAULT_DATASET_DIR,
            dataset_key="videos",
            dtype=np.float32,
            as_torch=False,
            workers=None,  # auto
            verbose=True,
        )
        print(f"Loaded concatenated data: shape={data.shape}, dtype={data.dtype}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")


