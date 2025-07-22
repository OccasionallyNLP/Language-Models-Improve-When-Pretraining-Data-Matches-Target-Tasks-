import numpy as np
import faiss
import argparse 
import glob
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./encoded_data")
    parser.add_argument('--benchmark_path', type=str, default="./encoded_benchmark_data.npy")
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--save_path', type=str, default="./scores.npy")
    return parser.parse_args()

def get_score():
    pass

if __name__ == "__main__":
    args = get_args()
    # dataset
    subfolders = [name for name in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, name))]
    subfolders = sorted(subfolders, key=lambda x: int(x))

    index = faiss.IndexFlatL2(args.dim)
    for folder in subfolders:
        npy_files = glob.glob(os.path.join(args.data_dir, folder, '*.npy'))
        assert npy_files
        for file in npy_files:
            ds = np.fromfile(file, dtype=np.float32)
            ds = ds.reshape(-1, args.dim)
            index.add(ds)
    faiss.normalize_L2(ds)
    index.add(ds)
    # benchmark dataset
    bm_ds = np.fromfile(args.benchmark_path, dtype=np.float32)
    bm_ds = bm_ds.reshape(-1, args.dim)
    _scores = index.search(bm_ds, k=index.ntotal) # len(bm_ds), index.ntotal
    scores = _scores[1].min(axis=0)
    # save_path
    memmap_arr = np.memmap(f"{args.save_path}", dtype=np.float32, mode="w+", shape=(len(scores)))
    memmap_arr[:] = 1/(scores+1)
    memmap_arr.flush()
    print(f"Scores saved to {args.save_path}")
