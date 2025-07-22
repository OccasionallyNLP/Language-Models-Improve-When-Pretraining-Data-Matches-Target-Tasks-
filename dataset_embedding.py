# embedding -> similarity -> aggregate -> score
# data 
import argparse
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
import numpy as np
from tqdm import tqdm
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../prolong/fineweb-edu-dedup-subset.jsonl")
    parser.add_argument('--benchmark_path', type=str) #, default="./benchmark_data.jsonl")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--streaming', type=bool, default=True)
    parser.add_argument('--total_data_size', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default="./encoded_data")
    return parser.parse_args()

def get_data(args):
    if args.data_path.endswith('.jsonl'):
        ds = load_dataset('json', data_files=args.data_path, split='train', streaming=args.streaming)
    elif args.data_path.endswith('parquet'):
        ds = load_dataset('parquet', data_files=args.data_path, split='train', streaming=args.streaming)
    if args.benchmark_path:
        benchmark_ds = load_dataset('json', data_files=args.benchmark_path, split='train')
    else:
        benchmark_ds = None
    return ds, benchmark_ds

if __name__ ==  '__main__':
    args = get_args()
    # Load the model
    model = SentenceTransformer(args.model_name, device = args.device, model_kwargs={"torch_dtype": "bfloat16"})
    # dataset
    ds, benchmark_ds = get_data(args)

    # sanity check
    os.makedirs(args.output_dir, exist_ok=True)
    
    # for ds - use np.memmap for memory efficiency
    if args.streaming:
        # create a np.memmap for the encoded data
        memmap_arr = np.memmap(f"{args.output_dir}/encoded_data.npy", dtype=np.float32, mode="w+", shape=(args.total_data_size, args.dim))
        cnt = 0
        texts = []
        start_idx = 0
        for d in tqdm(ds):
            texts.append(d['text'])
            cnt+=1
            if cnt>=args.total_data_size:
                break
            if len(texts) == args.batch_size:
                encoded_data = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy = True)
                end_idx = start_idx + args.batch_size
                memmap_arr[start_idx:end_idx] = encoded_data
                start_idx = end_idx
                texts = []
        if len(texts) > 0:
            encoded_data = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy = True)
            end_idx = start_idx + len(texts)
            memmap_arr[start_idx:end_idx] = encoded_data
            start_idx = end_idx
            texts = []
    else:
        memmap_arr = np.memmap(f"{args.output_dir}/encoded_data.npy", dtype=np.float32, mode="w+", shape=(len(ds), args.dim))
        encoded_data = model.encode(ds['text'], batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy = True)
        memmap_arr[:] = encoded_data
    memmap_arr.flush()

    # for benchmark_ds
    if args.benchmark_path:
        bm_memmap_arr = np.memmap(f"{args.output_dir}/encoded_benchmark_data.npy", dtype=np.float32, mode="w+", shape=(len(benchmark_ds), args.dim))
        encoded_data = model.encode(benchmark_ds['text'], batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy = True)
        bm_memmap_arr[:] = encoded_data
        bm_memmap_arr.flush()

