# embedding -> similarity -> aggregate -> score
# data 
import argparse
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./prolong/fineweb-edu-dedup-subset.jsonl")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--streaming', type=bool, default=True)
    return parser.parse_args()

if __name__ ==  '__main__':
    args = get_args()
    # Load the model
    
    model = SentenceTransformer(args.model_name, device = args.device)
    if args.data_path.endswith('.jsonl'):
    # splitted dataset
        ds = load_dataset('json', data_files=args.data_path, split='train', streaming=args.streaming)
    elif args.data_path.endswith('parquet'):
        ds = load_dataset('parquet', data_files=args.data_path, split='train', streaming=args.streaming)






# queries = [
#     "What is the capital of China?",
#     "Explain gravity",
# ]


query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents) 

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)

print()

# faiss
index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
index.add_with_ids(encoded_data, np.array(range(0, len(data))))

faiss.write_index(index, 'abc_news')