# Language Models Improve When Pretraining Data Matches Target Tasks
<img width="1115" height="713" alt="image" src="https://github.com/user-attachments/assets/9031b9b1-d6e8-441b-bcf7-6be7b753bda2" />

## process
1. Download benchmark dataset
  You have to preprocess benchmark dataset such as concatenate question and answer.
  Select equal number of datasets. If you have 5 datasets, select 1000 for each datasets (This is described in paper)
  Shuffle and concatenate your benchmark dataset. 
  You can refer benchmark_data.py file.

2. Embed benchmark data and pretrained dataset.
3. 

## component
- Embed training and benchmark examples
  - huggingface & faiss
    - 각 GPU마다, huggingface model을 load해서 embedding 진행.
    - data shard가 필요함. 
- train embedding model
  - huggingface
- scoring & filter
  - huggingface

## benchmark -> 각 dataset마다 1500개씩 select할 예정.
- mmlu (test) 
  : 57개의 domain이 존재함. 최소 test set은 100개임 -> 총 5700개 test 덩어리로 구성됨.
- mmlu pro (test) 
  : domain 91개. 최소 dataset 길이 17. # 1547개
- winogrande (winograde xl - test) 
  : dataset 개수 1767
- ai2_arc (ARC-Easy - test) 
  : dataset 개수 : 2376
- hellaswag (test)
  : dataset 개수 : 10003
- bigbench (default) 
  : min value: 8. dataset 종류 167. 총 dataset 