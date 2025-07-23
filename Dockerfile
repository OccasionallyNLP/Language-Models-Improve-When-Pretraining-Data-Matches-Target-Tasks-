FROM huggingface/transformers-pytorch-gpu:latest
WORKDIR /workspace

# 현재 로컬 폴더를 Docker 이미지에 복사
COPY . /workspace/Language-Models-Improve-When-Pretraining-Data-Matches-Target-Tasks-

RUN pip install --upgrade pip
RUN pip install --upgrade datasets
RUN pip install --upgrade accelerate
RUN pip install --upgrade sentence_transformers
RUN pip install --upgrade faiss-cpu
