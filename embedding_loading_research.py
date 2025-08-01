"""
embedding_loading_research.py

本文件用于研究和实验大规模embedding文件的高效加载方案。
适用场景：百万级用户/千万级item，embedding文件TB级别。

思路：
1. 多进程/多线程并行加载多个文件。
2. 分块读取大文件，减少内存峰值。
3. 只加载当前epoch/batch需要的embedding（按需加载/懒加载）。
4. mmap内存映射大文件，避免一次性读入全部。
5. embedding持久化为高效二进制格式（如npz/hdf5/自定义二进制），加快I/O。
6. 结合分布式训练，embedding分片存储。

以下为部分实现伪代码和建议：
"""

import os
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

# 1. 多线程/多进程并行加载

def load_embedding_parallel(emb_dir, file_pattern='part-*', max_workers=8):
    """
    并行加载embedding目录下所有文件，返回合并后的embedding字典。
    """
    emb_dict = {}
    files = list(Path(emb_dir).glob(file_pattern))
    def load_file(f):
        local_dict = {}
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                emb = np.array(data['emb'], dtype=np.float32)
                local_dict[data['anonymous_cid']] = emb
        return local_dict
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(load_file, f): f for f in files}
        for future in as_completed(future_to_file):
            local_dict = future.result()
            emb_dict.update(local_dict)
    return emb_dict

# 2. 分块读取大文件

def load_embedding_chunked(file_path, chunk_size=100000):
    """
    分块读取单个大embedding文件，减少内存峰值。
    """
    emb_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        chunk = []
        for i, line in enumerate(file):
            chunk.append(line)
            if (i+1) % chunk_size == 0:
                for l in chunk:
                    data = json.loads(l.strip())
                    emb = np.array(data['emb'], dtype=np.float32)
                    emb_dict[data['anonymous_cid']] = emb
                chunk = []
        # 处理最后一块
        for l in chunk:
            data = json.loads(l.strip())
            emb = np.array(data['emb'], dtype=np.float32)
            emb_dict[data['anonymous_cid']] = emb
    return emb_dict

# 3. mmap内存映射/二进制格式加载（建议将embedding预处理为npy/npz/hdf5等格式）
# 4. 按需加载/懒加载方案可结合__getitem__实现，或用LRU cache。

# 5. 分布式/分片存储方案建议结合具体平台和训练框架实现。

if __name__ == '__main__':
    # 示例：并行加载
    emb_dir = '/path/to/emb_82_1024'
    emb_dict = load_embedding_parallel(emb_dir, file_pattern='part-*', max_workers=16)
    print(f"Loaded {len(emb_dict)} embeddings.")
    # 示例：分块加载
    # emb_dict = load_embedding_chunked('/path/to/emb_82_1024/part-00000', chunk_size=50000)
    # print(f"Loaded {len(emb_dict)} embeddings.")