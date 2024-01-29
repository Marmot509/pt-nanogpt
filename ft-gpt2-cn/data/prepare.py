import os
import numpy as np
from tqdm import tqdm
import tiktoken
from datasets import load_dataset, DatasetDict

num_proc = 2        #tokenize时使用的进程数

def load_jsonl_dataset(file_path):
    dataset = load_dataset('json', data_files=file_path, split='train')
    return dataset

if __name__ == '__main__':
    file_path = 'lyric_data_for_CL_no_id.jsonl'
    dataset = load_jsonl_dataset(file_path)
    dataset = dataset.remove_columns('dataset_split')   # 移除 dataset_split 字段

    # 创建训练、验证和测试分割
    train_test_dataset = dataset.train_test_split(test_size=0.05, seed=1024, shuffle=True)
    train_val_dataset = train_test_dataset['train'].train_test_split(test_size=0.1, seed=1024, shuffle=True)

    # 现在，创建包含训练集、验证集和测试集的 DatasetDict
    split_dataset = DatasetDict({
        'train': train_val_dataset['train'],
        'val': train_val_dataset['test'],   # 注意这里使用了 'test' 作为验证集
        'test': train_test_dataset['test']
    })


    # 定义编码函数
    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        lyrics_text = '，'.join(example['lyric'])        # 将列表中的字符串使用逗号连接成单个字符串
        ids = enc.encode_ordinary(lyrics_text)          # tokenization
        ids.append(enc.eot_token)                       # 增加文本结束标识符
        out = {'ids': ids, 'len': len(ids)}             # 构建输出字典
        return out


    # 对数据进行编码
    tokenized = split_dataset.map(      
        process,                        # 使用process函数进行数据处理
        remove_columns=['lyric'],       # 移除文本形态的数据，仅保留由process函数生成的tokenized的数据
        desc="tokenizing the splits",   # tqdm进度条旁边显示的文本
        num_proc=num_proc,              # 使用的进程数
    )

    # 生成训练数据文件
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)                          # 计算数据集中所有数据的长度
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')      # 生成数据文件的路径
        dtype = np.uint16                                                       
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))     # 创建一个内存映射文件，用于存储数据
        total_batches = 1024                                                    # 将数据集分成1024个batch写入内存映射文件

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):        # tqdm进度条
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
