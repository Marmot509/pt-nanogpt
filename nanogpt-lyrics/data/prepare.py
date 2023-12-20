import os
import numpy as np
from tqdm import tqdm
import tiktoken
from datasets import load_dataset, DatasetDict

num_proc = 8

def load_jsonl_dataset(file_path):
    dataset = load_dataset('json', data_files=file_path, split='train')
    return dataset

if __name__ == '__main__':
    file_path = 'lyric_data_for_CL_no_id.jsonl'
    dataset = load_jsonl_dataset(file_path)

    # 移除 dataset_split 字段
    dataset = dataset.remove_columns('dataset_split')

    # 创建训练、验证和测试分割
    train_test_dataset = dataset.train_test_split(test_size=0.2, seed=2357, shuffle=True)
    test_val_dataset = train_test_dataset['test'].train_test_split(test_size=0.5, seed=2357, shuffle=True)
    split_dataset = DatasetDict({
        'train': train_test_dataset['train'],
        'test': test_val_dataset['test'],
        'val': test_val_dataset['train']
    })

    # 定义编码函数
    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        ids = enc.encode_ordinary(example['lyric'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        
        return out

    # 对数据进行编码
    tokenized = split_dataset.map(
        process,
        remove_columns=['lyric'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # 生成训练数据文件
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
