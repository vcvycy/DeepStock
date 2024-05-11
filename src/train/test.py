import torch
from torch.utils.data import IterableDataset, DataLoader

# 定义一个自定义的 IterableDataset
class MyIterableDataset(IterableDataset):
    def __init__(self, num_samples):
        super(MyIterableDataset, self).__init__()
        self.num_samples = num_samples

    def __iter__(self):
        for i in range(self.num_samples):
            input_data = [1, 2, 3]  # 每个样本的输入数据
            label = i % 2  # 标签在 0 和 1 之间交替
            print("input: %s label: %s" %(input_data, label))
            yield input_data, label

# 创建一个 MyIterableDataset 实例
dataset = MyIterableDataset(num_samples=10)

# 创建一个 DataLoader 实例
dataloader = DataLoader(dataset, batch_size=2)

# 遍历 DataLoader 并打印数据
for batch in dataloader:
    inputs, labels = batch
    print(f"Inputs: {inputs}, Labels: {labels}")