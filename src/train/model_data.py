
from util import *
from common.stock_pb2 import Instance
import torch
from torch.utils.data import IterableDataset
from google.protobuf import text_format
from train.resource_manager import RM
class FFMIterableDataset(IterableDataset):
    def __init__(self, conf):
        self.conf = conf

    def _create_field_dict(self):
        field_dict = {}
        with open(self.files, 'r') as f:
            for line in f:
                instance = Instance()
                text_format.Parse(line, instance)
                for feature in instance.feature:
                    if feature.slot not in field_dict:
                        field_dict[feature.slot] = len(field_dict)
        return field_dict

    def parse_instance(self, line):
        instance = Instance()
        text_format.Parse(line, instance)
        features = []
        labels = list(instance.label.values())
        for feature in instance.feature:
            slot = feature.slot
            for fid in feature.fids:
                features.append((self.field_dict[slot], fid))
        return torch.tensor(features, dtype=torch.long), torch.tensor(labels, dtype=torch.float)

    def __iter__(self):
        inputs = []
        for ins in enum_instance(self.config.files):
            for feature in ins.feature:
                assert len(feature.fids) == 1, "fids!=0"
                inputs.append(feature.fids[0])
            inputs.sort(key = lambda x : x>>54)  # 按slot排序
            label = ins.label["next_3d"]
    
if __name__ == "__main__":# 使用示例
    import yaml
    from easydict import EasyDict
    conf = EasyDict(yaml.safe_load(open("./train/train.yaml", 'r') .read()))

    print(conf.data.files) 

    # 使用示例
    cross_fields = [(1, 5, 64), (1, 4, 32)]  # 这是一个示例，具体请根据实际情况
    dataset = FFMIterableDataset(conf.data.files, cross_fields)
    for features, labels in dataset:
        print(features, labels)
        input("..")