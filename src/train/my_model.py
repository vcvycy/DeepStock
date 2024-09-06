
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from util import *
from utils3 import mprint   
from train.resource_manager import RM  # 确保这个导入是有效的
from train.fid_embedding import FidEmbeddingV2
import json

# from train.distill_loss import DistillLoss
class MyReLU(nn.Module):
    def __init__(self, name ):
        super(MyReLU, self).__init__()
        self.name = name

    def forward(self, x):
        x = nn.functional.relu(x)
        zero_fraction = (x == 0).float()
        RM.emit_summary(self.name, zero_fraction, var=False)
        return x

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, name):
        super(MyLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.name = name
    def forward(self, x):
        RM.emit_summary(self.name +"/input", x, hist=True)
        x = self.linear(x)
        RM.emit_summary(self.name +"/output", x, hist=True)
        return x
class DistillLoss(nn.Module):
    def __init__(self, input_dims):
        super(DistillLoss, self).__init__() 
        self.cross_entropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        # distill分桶
        self.min_value = -0.2
        self.max_value = 0.2
        self.bucket = 40
        self.interval = (self.max_value-self.min_value)/self.bucket
        self.bounds = torch.tensor([self.min_value + i * self.interval + self.interval/2 for i in range(self.bucket)]).to(RM.device)
        self.layers = nn.Sequential(
            MyLinear(input_dims, self.bucket, name='distill/linear_bucket')
        )
    # @Decorator.timing(func_name="DistillLoss-label2onehot")
    def label2onehot(self, label):
        label = label.view(-1)  # 转为一维 
        # indices: 即0~ bucket-1中的数
        indices = torch.clamp(torch.floor((label - self.min_value) / self.interval), 0, self.bucket - 1)
        label_one_hot = torch.nn.functional.one_hot(indices.to(torch.int64), num_classes=self.bucket).float()
        return label_one_hot

    # @Decorator.timing(func_name="DistillLoss-forward")
    def forward(self, in_tensor, label=None):
        # label转one hot
        # 计算logits和probs多分类
        logits = self.layers(in_tensor)
        probs = self.softmax(logits)
        # 计算predictions
        bucket_pred = probs * self.bounds
        pred = torch.sum(bucket_pred, dim=1)
        # 计算loss 
        if label is not None:
            label_one_hot = self.label2onehot(label)
            if RM.can_emit_summary():
                for i in range(self.bucket):
                    RM.emit_summary("distill_softmax/%s/%.3f/pred" %(i, self.bounds[i]), probs[:, i], var=False)
                    RM.emit_summary("distill_softmax/%s/%.3f/label" %(i, self.bounds[i]), label_one_hot[:, i], var=False)
            loss = - torch.sum(label_one_hot * torch.log(probs))
            return pred, loss
        else:
            return pred, None



class DistillModel(nn.Module):
    def __init__(self):
        super(DistillModel, self).__init__()
        self.first_time = True
        # 添加一些初始参数
        self.slot_num = RM.data_source.slot_num
        self.embed_dims = 4
        self.fid_embedding = FidEmbeddingV2(embed_dims = self.embed_dims, max_fid_num = 1000)
        nn_dims = [16, 8]
        self.layers = nn.Sequential(
            MyLinear(self.embed_dims * self.slot_num, nn_dims[0], name='distill/linear1'),
            MyReLU("distill/relu1"),  
            MyLinear(nn_dims[0], nn_dims[1], name='distill/linear2'),
            MyReLU("distill/relu2"),  
        )
        self.distill_loss = DistillLoss(nn_dims[-1])
        return 

    @Decorator.timing(func_name="DistillModel-forward")
    def forward(self, fids_batch, label=None):
        embed, bias = self.fid_embedding(fids_batch) 
        RM.emit_summary("distill/input_embed", embed, hist=True)
        # logits = torch.sum(bias, dim=1)
        embed = embed.view(-1, RM.data_source.slot_num * self.embed_dims)
        embed = self.layers(embed)
        pred, loss = self.distill_loss(embed, label)
        pred = pred.squeeze() 
        if self.first_time:
            logging.info(f"DistillModel: fid embeddings 过nn({embed.shape}) fid bias({bias.shape}), pred({pred.shape})")
            self.first_time = False 
        
        if label is not None:
            return pred, loss
        else:
            return pred
    def post_process(self, *args, **kwargs):
        self.fid_embedding.emit_summary()
        pass
    def embed_show(self):
        self.fid_embedding.show()
        return 

class LRModelV2(nn.Module):

    def __init__(self):
        super(LRModelV2, self).__init__()
        self.first_time = True
        # 添加一些初始参数
        self.fid_embedding = FidEmbeddingV2()
    @Decorator.timing()
    def forward(self, fids_batch, label = None):
        embed, bias = self.fid_embedding(fids_batch) 
        logits = torch.sum(bias, dim=1)
        
        prediction = logits 
        if self.first_time:
            logging.info(f"LRModel: embeddings({embed.shape}) bias({bias.shape}), logits({logits.shape})")
            self.first_time = False 
        prediction = prediction.squeeze()
        if label is not None:
            loss = F.mse_loss(prediction, label.to(RM.device), reduction='sum')
            return prediction, loss
        else:
            return prediction
    def post_process(self, *args, **kwargs):
        self.fid_embedding.emit_summary()
        pass
    def embed_show(self):
        self.fid_embedding.show()
        return 

def test_distill_loss():
    distill_loss = DistillLoss(input_dims=10)
    input = torch.randn(4, 10)
    label = torch.randn(4, 1)
    pred, loss = distill_loss(input, label)
    print("pred".center(100, "*"))
    print(pred)
    print("loss".center(100, "*"))
    print(loss)
    return 
if __name__ == "__main__":
    pass