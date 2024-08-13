
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from util import *
from utils3 import mprint   
from train.resource_manager import RM  # 确保这个导入是有效的
from train.fid_embedding import fidembeding
import json
class LRModel(nn.Module):
    """ 逻辑回归
        注意: epoch得多训练几次, avg_label才能对的上
    """
    def __init__(self):
        super(LRModel, self).__init__()
        self.first_time = True
        # 添加一些初始参数
        self.dummy = nn.Parameter(torch.zeros(1, requires_grad=True,  device=RM.device))
    @Decorator.timing
    def forward(self, fids_batch):
        embeddings = []
        # test_count = 0
        for fids in fids_batch:
            embeddings.append(torch.cat([fidembeding.get_embedding(fid, 1) for fid in fids])) 
        embeddings = torch.stack(embeddings).to(RM.device)
        logits = torch.sum(embeddings, dim=1)
        
        prediction = logits 
        if self.first_time:
            logging.info(f"LRModel: embeddings({embeddings.shape}), logits({logits.shape})")
            self.first_time = False
        return prediction.squeeze()  # 确保预测值的尺寸与标签的尺寸一致
    def post_process(self, *args, **kwargs):
        pass


class DNNModel(nn.Module):
    """ 逻辑回归
    """
    def __init__(self):
        def initialize_weights(m):
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0)
        from torch.nn import Linear
        from torch.nn import ReLU
        super(DNNModel, self).__init__()
        self.first_time = True
        self.fid_dims = 4
        # 添加一些初始参数
        self.dummy = nn.Parameter(torch.zeros(1, requires_grad=True,  device=RM.device))
        hidden_dims=[16, 8]
        input_dims = self.fid_dims * RM.data_source.slot_num
        self.layers = nn.Sequential(
            Linear(input_dims, hidden_dims[0]),  # 第一隐藏层
            ReLU(),                             # 激活函数
            Linear(hidden_dims[0], hidden_dims[1]),  # 第二隐藏层
            ReLU(),                             # 激活函数
            Linear(hidden_dims[1], 1)   # 输出层
        )
        self.layers.apply(initialize_weights)
    @Decorator.timing
    def forward(self, fids_batch):
        embeddings = []
        # test_count = 0
        bias_list = []
        for fids in fids_batch:
            embedding = [fidembeding.get_embedding(fid, self.fid_dims, include_bias = True, device =RM.device) for fid in fids]
            bias_list.append(torch.cat([b for w, b in embedding]))
            embeddings.append(torch.cat([w for w, b in embedding])) 
        embeddings = torch.stack(embeddings).to(RM.device)
        
        RM.emit_summary("dnn_input", embeddings, step =  RM.step)
        self.nn_out = self.layers(embeddings).squeeze()  
        self.bias_sum = torch.mean(torch.stack(bias_list), dim=1)
        prediction =   self.bias_sum +self.nn_out 
        if self.first_time:
            logging.info(f"DNNModel: embeddings({embeddings.shape}), logits({self.nn_out.shape})")
            self.first_time = False
        return prediction.squeeze()  

    def post_process(self, step):
        layers = self.layers
        # 遍历并记录权重和偏置
        with torch.no_grad():
            for idx, layer in enumerate(layers):
                if isinstance(layer, nn.Linear):  # 检查是否为线性层
                    tag_base = f"dnn/layer_{idx}/"
                    RM.emit_summary(tag_base + "weight", layer.weight, step)
                    if layer.weight.grad is not None:
                        RM.emit_summary(tag_base + "weight/grad", layer.weight.grad, step)
                    RM.emit_summary(tag_base + "bias", layer.bias, step)
                
            RM.emit_summary("logits/bias_sum", self.bias_sum, step)
            RM.emit_summary("logits/nn_out", self.nn_out, step)
        return 
