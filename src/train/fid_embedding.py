from util import *
import torch

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from train.resource_manager import RM
import logging

class FidEmbedding():
    """ FidEmbeding 为什么要单独拆出来：
        假设: batch_size =1000, 但是某个fid 在batch中只出现了10次,那么计算MSE的时候,这个fid的梯度会/batch_size,导致梯度消失。
            那为什么不用MSE(reduction = "sum") ? 这样会导致dense特征梯度非常大
        
        传给FidEmbedding的梯度，必须为reduction="SUM"
    """
    def __init__(self):
        self.fid2embedding = nn.ParameterDict()
        self.fid2lr = {}
        self.weight_decay = 1e-4
    @Decorator.timing
    def get_embedding(self, fid, dims = 1, include_bias = False, device = None):
        if str(fid) not in self.fid2embedding:
            if include_bias:
                dims +=1 
            embedding =nn.Parameter(torch.zeros(dims, requires_grad=True)) # nn.Parameter(torch.randn(1, requires_grad=True)) * 0.02
            if device is not None:
                embedding = embedding.to(device)
            self.fid2embedding[str(fid)] = embedding
            self.fid2lr[str(fid)] = 1e-3
        embed = self.fid2embedding[str(fid)]
        # print("fidembedding device: %s set to %s" %(embed.device, device))
        if include_bias:
            return embed[:-1], embed[-1:]
        else:
            return embed
    @Decorator.timing
    def update_embedding(self, fids_batch, step):
        # 计算每个fid的频次
        fid2count = {}
        for fids in fids_batch:
            for fid in fids:
                if str(fid) not in fid2count:
                    fid2count[str(fid)] = 0
                fid2count[str(fid)] += 1
        # 更新每个fids的embedding
        for fid in fid2count:
            # 学习率
            learning_rate = self.fid2lr[fid] 
            self.fid2lr[fid] = max(1e-4, self.fid2lr[fid] - 1e-5)  # 学习率从 1e-2下降到1e-3
            #
            embedding = self.fid2embedding[fid]
            updated_embedding = (1-self.weight_decay) * embedding - (learning_rate * embedding.grad / fid2count[str(fid)])
            # if fid in ["2425480359478669827", "2427379723444939871"]:
            #     print("update_embedding:%s %s -> grad %s lr: %.3f cnt: %d"  %(fid, embedding.detach().numpy(), embedding.grad.detach().numpy(), learning_rate, fid2count[str(fid)]))
            # 用新计算的embedding替换原来的embedding
            self.fid2embedding[str(fid)].data.copy_(updated_embedding.data)
            # summary
            tag_base = f"FidEmbedding/{int(fid)>>54}/{fid}"
            RM.emit_summary(tag_base + "_weight", embedding, step)
            RM.emit_summary(tag_base + "_grad", embedding.grad, step)
        # 清空梯度
        for emb in self.fid2embedding.values():
            emb.grad.zero_()
        return 
# 初始化全局Instance
fidembeding = FidEmbedding()
