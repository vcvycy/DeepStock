from util import *
import torch

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from train.resource_manager import RM
import logging
import math
import os
class FidIndex:
    f2i = {}
    i2f = {}
    @staticmethod
    def to_index(fids):
        if isinstance(fids, list):
            return [FidIndex.to_index(fid) for fid in fids]
        elif isinstance(fids, int):
            f = fids
            f2i = FidIndex.f2i
            i2f = FidIndex.i2f
            if f not in f2i:
                index = len(f2i) + 1
                f2i[f] =index
                i2f[index] = f
            return f2i[f]
        else:
            raise Exception("fid must be int or list")
        return 
    @Decorator.timing(func_name = "FidIndex.to_index_batch")
    def to_index_batch(fids_batch):
        indexs_batch = []
        for fids in fids_batch:
            indexs = [FidIndex.to_index(fid) for fid in fids]
            indexs_batch.append(indexs)
        return indexs_batch
    def to_fid(indexs):
        f2i = FidIndex.f2i
        i2f = FidIndex.i2f
        if isinstance(indexs, list):
            return [FidIndex.to_fid(index) for index in indexs]
        elif isinstance(indexs, int):
            index = indexs
            if index not in i2f:
                raise Exception("index not in i2f")
            return i2f[index]
        else:
            raise Exception("index must be int or list")
        return 

class FidEmbeddingV2(nn.Module):
    def __init__(self, embed_dims = 4, max_fid_num = 1000):
        super(FidEmbeddingV2, self).__init__()
        # self._max_fid_num = max_fid_num
        self._embed_dims = embed_dims     # 预计有100个维度
        self.variance = 0.02
        # shape = fid_num * dims
        self._fid_embedding = nn.Parameter(
            torch.randn(max_fid_num, self._embed_dims, requires_grad = True) * math.sqrt(self.variance))
        self._fid_bias = nn.Parameter(torch.zeros(max_fid_num, 1, requires_grad = True))
        # self._fid_bias = torch.arange(0, max_fid_num)*0.01

    @Decorator.timing(func_name = "FidEmbeddingV2.forward")
    def forward(self, fids_batch):
        """ fids_batch维度: batch_size x dimension
        """
        if not isinstance(fids_batch, torch.Tensor):
            fids_batch = FidIndex.to_index_batch(fids_batch)
            fids_batch = torch.tensor(fids_batch, dtype=torch.long).to(RM.device)
        # fids_batch:   batch_size * slot_num
        slot_num = fids_batch.shape[1]   
        # logging.info("slot_num: %s fids_batch: %s" %(slot_num, fids_batch.shape))
        try:
            batch_embd = torch.index_select(self._fid_embedding, dim=0, index=fids_batch.view(-1)) 
        except Exception as e:
            logging.error("fid max: %s fid_embedding.shape: %s" %(fids_batch.max(), self._fid_embedding.shape))
            raise e
        # print("batch_embd.shape before: %s" %(batch_embd.shape,))
        batch_embd = batch_embd.view(-1, slot_num, self._embed_dims)  # shape = batch_size * slot_num * dims

        batch_bias = torch.index_select(self._fid_bias, dim=0, index=fids_batch.view(-1))
        batch_bias = batch_bias.view(-1, slot_num)     # shape = batch_size * slot_num
        
        # logging.info("batch_embd: %s batch_bias: %s" %(batch_embd.shape, batch_bias.shape))
        return batch_embd, batch_bias
    
    def emit_summary(self):
        fid_num = len(FidIndex.i2f)
        bias = self._fid_bias
        embd = self._fid_embedding
        for idx in FidIndex.i2f:
            fid = FidIndex.i2f[idx]
            # RM.summary_writer.add_scalar(f"FidV2/bias/{fid}", bias[idx][0], global_step = RM.step)
            # RM.emit_summary(f"FidV2/slot{fid>>54}/{fid}/bias", bias[idx])
            RM.emit_summary(f"FidV2/slot{fid>>54}/{fid}/embd", embd[idx], var=False)
            if bias.grad is not None:
                RM.emit_summary(f"FidV2_grad/slot{fid>>54}/{fid}", bias.grad[idx])
        return 
    def show(self):
        """展示fid的embeding/bias
        """
        bias = self._fid_bias.detach().numpy()
        embd = self._fid_embedding.detach().numpy()
        items = []
        fid_num = len(FidIndex.i2f)
        for idx in FidIndex.i2f:
            fid = FidIndex.i2f[idx]
            items.append({
                "idx" : idx,
                "slot" : fid>>54,
                "fid" : fid,
                "bias" : bias[idx][0],
                f"embed({embd.shape[1]})" : embd[idx][:4]
            })
        items.sort(key = lambda x: -x['bias'])
        mprint(items)
        return 


def test_fidembedding_v2():
    print("test_fidembedding_v2".center(100, '-'))
    fidembeding = FidEmbeddingV2()

    # Example batch (assuming a batch size of 3 and each sample has 2 FIDs)
    fids_batch = [[10, 20], [10, 40], [20, 60]]
    # fids_batch = FidIndex.to_index(fids_batch)
    embd, bias = fidembeding(fids_batch)
    print(embd.shape)
    print(bias.shape)
    exit(0)
    return 

# 初始化全局Instance
# fidembeding = FidEmbedding()
if __name__ == "__main__":
    def test_fid_index():
        fids = [
            [12,3,4,5],
            [23,3,435,1]
        ]
        idxs = FidIndex.to_index(fids)
        fids2 = FidIndex.to_fid(idxs)
        print(idxs)
        print(fids2)
        return 

    test_fidembedding_v2()