import random
import torch
from torch.utils.data import IterableDataset, DataLoader
import logging
from train.resource_manager import RM  # 确保这个导入是有效的
from util import Latency
from utils3 import mprint   
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time


class StockIterableDataset(IterableDataset):
    """输入 fids  = [3234, 46457, 6745, ...]
       返回: embedding = {slot1 : torch.tensor(dim = 128), slot2: torch.tensor(dim = 124), ...}
    """
    def __init__(self, data_source_getter):
        super(StockIterableDataset, self).__init__()
        self.shuffle_batch = []
        self.shuffle_size = 20000
        self.terms_generated = 0
        self.count = 0
        self.data_source_getter = data_source_getter

    def __iter__(self):
        return self

    def update_shuffle_batch(self):
        if len(self.shuffle_batch) > 0:
            return
        latency = Latency()
        while len(self.shuffle_batch) < self.shuffle_size:
            item = self.data_source_getter()
            if item is None:
                break
            self.shuffle_batch.append(item)
            self.terms_generated += 1
        # random.shuffle(self.shuffle_batch)   # TODO
        self.count += len(self.shuffle_batch)
        logging.info(f"[IterableDataset] 获取{len(self.shuffle_batch)}条数据shuffle, 总数据: {self.count}, 耗时:{latency.count():.3f}s")

    def __next__(self):
        if not self.shuffle_batch:
            self.update_shuffle_batch()
            if not self.shuffle_batch:
                raise StopIteration
        fids, label, ins = self.shuffle_batch.pop(0)  # 使用 pop(0) 移除并返回第一个元素
        return fids, label, ins
def custom_collate(batch):
    fids_list = [item[0] for item in batch]
    labels_list = torch.tensor([item[1] for item in batch])
    return fids_list, labels_list

class FidEmbedding():
    """ FidEmbeding 为什么要单独拆出来：
        假设: batch_size =1000, 但是某个fid 在batch中只出现了10次,那么计算MSE的时候,这个fid的梯度会/batch_size,导致梯度消失。
            那为什么不用MSE(reduction = "sum") ? 这样会导致dense特征梯度非常大
    """
    def __init__(self):
        self.fid2embedding = nn.ParameterDict()
        pass
    def get_embedding(self, fid):
        if str(fid) not in self.fid2embedding:
            embedding =nn.Parameter(torch.zeros(1, requires_grad=True)) # nn.Parameter(torch.randn(1, requires_grad=True)) * 0.02
            self.fid2embedding[str(fid)] = embedding
        return self.fid2embedding[str(fid)]

    def update_embedding(self, fids_batch):
        # 计算每个fid的频次
        fid2count = {}
        for fids in fids_batch:
            for fid in fids:
                if str(fid) not in fid2count:
                    fid2count[str(fid)] = 0
                fid2count[str(fid)] += 1
        # 更新每个fids的embedding
        learning_rate = 0.1
        for fid in self.fid2embedding:
            if str(fid) in fid2count:
                embedding = self.fid2embedding[fid]
                updated_embedding = embedding - (learning_rate * embedding.grad / fid2count[str(fid)])
                # 用新计算的embedding替换原来的embedding
                self.fid2embedding[str(fid)].data.copy_(updated_embedding.data)
                # embedding -= learning_rate * embedding.grad / fid2count[str(fid)]
                # print("fid: %s , 更新梯度: %s" %(embedding.grad, fid2count[str(fid)]))
        # 清空梯度
        for emb in self.fid2embedding.values():
            emb.grad.zero_()
        return 

fidembeding = FidEmbedding()
class LRModel(nn.Module):
    """ 逻辑回归
    """
    def __init__(self):
        super(LRModel, self).__init__()
        self.first_time = True
        # 添加一些初始参数
        self.dummy = nn.Parameter(torch.zeros(1, requires_grad=True))
    def forward(self, fids_batch):
        embeddings = []
        # test_count = 0
        for fids in fids_batch:
            embeddings.append(torch.cat([fidembeding.get_embedding(fid) for fid in fids])) 
        #     if 2422899029155717707 in fids:
        #         test_count += 1
        # logging.info("当前batch, 2422899029155717707 test_count :%s" %(test_count))
        embeddings = torch.stack(embeddings)
        output = torch.sum(embeddings, dim=1)
        prediction = output  # torch.sigmoid(output)
        if self.first_time:
            logging.info(f"LRModel: embeddings({embeddings.shape}), output({output.shape})")
            self.first_time = False
        return prediction.squeeze()  # 确保预测值的尺寸与标签的尺寸一致

def main():
    writer_file= 'runs/%s' %(int(time.time()))
    writer = SummaryWriter(writer_file)
    logging.info("数据写到: %s" %(writer_file))
    model = LRModel()
    batch_size = 908
    train_data = DataLoader(StockIterableDataset(data_source_getter=RM.data_source.next_train), batch_size=batch_size, collate_fn=custom_collate)
    test_data = DataLoader(StockIterableDataset(data_source_getter=RM.data_source.next_test), batch_size=batch_size, collate_fn=custom_collate)
    # mprint(model.parameters(), title = "model.parameters")
    # for params in model.parameters():
    #     logging.info('='*100)
    #     logging.info(params)
    

    data = []
    step =0 
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for fids, label in train_data:
        step +=1
        optimizer.zero_grad()  # 清零梯度
        prediction = model(fids)
        loss = F.mse_loss(prediction, label, reduction='sum')  # mean reduction，会导致低频的fid的梯度被大幅度降低, 所以fid单独拆出去
        loss.backward()  # 计算梯度
        item = {
            # "pred" : prediction[0].item(),
            # "pred_grad" : prediction[0].grad,
            "round" : len(data),
            "loss" : loss.item(),
            "embed" : len(list(model.parameters()))
        }
        if 2422899029155717707 in fids:
            logging.info("fid: %s, label: %s, prediction: %s" %(model.fid2embedding[fid][0].item(), label, prediction))
        # if step %100 == 1:
        #     logging.info("step[%s] loss = %s" %(step, loss.item()))
        writer.add_scalar('prediction', torch.mean(prediction), step)
        writer.add_scalar('label', torch.mean(label), step)
        writer.add_scalar('loss', torch.mean(loss), step)
        for fid in fidembeding.fid2embedding:
            embed = fidembeding.fid2embedding[fid]
            item[fid] = "v:%.4f;g:%.4f" %(embed.item(), embed.grad[0].item())
            writer.add_scalar("slot_%s/fid_%s" %(int(fid)>>54, fid), embed, step)
        # logging.info("embedding数量: %s" %(len(model.fid2embedding)))
        fidembeding.update_embedding(fids, )
        data.append(item)
        optimizer.step()  # 更新参数
        # logging.info(model.state_dict())
        # input(".......")

    # logging.info(model.state_dict())
        # state_dict = optimizer.state_dict()
        # mprint(state_dict["state"], "optimizer.state")
        # logging.info(state_dict["param_groups"])
        # # mprint(state_dict["param_groups"], "optimizer.param_groups")
        # input("..")
    # for p in model.parameters():
    #     logging.info(p)
    # input("....")
    mprint(model.state_dict(), title = "模型dense state_dict权重")
    mprint(fidembeding.fid2embedding.state_dict(), title = "模型dense state_dict权重")
    # mprint(data)
    torch.save(model.state_dict(), 'model_weight.pth')
    logging.info("模型权重保存到: model_weight.pth")
    logging.info("总训练样本: %s step: %s" %(RM.data_source.train_count, step))

def test_data_loader():
    slot2dim = {i: 1 for i in range(1024)}
    batch_size = 525
    train_data = DataLoader(StockIterableDataset(data_source_getter=RM.data_source.next_train), batch_size=batch_size, collate_fn=custom_collate)
    test_data = DataLoader(StockIterableDataset(data_source_getter=RM.data_source.next_test), batch_size=batch_size, collate_fn=custom_collate)
    for fids, label in train_data:
        logging.info(f"训练数据, fids {fids}")
        logging.info(f"训练数据, label {label}")
        break
    for fids, label in test_data:
        logging.info(f"测试数据, fids {fids}")
        logging.info(f"测试数据, label {label}")
        break

if __name__ == "__main__":
    # test_data_loader()
    main()