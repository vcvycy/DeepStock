import random
import torch
from torch.utils.data import IterableDataset, DataLoader
import logging
from train.resource_manager import RM  # 确保这个导入是有效的
from util import *
from utils3 import mprint   
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import time
from datetime import datetime
from train.fid_embedding import fidembeding
import json

import torch.nn.init as init


class StockIterableDataset(IterableDataset):
    """输入 fids  = [3234, 46457, 6745, ...]
       返回: embedding = {slot1 : torch.tensor(dim = 128), slot2: torch.tensor(dim = 124), ...}
    """
    def __init__(self, data_source_getter):
        super(StockIterableDataset, self).__init__()
        self.shuffle_batch = []
        self.shuffle_size = 12345
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
        random.shuffle(self.shuffle_batch)   # TODO
        self.count += len(self.shuffle_batch)
        # logging.info(f"[IterableDataset] 获取{len(self.shuffle_batch)}条数据shuffle, 总数据: {self.count}, 耗时:{latency.count():.3f}s")
    @Decorator.timing
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
    return fids_list, labels_list, [item[2] for item in batch]


class LRModel(nn.Module):
    """ 逻辑回归
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

@Decorator.timing
def validate(model):
    test_data = DataLoader(StockIterableDataset(data_source_getter=RM.data_source.next_test), batch_size=1000, collate_fn=custom_collate)
    ## test
    results = []
    for fids_batch, label_batch , ins_batch in test_data:
        pred_batch = model(fids_batch)
        # print(fids_batch)
        # print(fidembeding.fid2embedding)
        # print(pred_batch)
        # input("..")
        for i in range(len(fids_batch)):
            fids = fids_batch[i]
            label = label_batch[i].item()
            ins = ins_batch[i]
            pred = pred_batch[i].item()
            # print("%s %s %s" %(ins.name, ins.date, pred))
            results.append({
                "name" : ins.name,
                "date" : ins.date,
                "pred" : pred,
                "fids" : fids,
                "label" : label,
                "certainly" : 1,
                "raw_label" : ins.label['next_7d_14d_mean_price']
            })
    results.sort(key = lambda x : -x["pred"])
    final_result = {
        "start_at" :  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "conf" : "no",
        "validate" : results
    }
    save_file = "runs/result.%s.json" %(datetime.now().strftime("%Y%m%d%H%M"))
    logging.info("预估结果保存到: %s, 数量: %s" %(save_file, len(results)))
    open(save_file, "w").write(json.dumps(final_result, cls=NumpyEncoder, indent = 4, ensure_ascii=False))
    return 

@Decorator.timing
def backward(loss):
    loss.backward()

@Decorator.timing
def opt_step(opt):
    opt.step()

@Decorator.timing
def main():
    latency = Latency()
    model = LRModel().to(RM.device)
    batch_size = RM.conf.train.batch_size
    train_data = DataLoader(StockIterableDataset(data_source_getter=RM.data_source.next_train), batch_size=batch_size, collate_fn=custom_collate)

    data = []
    RM.step = 0 
    learning_rate  = RM.conf.train.learning_rate
    logging.info("lr: %s bs: %s" %(learning_rate, batch_size))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for fids, label, _ in train_data:
        RM.step +=1
        step = RM.step
        optimizer.zero_grad()  # 清零梯度
        prediction = model(fids)
        loss = F.mse_loss(prediction, label.to(RM.device), reduction='sum')  # mean reduction，会导致低频的fid的梯度被大幅度降低, 所以fid单独拆出去
        backward(loss)# loss.backward() 
        # if 2422899029155717707 in fids:
        #     logging.info("fid: %s, label: %s, prediction: %s" %(model.fid2embedding[fid][0].item(), label, prediction))
        if step %100 == 1:
            Decorator.timing_stat() # 计算耗时
            logging.info("step[%s] loss = %.3f; Latency: %.2fs train/test_queue: %s/%s" %(step, loss.item(), latency.count(),  RM.data_source.train_queue.qsize(), RM.data_source.test_queue.qsize()))
        RM.emit_summary('prediction', prediction, step)
        RM.emit_summary('label', label, step)
        RM.emit_summary('loss', loss, step)
        model.post_process(step)
        opt_step(optimizer)# optimizer.step()  # 更新参数
        fidembeding.update_embedding(fids, step)
    
    # mprint(model.state_dict(), title = "模型dense state_dict权重")
    mprint(fidembeding.fid2embedding.state_dict(), title = "模型dense state_dict权重")
    # mprint(data)
    torch.save(model.state_dict(), 'model_weight.pth')
    logging.info("模型权重保存到: model_weight.pth")
    logging.info("总训练样本: %s step: %s, 耗时: %s" %(RM.data_source.train_count, step, latency.count()))
    ####
    validate(model)
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

# def test_fid2embedding():
#     print(fidembeding.get_embedding(12345, 5))
#     print(fidembeding.fid2embedding)
#     return 
# test_fid2embedding()
# exit(0)
if __name__ == "__main__":
    # test_data_loader()
    main()
    Decorator.timing_stat() # 计算耗时