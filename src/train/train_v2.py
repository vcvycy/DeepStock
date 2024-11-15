import random
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import logging
from train.resource_manager import RM  # 确保这个导入是有效的
from util import *
from utils3 import mprint, coloring
import time
from datetime import datetime
import json
from train.my_model import *
import os, sys
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import threading
from tqdm import tqdm


class StockIterableDataset(IterableDataset):
    """输入 fids  = [3234, 46457, 6745, ...]
       返回: embedding = {slot1 : torch.tensor(dim = 128), slot2: torch.tensor(dim = 124), ...}
    """
    def __init__(self, data_source_getter):
        super(StockIterableDataset, self).__init__()
        self.count = 0
        self.data_source_getter = data_source_getter

    def __iter__(self):
        return self
    @Decorator.timing(func_name = "next-获取数据")
    def __next__(self):
        item =  self.data_source_getter()
        if item is None:
            raise StopIteration
        self.count += 1
        return item
def custom_collate(batch):
    fids_list = [item[0] for item in batch]
    labels_list = torch.tensor([item[1] for item in batch])
    return fids_list, labels_list, [item[2] for item in batch]


@Decorator.timing()
def validate(model, test_data_list):
    if len(test_data_list) == 0:
        logging.info("validation数据尚未ready...")
        return 
    logging.info("正在跑validation数据集....")
    # test_data = DataLoader(StockIterableDataset(data_source_getter=RM.data_source.next_test), batch_size=1000, collate_fn=custom_collate)
    ## test
    results = []
    for fids_batch, label_batch , ins_batch in test_data_list:
        pred_batch = model(fids_batch)
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
    save_model_weight(model, RM.step)
    save_file = "%s/result.step_%s.json" %(RM.train_save_dir, RM.step)
    logging.info("预估结果保存到: %s, 数量: %s" %(save_file, len(results)))
    open(save_file, "w").write(json.dumps(final_result, cls=NumpyEncoder, indent = 4, ensure_ascii=False))

    #### 评估结果: 跑网站得到结果 ####
    @Decorator.timing(func_name = "validation-eval_result_from_web")
    def eval_result_from_web(save_file, step):
        try:
            import requests
            find_better_model = False
            for topk in [2, 4, 8, 16]:
                url = "http://localhost:8080/model_result_process"
                post_params = {
                    "min_certainly":0,
                    "path": os.path.abspath(save_file),
                    "topk":topk,
                    "dedup":True
                }
                text = requests.post(url, data=post_params).text
                rsp = json.loads(text)[0]
                summary = rsp['summary']
                RM.emit_summary(f'validation/top_{topk}/avg', 100*summary['return_all'], var=False, emit_anyway=True)
                RM.emit_summary(f'validation/top_{topk}/p50', 100*summary['return_p50'], var=False, emit_anyway=True)
                logging.info("评估结果-topk(%s) step:%s  %s" %(topk, step, rsp['summary'])) 
                if topk == 2 and summary['return_all'] > RM.validation_best:
                    RM.validation_best = summary['return_all']
                    find_better_model = True
            # 删除文件
            if not find_better_model:
                os.remove(save_file)
        except Exception as e:
            print(post_params)
            logging.error("评估结果失败: %s" %e)
            raise e
    # eval_result_from_web(save_file, RM.step)
    threading.Thread(target=eval_result_from_web, args=(save_file, RM.step, )).start()
    return 

@Decorator.timing(func_name = "backward-计算梯度")
def backward(loss):
    loss.backward()

@Decorator.timing(func_name = "opt_step-更新梯度")
def opt_step(opt):
    opt.step()

def save_model_weight(model, step):
    model_weight_file = "%s/model_weight_step_%s.pth" %(RM.train_save_dir, step)
    torch.save(model.state_dict(), model_weight_file)
    logging.info("模型权重保存到: %s" %(model_weight_file))
# @Decorator.timing()
def main():
    latency = Latency()
    model = DistillModel().to(RM.device)
    batch_size = RM.conf.train.batch_size
    train_data = DataLoader(StockIterableDataset(data_source_getter=RM.data_source.next_train), batch_size=batch_size, collate_fn=custom_collate)
    test_data = DataLoader(StockIterableDataset(data_source_getter=RM.data_source.next_test), batch_size=1000, collate_fn=custom_collate)
    test_data_list = []
    data = []
    RM.step = 0 
    learning_rate  = RM.conf.train.learning_rate
    weight_decay  = RM.conf.train.weight_decay
    logging.info("lr: %s weight_decay: %s bs: %s" %(learning_rate, weight_decay, batch_size))
    # for name, p in model.named_parameters():
    #     logging.info("model parameters: %s %s" %(name, p.size()))
    logging.info("模型参数:")
    mprint({k : v.size() for k, v in model.named_parameters()})
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), betas = (0.9, 0.99), weight_decay = weight_decay, lr=learning_rate)
    # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    
    # scheduler = StepLR(optimizer, step_size=500, gamma=0.3)  # 学习率每隔step_size个step，就下降到0.1倍
    scheduler = CosineAnnealingLR(optimizer, T_max=40000, eta_min=1e-4)
    validation_step = RM.conf.validation.step
    logging.info("配置: 每隔%s步跑一次validation" %(validation_step))
    bar = tqdm(total = 50000) 
    for fids_batch, label, _ in train_data:
        try:
            label = label.to(RM.device)
            RM.step +=1
            step = RM.step
            optimizer.zero_grad()  # 清零梯度
            prediction, loss = model(fids_batch, label = label)
            # loss = F.mse_loss(prediction, label.to(RM.device), reduction='sum')  # mean reduction，会导致低频的fid的梯度被大幅度降低, 所以fid单独拆出去
            backward(loss)
            if step % RM.conf.train.log_step == 1:
                Decorator.timing_stat() # 计算耗时
                logging.info("step[%s] loss = %.3f; Latency: %.2fs train/test_queue: %s/%s" %(step, loss.item(), latency.count(),  RM.data_source.train_queue.qsize(), RM.data_source.test_queue.qsize()))
            
            RM.emit_summary('core/prediction', prediction, hist = True)
            RM.emit_summary('core/label', label, hist = True)
            RM.emit_summary('core/loss', loss)
            RM.emit_summary('core/learning_rate', optimizer.param_groups[0]['lr']) 
            # 工程侧监控
            RM.emit_summary('engineer/train_queue', RM.data_source.train_queue.qsize())
            RM.emit_summary('engineer/test_queue', RM.data_source.test_queue.qsize())

            model.post_process(step)
            opt_step(optimizer)# optimizer.step()  # 更新参数
            scheduler.step()
            if len(test_data_list) == 0 and RM.data_source.is_test_finished:
                test_data_list = list(test_data)
            
            if RM.step % validation_step == 0:
                validate(model, test_data_list)
            # 进度条
            if RM.step % 5 == 1:   
                bar.set_postfix({
                    "train_queue": RM.data_source.train_queue.qsize(), 
                    "test_queue": RM.data_source.test_queue.qsize(),
                    "loss": loss.item(),
                    "latency_per_step": "%.2f秒" %(latency.count()/RM.step),
                    "device" : RM.conf.env.device
                })
                bar.update(5)
        except KeyboardInterrupt:
            logging.info("手动退出训练过程...")
            break

    model.embed_show()
    logging.info("总训练样本: %s step: %s, 耗时: %s" %(RM.data_source.train_count, coloring(step), latency.count()))
    #### 跑完所有数据后，再跑一次验证
    validate(model, test_data_list)
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
    os.environ['ins_memory_optimize'] = "1"  # data_source
    # test_data_loader()
    main()
    Decorator.timing_stat() # 计算耗时