from easydict import EasyDict
from util import enum_instance
import yaml, logging
from queue import Queue
import threading
import pandas as pd
import time
from train.resource_manager import RM
from data.sqlite import sql_api 
from util import MeanCounter, mprint
import argparse

# 配置日志记录器

RM.conf.data.epoch = 1
RM.conf.data.label.sub_avg_label = False   # 不去取平均label

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def update_fid_and_date_avg_label():
    """
      更新2个表:
        fid  -> avg_label
        date -> avg_label
    """
    counter = MeanCounter()
    label_keys = set()
    conf = RM.conf

    max_ins = 1e10 if conf.data.get('max_ins') is None else conf.data.max_ins
    if max_ins != 1e10:
        input(f"样本数不完全: max_ins={max_ins}, 即调试模式，回车继续...")
    ins_num = 0
    valid_keys = ['next_7d_14d_mean_price']#, 'next_3d_close_price']
    fid2feature = {}
    while True:
        item  = RM.data_source.next_train() or RM.data_source.next_test()
        if item is None:
            break
        fids, label, ins = item
        ins_num += 1 
        label_keys = label_keys.union(set(ins.label.keys()))
        label = ins.label
        date = ins.date 
        for label_key in valid_keys:
            if label_key not in label:
                continue 
            #### 计算date2label
            counter.add('date', (date, label_key), label[label_key])
            #### 计算fid2label
            for f in ins.feature:
                fid = f.fids[0]
                if fid not in fids:  # 训练数据有，才会写入表里
                    continue
                fid2feature[fid] = f
                counter.add('fid', (fid, label_key), label[label_key])
    
    print("样本数量: %s" % (ins_num))
    
    # 计算并更新fid2label_count
    fid2label_count = counter.get_avg('fid')
    # mprint(fid2label_count)
    # input("上面为要写入的数据.. press any key to continue...")
    sql_api.update_fid_avg_label(fid2label_count, fid2feature)

    # 计算并更新date2label_count
    date2label_count = counter.get_avg('date')
    sql_api.update_date_avg_label_table(date2label_count)

def show_avg_label_table():
    data = sql_api.read_fid_avg_label("next_7d_14d_mean_price", True)
    data.sort(key = lambda item : item['avg_label'])
    mprint(data)
    return 

if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true', help='更新')
    args = parser.parse_args()
    # 根据 --show 参数的值选择执行的逻辑
    if args.show:
        show_avg_label_table()
    else:
        update_fid_and_date_avg_label()
        show_avg_label_table()