from easydict import EasyDict
from util import enum_instance
import yaml, logging
from queue import Queue
import threading
import pandas as pd
import time
from train.resource_manager import RM
from data.sqlite import sql_api 
from util import MeanCounter
# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def update_fid_and_date_avg_label():
    counter = MeanCounter()
    label_keys = set()
    conf = RM.conf
    
    max_ins = 1e10 if conf.data.get('max_ins') is None else conf.data.max_ins
    if max_ins != 1e10:
        input(f"样本数不完全: max_ins={max_ins}, 即调试模式，回车继续...")
    ins_num = 0
    valid_keys = ['next_7d_14d_mean_price']#, 'next_3d_close_price']
    for ins in enum_instance(conf.data.files, max_ins=max_ins, disable_tqdm = True):
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
                counter.add('fid', (fid, label_key), label[label_key])
    
    print("样本数量: %s" % (ins_num))
    
    # 计算并更新fid2label_count
    fid2label_count = counter.get_avg('fid')
    sql_api.update_fid_avg_label(fid2label_count)

    # 计算并更新date2label_count
    date2label_count = counter.get_avg('date')
    sql_api.update_date_avg_label_table(date2label_count)

if __name__ == "__main__":
    update_fid_and_date_avg_label()