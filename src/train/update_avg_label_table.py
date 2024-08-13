from easydict import EasyDict
from util import enum_instance
from utils3 import read_config_json_yaml, coloring
import yaml, logging
from queue import Queue
import threading
import pandas as pd
import time
from train.resource_manager import RM
from data.sqlite import sql_api 
from util import MeanCounter, mprint
from utils3 import coloring
import argparse

# 配置日志记录器

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def update_date_avg_label_table():
    """
        更新每天的平均label
    """
    logging.info("更新每日平均label".center(100, '='))
    RM.reset()
    RM.conf.data.epoch = 1
    RM.conf.data.label.sub_avg_label = False   # 不去取平均label, 因为平均label是在这里计算的
    
    counter = MeanCounter() 
    conf = RM.conf
    ins_num = 0
    fid2feature = {}
    while True:
        item  = RM.data_source.next_train() or RM.data_source.next_test()
        if item is None:
            break
        fids, label, ins = item
        ins_num += 1 
        label = ins.label
        date = ins.date 
        for label_key in ['next_7d_14d_mean_price']:
            if label_key not in label:
                continue 
            #### 计算date2label
            counter.add('date', (date, label_key), label[label_key])

    # 计算并更新date2label_count
    date2label_count = counter.get_avg('date')
    sql_api.update_date_avg_label_table(date2label_count)
    logging.info("[每天平均label]参与计算的样本数量: %s" % (coloring(ins_num)))
    return

def update_fid2avg_label_table():
    """更新fid的平均label : 只计算train_data
    """
    logging.info("更新fid的平均label".center(100, '='))
    RM.reset()
    RM.conf.data.epoch = 1
    # print(RM.conf.data.label)

    counter = MeanCounter() 
    conf = RM.conf
    ins_num = 0
    fid2feature = {}
    while True:
        item  = RM.data_source.next_train()
        if item is None:
            break
        fids, label, ins = item
        ins_num += 1 
        for f in ins.feature:
            fid = f.fids[0]
            if fid not in fids:  # 被过滤的fid不计算
                continue
            fid2feature[fid] = f
            counter.add('fid', (fid, 'train_label'), label) 
    
    
    # 计算并更新fid2label_count
    fid2label_count = counter.get_avg('fid')
    # mprint(fid2label_count)
    sql_api.update_fid_avg_label(fid2label_count, fid2feature)
    logging.info("[fid平均label]参与计算的样本数量: %s" % (coloring(ins_num)))
    return

def show_fid_avg_label_table():
    """展示fid平均label
    """
    logging.info("fid平均label如下".center(100, '='))
    feature_list = read_config_json_yaml("/Users/jianfeng/Documents/DeepLearningStock/src/feature_list.yaml")['feature_columns']
    slot2conf = {item['slot'] : item for item in feature_list}
    data = sql_api.read_fid_avg_label("next_7d_14d_mean_price", True)
    data.sort(key = lambda item : (item['key'], item['slot'] - item['avg_label']))
    ins_num = 0
    for i, item in enumerate(data):
        # if item['avg_label'] > 0.02:
        #     item['avg_label'] = coloring("%.3f" %item['avg_label']) 
        ins_num += item['count']
        item['idx'] = i+1
        feature_conf = slot2conf[item['slot']]
        # print(feature_conf)
        item['desc'] = feature_conf.get('name', "") + " | " + feature_conf.get("description", "")
    mprint(data, 
            col_names = ['idx', 'slot', 'fid', 'key', 'avg_label', 'count', 'desc', 'raw_feature', 'extracted_features'],
            title = "fid的平均label(总ins数量: %s)" %(ins_num))
    return 

if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true', help='更新')
    args = parser.parse_args()
    # 根据 --show 参数的值选择执行的逻辑
    if args.show:
        show_fid_avg_label_table()
    else:
        max_ins = 1e10 if RM.conf.data.get('max_ins') is None else RM.conf.data.max_ins
        if max_ins != 1e10:
            input(f"【友情提示】样本数不完全: max_ins={max_ins}, 即调试模式，回车继续...")
        update_date_avg_label_table()
        update_fid2avg_label_table()
        show_fid_avg_label_table()