from easydict import EasyDict
from util import enum_instance
from utils3 import mprint, coloring
import yaml, logging
from queue import Queue
import sys, os
import threading
import re
import time
from data.sqlite import sql_api
""" 从训练文件读取数据，多线程把数据写到 train_queue, test_queue。然后每次可以从这里读取数据
    包括 : 
        (1) 读取instances : 
            a. label -avg(label)
            b. slot 黑名单/白名单
            c. 拆分训练集、测试集
        (2) filter
        (3) 处理fid, label并返回
"""
class DataSource():
    """ 从文件，还是从sqlite3读取
    """
    def __init__(self, conf):
        self.train_count = 0
        self.test_count = 0
        self.train_queue = Queue()
        self.test_queue = Queue()
        self.is_finished = False
        self.conf = conf
        self.filter_reason = {}
        self.lkey = conf.data.label.key
        self._date2label = None 
        self._slot_num = None  #保证所有slot数量一致
        # slot白名单
        self.slot_whitelist = [int(n) for n in str(self.conf.data.get("slot_whitelist", '')).split(',') if n != ""]
        self.slot_blacklist = [int(n) for n in str(self.conf.data.get("slot_blacklist", '')).split(',') if n != ""]

        if len(self.slot_whitelist) > 0:
            logging.info("slot_whitelist: %s" %(self.slot_whitelist))
        if len(self.slot_blacklist) > 0:
            logging.info("slot_blacklist: %s" %(self.slot_blacklist))
        # mprint(self.date2label)
        threading.Thread(target=self.thread_func).start()
        pass
    @property
    def date2label(self):
        while self._date2label is None:
            self._date2label = sql_api.read_date_avg_label(self.lkey)
        return self._date2label
    @property
    def slot_num(self):
        while self._slot_num is None:
            time.sleep(0.1)
        return self._slot_num
    def filter(self, item):
        fids, label, ins = item
        filters = self.conf.data.filters
        if not filters.get("enable"):
            return False
        if filters["only_etf"]:
            return "ETF" not in ins.name #or "LOF" not in ins.name
        if "valid_tscode" in filters:
            name = "valid_tscode" 
            conf = filters[name]
            reg = conf["regexp"]
            if conf.get("enable"):
                if len(re.findall(reg, ins.ts_code)) == 0:
                    self.filter_reason[name] = self.filter_reason.get(name, 0) + 1
                    return True
        if "fid_filter" in filters:
            conf = filters["fid_filter"]
            if conf.get("enable"): 
                if not hasattr(self, "filter_fids"):
                    self.filter_fids = set(conf.get("fids"))
                filter_fids = self.filter_fids
                for fid in fids:
                    if fid in filter_fids:
                        need_filter = True
                        r = "fid_filter_%s" %(fid)
                        self.filter_reason[r] = self.filter_reason.get(r, 0) + 1
                        # print("过滤: %s %s" %(ins.name, ins.date))
                        return True
        # 退市股过滤
        if "退" in ins.name:
            self.filter_reason["退市"] = self.filter_reason.get("退市", 0) + 1
            return True
        return False

    def post_process(self, ins):
        fids = []
        date = ins.date 
        try:
            label = ins.label[self.lkey] 
            if self.conf.data.label.get("sub_avg_label", False):
                label -= self.date2label[date] # 除去平均label 
        except Exception as e:
            mprint(self.date2label)
            logging.error("出错: %s, 可能是每日平均label不存在, 需要重新运行: stock_update_train_avg_label" %(e))
            os._exit(0)
        
        for f in ins.feature:
            assert len(f.fids) == 1, "fids !=0"
            fid = f.fids[0]
            slot = fid >> 54
            if len(self.slot_whitelist) > 0 and slot not in self.slot_whitelist:
                # 白名单有定义且不在白名单之内
                continue
            if len(self.slot_blacklist) > 0 and slot in self.slot_blacklist:
                # 在黑名单内
                continue
            fids.append(fid) 
            # 减少内存占用
            if os.getenv('ins_memory_optimize') is not None:
                f.ClearField("raw_feature")
                f.ClearField("extracted_features")
        # if os.getenv('ins_memory_optimize') is not None:
        #     ins.ClearField("label")   # 清除会导致update_avg_label_table无法读取label
        # 需要保证所有样本的slot一致
        if True:
            slots = set([f >> 54 for f in fids])
            if self._slot_num is None:
                self._slot_num = len(slots) 
            assert len(slots) == self._slot_num , "slot数量不一致 %s != %s" %(len(slots), len(self._slot_num))
        fids.sort(key = lambda fid : fid >> 54)
        return fids, label, ins
    def enum_instance(self):
        print("enum_instance未实现")
        raise NotImplementedError

    def next(self, is_train = True):
        queue = self.train_queue if is_train else self.test_queue
        while not queue.empty() or self.is_finished == False:
            if queue.empty():
                # print("获取训练数据为空, 等待读取新数据...")
                time.sleep(0.1)
                continue
            return queue.get()
        return None
    def next_train(self):
        return self.next(is_train= True)
    def next_test(self):
        return self.next(is_train = False)
    def get_train_data(self):
        while True:
            item = self.next(is_train = True)
            if item is not None:
                yield item
            else:
                break
        return 

    def get_test_data(self):
        while True:
            item = self.next(is_train = False)
            if item is not None:
                yield item
            else:
                break
        return 
    def thread_func(self):
        conf = self.conf
        for e in range(conf.data.epoch):
            dedup = set()
            for ins in self.enum_instance():
                while self.train_queue.qsize() >= 10000:
                    time.sleep(1)
                date = ins.date 
                if (ins.date, ins.ts_code) in dedup:
                    continue
                dedup.add((ins.date, ins.ts_code))
                item =  self.post_process(ins)    # 这一步耗时占enum_ins : 50%
                if self.filter(item):
                    continue
                # if len(set(["2427379723444939871", 2422899029155717707, 2423052998401735631]) & set(item[0])) == 0:
                #     continue
                if date <= conf.data.train_test_date:
                    self.train_count += 1 
                    self.train_queue.put(item)
                else:
                    if e == 0:
                        self.test_count += 1
                        self.test_queue.put(item)  # 修正这里
        if len(self.filter_reason) > 0:
            logging.info("[data_source] 过滤: ")
            mprint(self.filter_reason)
        logging.info("[data_source]训练集数量: %s, 还未消费数据%s" % (self.train_count, self.train_queue.qsize()))
        logging.info("[data_source]测试集数量: %s 还未消费数据%s" % (self.test_count, self.test_queue.qsize()))
        logging.info("[data_source]总数据量: %s" %(self.train_count + self.test_count))
        self.is_finished = True
        return 
class DataSourceFile(DataSource):
    def __init__(self, conf):
        super().__init__(conf)
    def enum_instance(self):
        conf = self.conf
        max_ins =  1e10 if conf.data.get("max_ins") is None else conf.data.max_ins
        for ins in enum_instance(conf.data.files, max_ins = max_ins, disable_tqdm = self.conf.data.disable_tqdm):
            yield ins
        return 
        

if __name__ == "__main__":
    from train.resource_manager import RM
    """python -m train.data_source"""
    print("多线程train.yaml中读取 data.files, ")
    file_source = DataSourceFile(RM.conf)
    for item in file_source.get_train_data():
        # fids, label, ins = item
        # print(ins)
        # print(fids)
        # print(label)
        # input("..")
        pass
        # print(ins.ts_code)