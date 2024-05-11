from easydict import EasyDict
from util import enum_instance
import yaml, logging
from queue import Queue
import threading
import time
from train.resource_manager import RM
""" 从训练文件读取数据，多线程把数据写到 train_queue, test_queue。然后每次可以从这里读取数据
相关配置:
   
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
        threading.Thread(target=self.thread_func).start()
        pass

    def thread_func(self):
        raise NotImplementedError

    def get_train_data(self):
        while not self.train_queue.empty() or self.is_finished == False:
            if self.train_queue.empty():
                time.sleep(0.1)
                continue
            yield self.train_queue.get()
        return 

    def get_test_data(self):
        while not self.test_queue.empty() or self.is_finished == False:
            if self.test_queue.empty():
                time.sleep(0.1)
                continue
            yield self.test_queue.get()
        return 
class DataSourceFile(DataSource):
    def __init__(self, conf):
        super().__init__(conf)

    def post_process(self, item):
        input = []
        label = item.label
        for f in item.feature:
            assert len(f.fids) == 1, "fids !=0"
            input.append(f.fids[0])
        return input, label

    def thread_func(self):
        conf = self.conf
        dedup = set()
        for e in range(conf.data.epoch):
            max_ins =  1e10 if conf.data.get("max_ins") is None else conf.data.max_ins
            for ins in enum_instance(conf.data.files, max_ins = max_ins, disable_tqdm = True):
                date = ins.date 
                if (ins.date, ins.ts_code) in dedup:
                    continue
                dedup.add((ins.date, ins.ts_code))
                item = self.post_process(ins)  # 修正这里
                if date <= conf.data.train_test_date:
                    self.train_count += 1 
                    self.train_queue.put(item)
                else:
                    if e == 0:
                        self.test_count += 1
                        self.test_queue.put(item)  # 修正这里
        print("训练集数量: %s, %s" % (self.train_count, self.train_queue.qsize()))
        print("测试集数量: %s %s" % (self.test_count, self.test_queue.qsize()))
        print("总数据量: %s" %(self.train_count + self.test_count))
        self.is_finished = True
        return 
        

if __name__ == "__main__":
    """python -m train.data_source"""
    print("多线程train.yaml中读取 data.files, ")
    file_source = DataSourceFile(RM.conf)
    for item in file_source.get_train_data():
        print(item)
        input("..")