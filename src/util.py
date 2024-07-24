import logging
from datetime import datetime, timedelta 
import struct
import math
# 在其他包之前配置basicConfig
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import time
from collections import defaultdict
import json 
from utils3 import mprint

class Decorator:
    """统计函数执行时间
    example:
        # 示例函数
        @Decorator.timing
        def example_function():
            time.sleep(2)
        # 调用示例函数
        example_function()
        Decorator.stat()
    """
    # 定义静态成员变量
    time_elapsed = {}

    @staticmethod
    def timing(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            # 将执行时间存储到静态成员变量中
            Decorator.time_elapsed[func.__name__] = Decorator.time_elapsed.get(func.__name__, 0) +execution_time
            return result
        return wrapper

    @staticmethod
    def timing_stat():
        total = sum(Decorator.time_elapsed.values())
        items = []
        for key, val in Decorator.time_elapsed.items():
            items.append({
                "Func": key, 
                "TimeElapsed": "%.2f秒" % val,
                "ratio": "%.2f%%" % (val / total * 100)
            })
        items.append({
            "Func": "Total",
            "TimeElapsed": "%.2f秒" % total,
            "ratio": "100%"
        })
        mprint(items)
        return 
class MeanCounter:
    """ 用于计算数组的avg, 每次添加一个val，同时保存sum(val)和count(*)
    example: 
        counter = MeanCounter()
        counter.add('a', 'salary', 1000)
        counter.add('a', 'salary', 2000)
        print(counter.count['a']['salary'])
        exit(0)
    """
    def __init__(self):
        self.count = {}#'fid': defaultdict(lambda: [0, 0]), 'date': defaultdict(lambda: [0, 0])}
    
    def add(self, section, key, v):
        if section not in self.count:
            self.count[section] = defaultdict(lambda: [0, 0])
        # return 
        self.count[section][key][0] += 1
        self.count[section][key][1] += v

    def get_avg(self, section):
        if section not in self.count:
            return {}
        avg_dict = {}
        for k, (count, sum_val) in self.count[section].items():
            avg_dict[k] = (sum_val / count, count)
        del self.count[section]
        return avg_dict
class Latency:
    """
        latency = Latency()
        latency.count()  # 计算耗时
        latency.clear()  # 重新计时
    """
    def __init__(self):
        self.start_time = time.time()
    def count(self):
        elapsed_time = time.time() - self.start_time  # 计算并返回经过的时间
        return elapsed_time
    def reset(self):
        self.start_time = time.time()  # 重置start_time为当前时间，如果需要连续测量，则这行可选
        return 
#静态类
class Counter:
    data = {}  # 使用类变量存储计数数据

    @staticmethod
    def add(key, val=1):
        Counter.data[key] = Counter.data.get(key, 0) + val

    @staticmethod
    def get(key):
        return Counter.data.get(key, 0)  # 返回键对应的计数，若键不存在则返回0
def get_memory_usage():
    import psutil
    import os
    pid = os.getpid()
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    # return str(memory_info)
    memory_usage = memory_info.rss /2**20  # 获取实际物理内存占用，单位为字节 /vms获取包括虚拟内存的=
    return memory_usage

def read_file_with_size(f, PBClass = None):
    data_size_bin = f.read(8)
    if len(data_size_bin) < 8:
        return 0, None
    data_size = struct.unpack('Q', data_size_bin)[0]
    assert data_size < 2**20
    data = f.read(data_size)
    if PBClass is not None:
        obj = PBClass()
        obj.ParseFromString(data)
        data = obj
    return data_size, data

def enum_instance(path, max_ins = 1e10, disable_tqdm = False):
    """
      path : 训练文件，可以单个，或者多个
      max_ins: 最多读取多少样本
    """
    from common.stock_pb2 import Instance
    from tqdm import tqdm
    if not isinstance(path, list):
        path = [path]
    bar = tqdm(total = 2000000) if not disable_tqdm else None
    hash_set = set()
    for p in path:
        f = open(p, "rb")
        while True:
            size, data = read_file_with_size(f, Instance)
            if size == 0 or max_ins <= 0:
                break
            hash_key = data.ts_code + data.date
            if hash_key in hash_set:
                continue
            hash_set.add(hash_key)
            max_ins -= 1
            if not disable_tqdm:
                bar.set_postfix({"内存": f"{int(get_memory_usage())}MB" })
                bar.update(1)
            yield data
    return 
# class MYDate():
#     def __init__():

# def date_format(date):
#     if isinstance(date, datetime):
#         return date.strftime('%Y%m%d')
#     else:
#         return date

def date_add(date_string, delta = 1):
    """ 日期字符串+1, 如20220131变成20220201
    """
    date_format = "%Y%m%d"
    # 将日期字符串解析为日期对象
    date = datetime.strptime(date_string, date_format)
    new_date = date + timedelta(days=delta) 
    new_date_string = new_date.strftime(date_format)
    return new_date_string

def get_date(delta = 0):
    """ 获取当前date: 如返回20231001, 返回字符串
    """
    return date_add(datetime.now().strftime('%Y%m%d'), delta)

def date_diff(date_str1, date_str2):
    """
    计算两个日期字符串之间的天数差。
    example: date_diff("20220101", "20240101")
    """
    # 定义日期格式
    date_format = "%Y%m%d"
    # 将日期字符串解析为日期对象
    date1 = datetime.strptime(date_str1, date_format)
    date2 = datetime.strptime(date_str2, date_format)
    # 计算两个日期之间的天数差
    date_diff = (date2 - date1).days
    return date_diff

def f3(data):
    """保留3位有效小数, x为数组/数字都可以
    """
    def _f3(f):
        print("f=%s" %(f))
        return int(f*1000)/1000
    if isinstance(data, list):
        return [_f3(i) for i in data] 
    else:
        return _f3(data)
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

if __name__ == "__main__":
    latency = Latency()
    print(latency.count())
    print(latency.count())
    # @Decorator.timing
    # def example_function():
    #     time.sleep(2)
    # # 调用示例函数
    # example_function()
    # Decorator.stat()
    # exit(0)
    # # print()
    # # print(get_date(-1))
    # # get_memory_usage()
    # is_support_gpu()
    # for ins in enum_instance("/Users/jianfeng/Documents/DeepLearningStock/training_data/data.daily.20240608_0124"):
    #     pass