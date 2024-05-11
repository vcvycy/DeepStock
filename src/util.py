import logging
from datetime import datetime, timedelta 
import struct
# 在其他包之前配置basicConfig
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_memory_usage():
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss  # 获取实际物理内存占用，单位为字节
    memory_usage_mb = memory_usage / (1024 * 1024)  # 转换为MB 
    # print(f"当前内存占用：{memory_usage_mb} MB")
    return memory_usage_mb

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
    bar = tqdm(total = 2000000)
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

if __name__ == "__main__":
    print()
    # print(get_date(-1))
    get_memory_usage()