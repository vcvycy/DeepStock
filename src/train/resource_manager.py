from easydict import EasyDict
from util import enum_instance
import yaml, logging
from data.sqlite import sql_api
class _RM:
    def __init__(self):
        self.date2thre = {}
        self.conf = EasyDict(yaml.safe_load(open("./train/train.yaml", 'r').read()))
        return 

    def read_avg_label_table(self):
        # 获取所有可用的 key
        # for key in sql_api.read_date_avg_label():
        #     print(key)

        # 获取指定 key 的数据
        date2label = sql_api.read_date_avg_label('next_7d_14d_mean_price')
        print(date2label)

        # # 获取所有可用的 key
        # for key in sql_api.read_fid_avg_label():
        #     print(key)

        # # 获取指定 key 的数据
        # for fid_avg_label in sql_api.read_fid_avg_label('next_7d_14d_mean_price'):
        #     print(fid_avg_label)
        return 

######初始化#######
RM = _RM()

if __name__ == "__main__":
    print(RM.conf) 
    RM.read_avg_label_table()