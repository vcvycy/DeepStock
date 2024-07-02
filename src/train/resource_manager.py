from easydict import EasyDict
from util import enum_instance
import yaml, logging
class _RM:
    def __init__(self):
        self.date2thre = {}
        self.conf = EasyDict(yaml.safe_load(open("./train/train.yaml", 'r').read()))
        self._data_source = None  # 初始化为None
        return 
    @property
    def data_source(self):
        if self._data_source is None:
            from train.data_source import DataSourceFile
            self._data_source = DataSourceFile(self.conf)
        return self._data_source


######初始化#######
RM = _RM()

if __name__ == "__main__":
    ""
    print(RM.conf) 
    # RM.read_avg_label_table()
    for item in RM.data_source.get_train_data():
        input(item)