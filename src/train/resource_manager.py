from easydict import EasyDict
from util import enum_instance, Decorator
import yaml, logging
import torch
import time
# from train.fid_embedding import FidEmbedding
class _RM:
    def __init__(self):
        self.reset()
    def reset(self):
        """重置RM中的参数: 主要是data_source可以重新读
        """
        self.conf = EasyDict(yaml.safe_load(open("./train/train.yaml", 'r').read()))
        self._data_source = None  # 初始化为None
        self._device = None
        self._summary_writer  = None
        self.step = 0
        return 
    @property
    def summary_writer(self):
        from datetime import datetime
        from torch.utils.tensorboard import SummaryWriter
        if self._summary_writer is None:
            writer_dir= 'runs/%s' %(datetime.now().strftime('%Y%m%d_%H%M%S'))
            logging.info("tensorboard writer dir: %s" %(writer_dir)) 
            self._summary_writer = SummaryWriter(writer_dir)
        return self._summary_writer
    @Decorator.timing
    def emit_summary(self, name, tensor, step = None):
        if step is None:
            step = self.step
        if step % 50 != 1: # 采样
            return
        # TensorBoard记录平均值和直方图
        mean = torch.mean(tensor)
        self.summary_writer.add_scalar(f'{name}/mean', mean, global_step=step)
        if tensor.numel() > 1:
            self.summary_writer.add_scalar(f'{name}/var', tensor.var(), global_step=step)
        # self.summary_writer.add_histogram(name, tensor, global_step=step)
        return
    @property
    def data_source(self):
        if self._data_source is None:
            from train.data_source import DataSourceFile
            self._data_source = DataSourceFile(self.conf)
        return self._data_source
    @property
    def device(self):
        if self._device is None: 
            self._device = torch.device("cpu")
            # if torch.backends.mps.is_available():
            #     self._device = torch.device("mps")
            # else:
            #     # device = torch.device("cpu")
            #     raise Exception("MPS device not found. Using CPU instead.")
        return self._device
    

######初始化#######
RM = _RM()

if __name__ == "__main__":
    """
    """
    print(RM.conf) 
    # RM.read_avg_label_table()
    # for item in RM.data_source.get_train_data():
    #     # input(item)
    #     pass
    print("slotNum: %s" %(RM.data_source.slot_num))