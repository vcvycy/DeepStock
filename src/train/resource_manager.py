from easydict import EasyDict
from util import enum_instance, Decorator
import yaml, logging
import torch
import time, os
from datetime import datetime
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
        self._train_save_dir = None
        return 
    @property
    def train_save_dir(self):
        if self._train_save_dir is None:
            self._train_save_dir = 'runs/train_%s' %(datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(self._train_save_dir, exist_ok=True)
        return self._train_save_dir
    @property
    def summary_writer(self):
        from datetime import datetime
        from torch.utils.tensorboard import SummaryWriter
        if self._summary_writer is None:
            writer_dir= self.train_save_dir
            logging.info("tensorboard writer dir: %s" %(writer_dir)) 
            self._summary_writer = SummaryWriter(writer_dir)
        return self._summary_writer
    def can_emit_summary(self):
        if self.step % 50 == 1 or self.step < 20: # 采样
            return True
        return False
    @Decorator.timing()
    def emit_summary(self, name, tensor, var=True, hist=False):
        if not self.can_emit_summary(): # 采样
            return
        step = self.step
        # TensorBoard记录平均值和直方图
        if isinstance(tensor, torch.Tensor):
            mean = torch.mean(tensor)
            self.summary_writer.add_scalar(f'{name}/mean', mean, global_step=step)
            if var and tensor.numel() > 1:
                self.summary_writer.add_scalar(f'{name}/var', tensor.var(), global_step=step)
            if hist:
                self.summary_writer.add_histogram(name, tensor, global_step=step)
        else:
            self.summary_writer.add_scalar(name, tensor, global_step=step)
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
            device = self.conf.env.device
            assert device in ["cpu", "mps"], "device must be cpu or mps"
            logging.info("device: %s" %(device))
            self._device = torch.device(device)
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