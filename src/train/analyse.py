from train.resource_manager import RM  # 确保这个导入是有效的
from util import MeanCounter, f3
from utils3 import mprint
import logging
def get_fid2avg_label():
    """ 计算fid的平均label
    """
    fid2fc = {}  # 每个fid的raw_feature/extrated_features等
    def go(data_iter):
        counter = MeanCounter()
        total = 0
        for fids, label, ins in data_iter():
            # print("%s %s %.4f" %(ins.name, ins.date, label))

            for f in ins.feature:
                assert len(f.fids) == 1, "fids !=0"
                fid = f.fids[0]
                fid2fc[fid] = f 
            # print(label)
            for fid in fids:
                counter.add("all", fid, label)
            total += 1
        avg_label = counter.get_avg("all")
        items = []
        # print(avg_label)
        for fid in avg_label:
            xx, count = avg_label[fid]
            xx = int(xx*100000)/100000   # 保留3位小数
            fc = fid2fc[fid] 
            items.append({
                "slot" : fid>>54,
                "fid" : fid,
                "name" : fc.name,
                "raw_feature" : ", ".join(fc.raw_feature),
                "extracted_features" : ", ".join(fc.extracted_features),
                "avg_label" : xx,
                "count" : count
            })
        items.append({
            "slot" : 9999,
            "extracted_features" : "",
            "count" : total
        })
        items.sort(key =lambda x : (x['slot'], x['extracted_features']))
        mprint(items)
        return avg_label
    ## 数据分析时配置epoch = 1
    RM.conf.data.epoch = 1
    logging.info("训练集".center(100, "="))
    go(RM.data_source.get_train_data)
    logging.info("测试集".center(100, "="))
    go(RM.data_source.get_test_data)
    return 

if __name__ == "__main__":
    get_fid2avg_label()