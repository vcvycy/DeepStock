import tushare as ts
import threading
import logging
import time
from util import get_date
"""
   global config
"""
client = ts.pro_api("009c49c7abe2f2bd16c823d4d8407f7e7fcbbc1883bf50eaae90ae5f")
"""
  装饰器, 失败重试(一般是qps超过阈值失败)
"""
# tushare_retry_lock = threading.Lock()   # 锁, 多线程读tushare
def _tushare_decorator(fun):  
    def wrapper(*args, **kwargs):
        # return fun(*args, **kwargs)
        retry = 6
        for i in range(retry):
            if i > 0:
                time.sleep(20)
            try:
                return fun(*args, **kwargs)
            except Exception as e:
                logging.error("[TushareApi-%s] exception: %s, retry: %s/%s, args: %s" %(fun.__name__, e, i+1, retry, args))
                
                if "抱歉，您每分钟最多访问该接口200次" in str(e):
                    time.sleep(10)
                else:
                    time.sleep(1)
        # 如果6次都失败则跑到这里
        raise Exception("[TushareApi] 重试%s次仍未成功 exit" %(retry))
        os._exit()
    return wrapper

@_tushare_decorator
def read_stock_basic():
    """
      读取所有股票列表, 返回df
    """
    return client.query('stock_basic', exchange='', list_status='L')

@_tushare_decorator
def read_stock_company():
    """ 股票对应的公司信息
    """
    return client.stock_company(exchange='', fields='ts_code,chairman,manager,secretary,reg_capital,setup_date,province,city, employees')

@_tushare_decorator
def read_stock_daily(ts_code, start_date = ''):
    """读取股票日线数据
    """
    return client.query('daily', ts_code=ts_code, start_date=start_date)

@_tushare_decorator
def read_daily_basic():
    """最新的basic信息: 市值、市净率
    """
    # 找到最近一天可以获取数据的
    delta = 0
    while True:
        date = get_date(delta = delta) 
        df = client.query('daily_basic', ts_code='', trade_date=date)
        if df.shape[0] > 0: 
            return df
        else:
            delta -= 1 

@_tushare_decorator
def get_trade_date():
    """ 获取可以交易的日期, 日期从大到小
    """
    df = client.query('trade_cal')
    return list(df[df['is_open']==1]['cal_date'])

if __name__ == "__main__":
    print(read_daily_basic())