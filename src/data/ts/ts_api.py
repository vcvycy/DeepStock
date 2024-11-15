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
def read_stock_daily(ts_code='', start_date = '', end_date = ''):
    """读取股票日线数据
    """
    return client.query('daily', ts_code=ts_code, start_date=start_date, end_date = end_date)

@_tushare_decorator
def read_daily_basic(trade_date):
    """获取所有股票date这天的市值、pe等
    """
    # 找到最近一天可以获取数据的
    df = client.query('daily_basic', ts_code='', trade_date=trade_date)
    return df

@_tushare_decorator
def get_trade_date(future_date = False):
    """ 获取可以交易的日期, 日期从大到小
    """
    df = client.query('trade_cal')
    date_list = list(df[df['is_open']==1]['cal_date'])
    date_list = [d for d in date_list if d>='20000101'] # 2000年后的才写进去
    if future_date:
        return date_list
    else:
        today = get_date()
        return  [d for d in date_list if d <= today]


if __name__ == "__main__":
    # print(get_index_weight(name="沪深300"))
    print(read_daily_basic('20240926'))