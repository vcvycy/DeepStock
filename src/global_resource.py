from data.ts import ts_api 
from dataclasses import dataclass
from datetime import datetime
from data.sqlite import sql_api 
from data.sqlite.define import TABLE 
from util import date_add
import math
# from functools import wraps
# def decorator(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         new_args = []
#         for arg in args:
#             if isinstance(arg, datetime):
#                 # 将 datetime 参数转换为指定的日期字符串格式
#                 new_args.append(arg.strftime('%Y%m%d'))
#             else:
#                 new_args.append(arg)
        
#         # 对于关键字参数，也可以进行类似的检查和转换，如果需要的话
#         new_kwargs = {k: (v.strftime('%Y%m%d') if isinstance(v, datetime) else v) for k, v in kwargs.items()}
        
#         # 调用原始函数，并传入可能已转换的参数
#         return func(*new_args, **new_kwargs)
    
#     return wrapper

@dataclass
class StockDaily:
    ts_code: str
    trade_date: str  
    open: float
    high: float
    low: float
    close: float
    pre_close: float
    change: float
    pct_chg: float
    vol: float
    amount: float
    update_time: str

@dataclass
class StockInfo:
    ts_code: str
    symbol: str
    name: str
    area: str
    industry: str
    market: str
    list_date: str
    update_time: datetime
    # 下面是最新日期的数据 stock_daily_basic_table
    trade_date: str
    close: float
    turnover_rate: float
    turnover_rate_f: float
    volume_ratio: float
    pe: float
    pe_ttm: float
    pb: float
    ps: float
    ps_ttm: float
    dv_ratio: float
    dv_ttm: float
    total_share: float
    float_share: float
    free_share: float
    total_mv: float
    circ_mv: float
    # date2daily是每一天的成交数据 date(如'20230101') -> StockDaily
    date2daily: dict 
    #### 
    cnspell: str
    act_name: str
    act_ent_type : str
    @classmethod
    def from_dict(cls, data: dict) -> 'StockInfo':
        data['update_time'] = datetime.strptime(data['update_time'], '%Y-%m-%d %H:%M:%S.%f')
        data['date2daily'] = {}
        return cls(**data)
    def daily(cls, date, end_date = None):
        if date not in cls.date2daily:
            if end_date is None:
                cls.date2daily[date] = None
            else:
                cur_date = date
                while cur_date <= end_date:
                    cls.date2daily[cur_date] = None
                    cur_date = date_add(cur_date, 1)
            for item in sql_api.get_stock_daily_info(cls.ts_code, date, end_date):
                cls.date2daily[item['trade_date']] = StockDaily(**item)
        if end_date is None:
            return cls.date2daily[date]
    def rise(cls, date, end_date = None):
        if end_date is None:
            end_date = date
        # print("%s [%s ~ %s] %s %s " %(cls.name, date, end_date, cls.daily(end_date).close, cls.daily(date).pre_close))
        rise = (cls.daily(end_date).close - cls.daily(date).pre_close) / cls.daily(date).pre_close
        # print(rise)
        return rise


class _ResourceCLS:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(_ResourceCLS, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    
    def __init__(self):
        
        self._trade_date = None
        # ts_code 到 stock_info
        self._ts_code_to_stock_info = None
        return 
     
    @property
    def trade_dates(self):
        """ 交易日期 [ '19910628', '19910627', '19910626', ...]
        """
        if self._trade_date is None:
            self._trade_date = ts_api.get_trade_date()
        return self._trade_date
    def trade_date_add(self, trade_date, delta = 1):
        # 往前推k个交易日
        one_delta = 1 if delta > 0 else -1
        while delta != 0:
            assert trade_date >= '19900101' and trade_date <= '20990101', ""
            trade_date = date_add(trade_date, one_delta)
            if trade_date in self.trade_dates:
                delta -= one_delta
        return trade_date
    @property
    def ts_code_list(self):
        return list(self.ts_code_to_stock_info)
    
    @property
    def ts_code_to_stock_info(self):
        """ ts_code到stock_info
        """
        if self._ts_code_to_stock_info is None:
            query = """
                select * from stock_basic_table join stock_daily_basic_table on stock_daily_basic_table.ts_code = stock_basic_table.ts_code;
                """
            all_items = sql_api.simple_execute(query)
            # input(all_items[0])
            self._ts_code_to_stock_info = {item['ts_code'] : StockInfo.from_dict(item) for item in all_items}
        return self._ts_code_to_stock_info
    ####### 方法 ######
    def find_stock(self, keyword, key = "ts_code"):
        """ 查找股票, 方式: ts_code、股票名
        """
        stock_list = []
        for ts_code in self.ts_code_to_stock_info:
            stock_info = self.ts_code_to_stock_info[ts_code]
            if keyword in getattr(stock_info, key):
                stock_list.append(stock_info)
        return stock_list 

Resource = _ResourceCLS()

if __name__ == "__main__":
    # print(Resource.trade_dates)
    # print(Resource.trade_date_add('20231228', -4))
    stock = Resource.ts_code_to_stock_info['000001.SZ']
    print(stock)
    # print(stock.daily('20231120', '20231220'))
    # print(Resource.find_stock("茅台", key = 'name'))
    # for stock in Resource.find_stock("白酒", key = 'industry'):
    #     print(stock.name)