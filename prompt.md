我有下面的代码:
1. 读取所有的股票: 
from global_resource import Resource
[Resource.ts_code_to_stock_info[k] for k in Resource.ts_code_to_stock_info ]
ts_code_to_stock_info返回类StockInfo
```python

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
    @classmethod
    def from_dict(cls, data: dict) -> 'StockInfo':
        data['update_time'] = datetime.strptime(data['update_time'], '%Y-%m-%d %H:%M:%S.%f')
        data['date2daily'] = {}
        return cls(**data)
    def daily(cls, date):
        if date not in cls.date2daily:
            cls.date2daily[date] = sql_api.get_stock_daily_info(cls.ts_code, date)[0]
        else:
            cls.date2daily[date] = None
        return cls.date2daily[date]
``` 

帮我实现一个功能: 
def get_fund_flow(date, days):
  """
    所有股票按industry分组，然后计算[date, date+days-1]的总成交额和(date -days, date-1)的差值，然后输出
  """