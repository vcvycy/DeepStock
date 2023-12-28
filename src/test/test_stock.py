
from data.sqlite import sql_api 
from data.sqlite.define import TABLE 
from dataclasses import dataclass
from datetime import datetime
import logging
import numpy as np
from global_resource import Resource
"""
    1. 筛选出 stock
        a. 固定股票, 如沪深300
        b. 根据条件筛选: 如市值、品类、某一段时间的涨跌
       返回list<ts_code>
    2. 分group
        a. 根据市值分组
        b. 根据品类分组
        返回list<list<ts_code>>
    3.  每一组统计指标, 如上涨
"""

class GroupByMethod:
    """ 股票分组
    """
    @staticmethod
    def depend_on_self(stocks, group_func):
        group2list = {}
        for stock in stocks:
            group_name = group_func(stock) 
            if group_name not in group2list:
                group2list[group_name] = []
            group2list[group_name].append(stock)
        return group2list
    @staticmethod
    def sort_and_group(stocks, sort_fun, groups= 10):
        """ 对股票进行排序，然后分成groups组
        """
        group2list = {}
        stocks.sort(key = sort_fun)
        size = len(stocks)
        split_size, remainder = divmod(size, groups)
        sizes = [split_size + 1 if i < remainder else split_size for i in range(groups)] 
        start = 0
        prev = "?"
        for i, split in enumerate(sizes):
            cur_stocks = stocks[start:start + split]
            next = sort_fun(cur_stocks[-1])
            group = f"Group[{i}]({prev}-{next})"
            group2list[group] = cur_stocks
            prev = next
            start += split
        return group2list 

class BaseSolver:
    def __init__(self):
        self.all_stocks = [Resource.ts_code_to_stock_info[k] for k in Resource.ts_code_to_stock_info ]
        return  
    def split_to_groups(self, arr, group_num):
        """ 把数组Arr分成group_num组
        """
        size = len(arr)
        split_size, remainder = divmod(size, group_num)
        sizes = [split_size + 1 if i < remainder else split_size for i in range(group_num)] 
        start = 0
        groups = []
        for i, split in enumerate(sizes):
            groups.append(arr[start:start + split] )
            start += split
        return groups
    def filter(self, stock):
        return True
    
    def grouper(self, stocks):
        """ 对股票进行排序, 然后分成group_num组
            返回group_name -> [stock1, stock2, ..]
        """
        def sort_fun(stock):
            return int(stock.total_mv/1e4)
        print("按市值从小到大")
        group_num = 10
        group2list = {}
        stocks.sort(key = sort_fun)
        stock_groups = self.split_to_groups(stocks, group_num)
        prev = "?"
        for i, cur_stocks in enumerate(stock_groups):
            next = sort_fun(cur_stocks[-1])
            group = f"Group[{i}]({prev}-{next})"

            group2list[group] = cur_stocks
            prev = next 
        return group2list 
    
    def scorer(self, stock):
        """ 给一个股票打分
        """
        if not hasattr(self, 'tscode2rise'):
            st = '20221031'
            ed = '20221206' 
            self.tscode2rise = sql_api.get_stock_rise_between_dates(st, ed)
        if stock.ts_code not in self.tscode2rise:
            return 
        
        open, close = self.tscode2rise[stock.ts_code]
        return close / open -1
    
    def deal(self):
        ######## 过滤股票 ########
        print("过滤股票".center(100, "-"))
        all_stocks = [Resource.ts_code_to_stock_info[k] for k in Resource.ts_code_to_stock_info ]
        stocks = []
        for stock in all_stocks:
            if self.filter(stock):
                stocks.append(stock)

        print("Filter后stock数量: %s 例如: %s" %(len(stocks), stocks[0].name)) 

        ######## 分组 ########
        print("股票分组".center(100, "-"))
        group2stock = self.grouper(stocks)
        print(f"分成 {len(group2stock)} 组")

        ######## 计算值 ########
        print("统计差值".center(100, "-"))
        for group in group2stock:
            stocks = group2stock[group]
            scores = [self.scorer(stock) for stock in stocks if self.scorer(stock) is not None]
            print(f"{group}, 平均得分: {np.mean(scores):.3f} size {len(scores)}  {stocks[-1].name}")
        return  

class industrySolver(BaseSolver):
    def __init__(self): 
        st = '20221018'
        mid = '20221118' 
        ed = '20221230' 
        self.stock2open_close_before = sql_api.get_stock_rise_between_dates(st, mid)
        self.stock2open_close_after = sql_api.get_stock_rise_between_dates(mid, ed)
        return 
    def filter(self, stock):
        return stock.ts_code in self.stock2open_close_before and stock.ts_code in self.stock2open_close_after

    def grouper(self, stocks):
        """ 对股票进行排序, 然后分成group_num组
        """
        def sort_fun(stock):
            open, close = self.stock2open_close_before[stock.ts_code] 
            return  close / open -1 
            # return int(stock.total_mv/1e4)
        group_num = 10
        group2list = {}
        stocks.sort(key = sort_fun)
        stock_groups = self.split_to_groups(stocks, group_num)
        prev = None
        for i, cur_stocks in enumerate(stock_groups):
            if prev is None:
                prev = sort_fun(cur_stocks[0])
            next = sort_fun(cur_stocks[-1])
            group = f"Group[{i}]({prev:.3f}~{next:.3f})"
            group2list[group] = cur_stocks
            prev = next 
        return group2list 

    def scorer(self, stock):
        """ 给一个股票打分
        """ 
        try:
            open, close = self.stock2open_close_after[stock.ts_code]
            return close / open -1
        except:
            return 

if __name__ == "__main__":
    industrySolver.deal()