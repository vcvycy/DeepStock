from global_resource import Resource
from datetime import datetime, timedelta
import numpy as np
# from util import date_add

def get_fund_flow(industry, date, days):
    # 将字符串日期转换为datetime对象
    start_date = Resource.trade_date_add(date, -days+1)
    end_date = date

    prev_start_date = Resource.trade_date_add(start_date, -days)
    prev_end_date = Resource.trade_date_add(start_date, -1)

    # 初始化industry的成交额字典
    industry_amount = {}
    # 读取所有股票信息
    stocks_info = [Resource.ts_code_to_stock_info[k] for k in Resource.ts_code_to_stock_info]
    # print(f"行业【{industry}】 {prev_start_date}~{prev_end_date}  {start_date}~{end_date}")
    # 遍历所有股票
    num = 0
    current_period_amount = 0
    previous_period_amount = 0 
    prev_price = []
    price = []
    for stock in stocks_info:
        try:
            # if '茅台' not in stock.name:
            #     continue
            if stock.industry != industry:
                continue
            num += 1 

            stock.daily(prev_start_date, end_date) 
            # 遍历当前时段内的日期
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date
                daily_info = stock.daily(date_str)
                if daily_info:
                    current_period_amount += daily_info.amount
                # print(f"当前时段 {date_str} {daily_info}")
                current_date = Resource.trade_date_add(current_date, 1)

            # 遍历前一个时段内的日期
            prev_date = prev_start_date
            while prev_date <= prev_end_date:
                date_str = prev_date
                daily_info = stock.daily(date_str)
                if daily_info:
                    previous_period_amount += daily_info.amount
                prev_date = Resource.trade_date_add(prev_date, 1)
            # try:
            # print(type(start_date))
            prev_price.append(stock.rise(prev_start_date, prev_end_date))
            # print(stock)
            price.append(stock.rise(start_date, end_date)) 
        except:
            print("股票失败 %s" %(stock.name))
            pass 
    # print(prev_price)
    # print(price)
    # exit(0)
    return num, current_period_amount - previous_period_amount, np.mean(prev_price), np.mean(price)

# 使用函数
# 假设我们要查看'20230101'这一天，以及它之前和之后5天的数据
industries = list(set([Resource.ts_code_to_stock_info[k].industry for k in Resource.ts_code_to_stock_info]))
industries.sort(key = lambda x : x)
# industries = ['白酒']
for industry in industries:
    num, fund_flows, prev_price, price = get_fund_flow(industry, '20231227', 3) 
    print(f"Industry: {industry},  股票数量: {num} 成交量变化: ¥{fund_flows/100000:.1f}亿  价格变化{prev_price:.2f} -> {price:.2f}")
    # break