
import logging
from util import date_add, date_diff, get_date
from data.sqlite import sql_api 
import numpy as np
from data.sqlite.define import TABLE
from data.ts import ts_api
from tqdm import tqdm
from datetime import datetime, timedelta 

def update_stock_basic():
    """ 全量更新stock_basic 表(即全量更新一次)
    """
    df = ts_api.read_stock_basic()
    sql_api.write_table_with_dataframe(TABLE.BASIC, df, if_exists = 'replace', add_update_time = True)
    logging.info("[update_stock_basic] success, 数据量: %s" %(df.shape[0]))
    return 

def update_stock_company():
    """ 全量更新TABLE.COMPANY
    """
    df = ts_api.read_stock_company()
    sql_api.write_table_with_dataframe(TABLE.COMPANY, df, if_exists = 'replace', add_update_time = True)
    logging.info("[update_stock_company] success, 数据量: %s" %(df.shape[0]))
    return 

def update_daily_basic():
    """ 全量更新股票最新的basic信息: 市值、市净率
    """
    df = ts_api.read_daily_basic() 
    sql_api.write_table_with_dataframe(TABLE.DAILY_BASIC, df, if_exists = 'replace', add_update_time = True)
    logging.info("[update_stock_daily_basic] success, 数据量: %s" %(df.shape[0]))

    return

def update_stock_daily():
    """ append更新每个股票的最新日线数据
    """
    today = get_date()
    # 获取每个股票最新交易日期 (注意: 对于新上市的股票, 其latest_trade_date为None)
    query_latest_date = f"""
        SELECT 
            {TABLE.BASIC}.ts_code, 
            MAX({TABLE.DAILY}.trade_date) AS latest_trade_date
        FROM {TABLE.BASIC}
        LEFT JOIN {TABLE.DAILY} ON {TABLE.DAILY}.ts_code = {TABLE.BASIC}.ts_code
        GROUP BY {TABLE.BASIC}.ts_code
        having latest_trade_date != '{today}' or latest_trade_date is null
    """
    stock_date = sql_api.simple_execute(query_latest_date, to_dict = False)
    # latest_trade_date=None 可能是新上市股票，所以默认更新一次
    new_stocks = [row[0] for row in stock_date if row[1] is None]  # 这些股票没有日线信息，需要新增
    estimated_update_count = np.sum([date_diff(row[1], today) for row in stock_date if row[1] is not None])
    # 更新每个股票到最新日线
    logging.info("[update_stock_daily] 需要更新股票数: %s 预计需要更新: (1)%s个样本, (2)新股票%s个(%s)" %(len(stock_date), estimated_update_count, len(new_stocks), ",".join(new_stocks[:15])))
    logging.info("[update_stock_daily] 更新前, TABLE.DAILY数量: %s" %(sql_api.get_table_count(TABLE.DAILY)))
    progress = tqdm(total = len(stock_date))
    for ts_code, latest_trade_date in stock_date:
        progress.update(1)
        progress.set_description(f"Processing {ts_code} ({latest_trade_date} ~ 最新)")
        if latest_trade_date is not None:
            latest_trade_date = date_add(latest_trade_date)
        stock_df = ts_api.read_stock_daily(ts_code, latest_trade_date)
        if stock_df.shape[0] == 0:
            # 没有最新数据
            continue
        sql_api.write_table_with_dataframe(TABLE.DAILY, stock_df, if_exists = 'append', add_update_time= True)
    logging.info("[update_stock_daily] 更新后总行数为: %s" %(sql_api.get_table_count(TABLE.DAILY)))
    return 

if __name__ == "__main__": 
    update_stock_basic()
    update_stock_company()
    update_stock_daily()
    update_daily_basic()