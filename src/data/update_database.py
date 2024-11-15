
import logging
from util import date_add, date_diff, get_date
from data.sqlite import sql_api 
import numpy as np
from data.sqlite.define import TABLE
from data.ts import ts_api
from tqdm import tqdm
from datetime import datetime, timedelta 
"""
数据库更新:
1. 全量更新使用 replace(删除旧表), 增量更新使用append
"""
def update_stock_basic():
    """ 全量更新stock_basic 表(即全量更新一次)
    """
    df = ts_api.read_stock_basic()
    assert df.shape[0] > 0, 'read_stock_basic return empty'
    sql_api.write_table_with_dataframe(TABLE.BASIC, df, if_exists = 'replace', add_update_time = True)
    logging.info("[update_stock_basic] success, 数据量: %s" %(df.shape[0]))
    # TODO: 索引
    sql_api.simple_execute(f"CREATE INDEX index_for_stock_basic on {TABLE.BASIC} (ts_code);")
    return 

def update_stock_company():
    """ 全量更新TABLE.COMPANY
    """
    df = ts_api.read_stock_company()
    assert df.shape[0] > 0, 'read_stock_company return empty'
    sql_api.write_table_with_dataframe(TABLE.COMPANY, df, if_exists = 'replace', add_update_time = True)
    logging.info("[update_stock_company] success, 数据量: %s" %(df.shape[0]))
    # 建索引
    sql_api.simple_execute(f"CREATE INDEX index_for_stock_company on {TABLE.COMPANY} (ts_code);")
    return 

def update_daily_basic():
    """ 增量更新股票的basic信息: 市值、市净率
    """
    trade_date_list = ts_api.get_trade_date()

    sql = "select trade_date from stock_daily_basic_table group by trade_date"
    exist_trade_date = [d[0] for d in sql_api.simple_execute(sql, to_dict = False)]
    logging.info("[update_stock_daily_basic] 当前数据库包含的交易日期数量: %s" %(len(exist_trade_date)))

    need_update_date = list(set(trade_date_list) - set(exist_trade_date))
    need_update_date.sort(reverse=True)
    if len(need_update_date) == 0:
        logging.info("[update_stock_daily_basic] 没有需要更新的交易日期")
        return
    logging.info("[update_stock_daily_basic] 待更新交易日期 %s个, top 10 个: %s" %(len(need_update_date), list(need_update_date)[:10]))
    progress = tqdm(total = len(need_update_date))
    total = 0
    for trade_date in need_update_date:
        progress.update(1)
        progress.set_description(f"Processing {trade_date}, total :{total}")
        df = ts_api.read_daily_basic(trade_date)
        total += df.shape[0]
        sql_api.write_table_with_dataframe(TABLE.DAILY_BASIC, df, if_exists = 'append', add_update_time = True)
    logging.info("[update_stock_daily_basic] success, 天数: %s 数据量: %s" %( len(need_update_date), total))
    return

def update_stock_daily_by_trade_date():
    """增量更新日线: 与update_stock_daily一致, 但是这里是以trade_date为主去跑的
    """
    trade_date_list = ts_api.get_trade_date()
    # 获取当前存在的最大日期
    sql = "select trade_date from stock_daily_table group by trade_date"
    exist_trade_date = [d[0] for d in sql_api.simple_execute(sql, to_dict = False)]
    logging.info("[update_stock_daily_by_trade_date] 当前数据库包含的交易日期数量: %s" %(len(exist_trade_date)))

    need_update_date = list(set(trade_date_list) - set(exist_trade_date))
    need_update_date.sort(reverse=True)

    if len(need_update_date) == 0:
        logging.info("[update_stock_daily_by_trade_date] 没有需要更新的交易日期")
        return
    logging.info("[update_stock_daily_by_trade_date] 待更新交易日期 %s个, top 10 个: %s" %(len(need_update_date), list(need_update_date)[:10]))

    # 逐天更新
    total = 0
    progress = tqdm(total = len(need_update_date))
    for trade_date in need_update_date:
        progress.update(1)
        progress.set_description(f"更新日线数据: {trade_date}, total :{total}")
        df = ts_api.read_stock_daily(start_date = trade_date, end_date= trade_date)
        total += df.shape[0]
        sql_api.write_table_with_dataframe(TABLE.DAILY, df, if_exists = 'append', add_update_time= True)
    logging.info("[update_stock_daily_by_trade_date] success, 天数: %s 数据量: %s" %( len(need_update_date), total))
    return 

def update_stock_daily():
    """ 增量更新日线: append更新每个股票的最新日线数据
    """
    assert False , "不推荐使用update_stock_daily"
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


# def update_ths_index():
#     """同花顺板块
#     """
#     ### 同花顺指数 #####
#     df = ts_api.client.ths_index()
#     sql_api.write_table_with_dataframe("ths_index", df, if_exists = 'replace', add_update_time= False)
#     return 
# update_ths_index()
# exit(0)

def update_index():
    """主要指数: 全量更新
         1. 指数列表
         2. 指数日线数据
         3. 指数成分股
    """
    # 定义表名
    table_index = 'index_basic'          
    table_index_daily = 'index_daily'   
    table_index_daily_basic = 'index_daily_basic'
    table_index_weight = 'index_weight'  
    # 每次更新，都清空已有的表 
    if sql_api.is_table_exists(table_index): sql_api.simple_execute(f"drop table {table_index}")
    if sql_api.is_table_exists(table_index_daily): sql_api.simple_execute(f"drop table {table_index_daily}")
    if sql_api.is_table_exists(table_index_weight): sql_api.simple_execute(f"drop table {table_index_weight}")
    if sql_api.is_table_exists(table_index_daily_basic): sql_api.simple_execute(f"drop table {table_index_daily_basic}")
    # 开始更新
    markets = ['SSE', 'CSI', 'SZSE'] # SSE是上海证券市场
    valid_index =  set(['沪深300', '上证指数', '上证50', '中证500', '中证1000', '中证2000', '创业板指', '科创50', '深证成指', '北证50'])
    bar = tqdm(range(len(valid_index)))
    cur_year = int(max(ts_api.get_trade_date())[:4])
    for m in markets:
        if len(valid_index) == 0: break
        df = ts_api.client.index_basic(market=m)
        sql_api.write_table_with_dataframe(table_index, df, if_exists = 'append', add_update_time= False)
        for i in range(len(df)):
            ts_code = df.iloc[i]['ts_code']
            name = df.iloc[i]['name']
            if name not in valid_index and ts_code not in valid_index:
                continue
            if name in valid_index : valid_index.remove(name)
            if ts_code in valid_index: valid_index.remove(ts_code)
            #### 指数daily baisic
            step = 10  # 10年取一次数据
            for year in range(1990, cur_year, step):
                st = f'{year}0101'
                ed = f'{year+step-1}1231'
                df_basic_dayild = ts_api.client.index_dailybasic(ts_code = ts_code, start_date=st, end_date=ed)
                sql_api.write_table_with_dataframe(table_index_daily_basic, df_basic_dayild, 'append', add_update_time= True)

                #### 指数日线图: 更新指数日线
                df_daily = ts_api.client.index_daily(ts_code=ts_code, start_date=st, end_date=ed)
                sql_api.write_table_with_dataframe(table_index_daily, df_daily, if_exists = 'append', add_update_time= False)

            #### 更新指数成分股, 每年都取
            year = cur_year
            while year >= 1990:
                df_weight = ts_api.client.index_weight(index_code=ts_code, start_date = f'{year}0101', end_date=f'{year}1231')
                if df_weight.shape[0] == 0:
                    break
                df_weight = df_weight[df_weight['trade_date']==df_weight.iloc[0]['trade_date']]  # 只保留最新的数据
                df_weight['index_name'] = name 
                year = int(df_weight.iloc[0]['trade_date'][:4]) -1
                df_weight['weight_pct'] = df_weight['weight'] / df_weight['weight'].sum()
                sql_api.write_table_with_dataframe(table_index_weight, df_weight, if_exists = 'append', add_update_time= False)
                # logging.info(f"取{name} {ts_code}, {year} shape={df_weight.shape} , 写入后大小 {sql_api.get_table_count(table_index_weight)}")
            bar.set_description(f"更新指数数据: {name} {ts_code}") 
            bar.update(1)
    logging.info("[update_index] 未更新的指数: {valid_index}")
    # 建立索引
    sql_api.simple_execute("CREATE INDEX index_for_index_daily_basic on index_daily_basic (ts_code, trade_date);")
    sql_api.simple_execute("CREATE INDEX index_for_index_daily on index_daily (ts_code, trade_date);")
    sql_api.simple_execute("CREATE INDEX index_for_index_weight on index_weight (trade_date);")
    return

def update_superset_temp_table(start_date = '20140601', end_date = '20151001', sample = 1):
    """更新中间表: 
       时间区间内会做一些归一化操作；
       1. 股票日线图: 这些表用于superset 展示
       2. 指数日线图
       :params sample : 1表示不采样，2表示每2天取一个数据，以此类推
    """ 
    dates = [
        (start_date, end_date),
    ]
    update_num = 0
    assert len(dates) == 1, "不同日期写到同一个表里"
    for start_date, end_date in dates:
        # k线日线图
        sql = f"""
            with t_index as (  -- ts_code -> 是否属于某个指数(由于原始tushare数据不完善, 这里只能取最新(2024)的数据, 有数据穿越)
                select 
                    con_code as ts_code,
                    avg(case when index_name = '科创50' then weight_pct else 0 end) as 科创50,
                    avg(case when index_name = '沪深300' then weight_pct else 0 end) as 沪深300,
                    avg(case when index_name = '中证500' then weight_pct else 0 end) as 中证500,
                    avg(case when index_name = '中证1000' then weight_pct else 0 end) as 中证1000,
                    avg(case when index_name = '中证2000' then weight_pct else 0 end) as 中证2000,
                    avg(case when index_name = '创业板指' then weight_pct else 0 end) as 创业板指, 
                    avg(case when index_code = '899050CNY01.CSI' then weight_pct else 0 end) as 北证50
                
                from 
                    index_weight
                group by 
                    con_code
                order 
                    by 中证1000 desc
            ),
            
            t1 as ( -- (ts_code, date) 的所有日线图
                select 
                    a.industry,  -- 行业
                    a.area,      -- 地区
                    a.list_date, -- 上市时间
                    a.act_ent_type, -- 民企 or 国企
                    a.name,
                    a.market,
                    c.*
                    ,c.total_mv * turnover_rate/10 as amount
                    -- 补充k线图
                    ,b.high as high         -- k线图
                    -- b.low as low,         -- k线图
                    -- b.open as open,         -- k线图
                    -- b.pre_close as pre_close,         -- k线图
                    -- b.amount as amount,
                    -- b.vol as vol,
                    ,b.pct_chg as pct_chg
                    
                from 
                    stock_basic_table a 
                        join stock_daily_basic_table  c on a.ts_code = c.ts_code 
                         join stock_daily_table b on a.ts_code  = b.ts_code and b.trade_date = c.trade_date
                where
                    (c.trade_date % {sample} = 0 or c.trade_date > '20240901') and  -- 9月份后的数据不采样
                    c.trade_date >= {start_date} and c.trade_date <= {end_date}
            ),

            t2 as(-- ts_code -> 最早/最晚交易日期
                select 
                    ts_code,
                    min(trade_date) as min_date,
                    max(trade_date) as max_date
                from 
                    t1
                group by ts_code
            ),
            t3 as ( -- ts_code -> 最早/最晚交易日期 + 第一个交易日的k线图
                select 
                    t1.*, 
                    min_date as min_date,
                    max_date as max_date
                from t1 join t2 on t1.ts_code = t2.ts_code and t1.trade_date = t2.min_date 
            )
            ,t4 as ( -- [ts_code, date] -> 日线图 + 首日k线图 + 市值排名 + 均值
                select 
                    t1.*,
                    -- 市值排名
                    dense_rank() over (order by t3.total_mv desc) as mv_rank,
                    dense_rank() over (order by t3.circ_mv desc) as circ_mv_rank,
                    --均值
                    max(t1.high) over (partition by t3.ts_code order by t1.trade_date rows between 5 preceding and 5 following) as high_10d, -- 前后10日最高点
                    avg(t1.close) over (partition by t3.ts_code order by t1.trade_date rows between {30//sample} preceding and current row) as close_avg_30d, -- 30日均值
                    avg(t1.close) over (partition by t3.ts_code order by t1.trade_date rows between {60//sample} preceding and current row) as close_avg_60d, -- 60日均值
                    avg(t1.close) over (partition by t3.ts_code order by t1.trade_date rows between {120//sample} preceding and current row) as close_avg_120d, -- 120日均值

                    -- 统计信息
                    t3.min_date as min_date,
                    t3.max_date as max_date,

                    t3.pe as day1_pe,
                    t3.pb as day1_pb,
                    t3.close as day1_close,
                    t3.total_mv as day1_total_mv,
                    t3.circ_mv as day1_circ_mv,
                    t3.turnover_rate_f as day1_turnover_rate_f,
                    t3.amount as day1_amount,
                    
                    -- 价格归一化
                    t1.close /t3.close as close_norm
                from 
                    t1 join t3 on t1.ts_code = t3.ts_code
                    
                )

            select 
                *,
                t4.*,
                case 
                    when day1_close < 2 then '0-2元股'
                    when day1_close < 5 then '2-5元股'
                    when day1_close < 10 then '5-10元股'
                    when day1_close < 20 then '10-20元股' 
                    else '>20元股'
                end as price_cat,
                case 
                    when name like '中%' then '中字头'
                    else '非中字头'
                end as is_zhong,
                CASE 
                    WHEN name LIKE '%ST%' THEN 'ST'
                    ELSE 'NoST'
                END AS is_ST,
                
                case 
                    WHEN day1_turnover_rate_f < 5 then '换手率0.05-'
                    WHEN day1_turnover_rate_f < 10 then '换手率0.05-0.1'
                    WHEN day1_turnover_rate_f < 20  then '换手率0.1-0.2'
                    WHEN day1_turnover_rate_f < 40  then '换手率0.2-0.4'
                    else '换手率0.4+'
                end as turnover_f_cat,
                
                case 
                    when list_date < 20000101 then '上市2000前'
                    else printf('上市%d', list_date/10000/2*2)
                end as list_date_cat,
                case 
                    WHEN day1_total_mv < 1000000 then '10亿-'
                    WHEN day1_total_mv < 2000000 then '10-20亿'
                    WHEN day1_total_mv < 5000000 then '20-50亿'
                    WHEN day1_total_mv < 10000000 then '50-100亿'
                    else '流通100亿+'
                end as mv_cat,
                case 
                    WHEN day1_circ_mv < 800000 then '流通8亿'
                    WHEN day1_circ_mv < 2000000 then '流通8-20亿'
                    WHEN day1_circ_mv < 5000000 then '流通20-50亿'
                    WHEN day1_circ_mv < 10000000 then '流通50-100亿'
                    else '流通100亿+'
                end as circ_mv_cat,
                case 
                    WHEN day1_pe is null then '亏损'
                    WHEN day1_pe <= 14 then 'PE14-'
                    WHEN day1_pe <= 25 then 'PE14-25'
                    else 'PE25+'
                end as pe_cat,
                
                case 
                    WHEN day1_pb <= 1 then 'PB1-'
                    WHEN day1_pb <= 2 then 'PB1-2'
                    WHEN day1_pb <= 5 then 'PB2-5'
                    else 'PB5+'
                end as pb_cat
                ,case 
                    WHEN day1_amount < 200000 then '成交额2亿-'
                    WHEN day1_amount < 500000 then '成交额2-5亿'
                    WHEN day1_amount < 1000000 then '成交额5-10亿'
                    else '成交额10亿+'
                end as amount_cat
            from 
                t4 left join t_index on t4.ts_code = t_index.ts_code
            where 
                1 
            """
        df = sql_api.simple_execute(sql, as_df = True)
        update_num += df.shape[0]
        table_name = "temp_superset_stock"
        sql_api.write_table_with_dataframe(table_name, df, if_exists = 'replace', add_update_time= False)
        logging.info("[update_superset_temp_table] %s df: %s, 更新后总行数为: %s" %(table_name, df.shape, sql_api.get_table_count(table_name)))
        logging.info("开始更新大盘指数temp表".center(100, "-"))
        # 指数日线图
        sql = f"""
            with t1 as ( -- stock_num
                -- 每个股票最早的日期
                SELECT
                ts_code,
                min(trade_date) as min_trade_date
                from 
                index_daily 
                where 
                trade_date >= {start_date} and trade_date <= {end_date}
                group by 
                ts_code
            ),
            t2 as ( -- stock num 每个指数的最早可交易日期
                select 
                    a.*,
                    b.min_trade_date
                from index_daily a  
                join t1 b on a.ts_code = b.ts_code  and a.trade_date = b.min_trade_date
            )
            select -- stock_num x DATE
                index_basic.name,
                t2.min_trade_date,
                t2.close as day1_close,
                t2.amount as day1_amount,
                index_daily.*,
                index_daily_basic.* ,
                avg(index_daily.amount) over (partition by name order by index_daily.trade_date rows between 4 preceding and current row) as amount_5d, -- 5日平均成交额
                avg(index_daily.pct_chg) over (partition by name order by index_daily.trade_date rows between 29 preceding and current row) as pct_chg_30d, -- n日累积涨幅

                avg(index_daily.close) over (partition by name order by index_daily.trade_date rows between 30 preceding and current row) as close_avg_30d, -- 30日均值
                avg(index_daily.close) over (partition by name order by index_daily.trade_date rows between 60 preceding and current row) as close_avg_60d, -- 60日均值
                avg(index_daily.close) over (partition by name order by index_daily.trade_date rows between 120 preceding and current row) as close_avg_120d -- 120日均值
            from 
                index_basic 
                join t2 on index_basic.ts_code = t2.ts_code
                join index_daily on index_basic.ts_code = index_daily.ts_code
                join index_daily_basic on index_daily.ts_code = index_daily_basic.ts_code and index_daily.trade_date = index_daily_basic.trade_date
            where 
                index_daily.trade_date >= {start_date} and index_daily.trade_date <= {end_date}
            """
        # input(sql)
        df = sql_api.simple_execute(sql, as_df=True)
        update_num += df.shape[0]
        index_table_name = "temp_superset_index_daily"
        sql_api.write_table_with_dataframe(index_table_name, df, if_exists = 'replace', add_update_time= False)
        logging.info("[update_superset_temp_table] %s df: %s, 更新后总行数为: %s" %(index_table_name, df.shape, sql_api.get_table_count(index_table_name)))
    logging.info("更新成功")
    return update_num

if __name__ == "__main__": 
    update_stock_basic()
    update_stock_company()
    update_daily_basic()
    update_stock_daily_by_trade_date()  # 之前的写法 update_stock_daily()
    update_index()  # 更新指数
    
    # update_superset_temp_table(start_date = '20140601', end_date = '20151001')