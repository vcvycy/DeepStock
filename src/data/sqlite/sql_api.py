### 所有操作sql的语句都加在这里 ###
import sqlite3
import pandas as pd
import os
from datetime import datetime
from data.sqlite.define import TABLE
import logging
DATABASE_NAME = os.environ.get('DB_FILE')
assert DATABASE_NAME is not None, "数据库名: DB_FILE未配置(环境变量)"
logging.info("[sql_api.py] 数据库 %s" %(DATABASE_NAME))
def get_conn():
    # 获取sqlite3连接对象
    global DATABASE_NAME
    return sqlite3.connect(DATABASE_NAME)

def simple_execute(query, params=None, to_dict=True, as_df = False):
    """ 执行query
    """
    def dict_factory(cursor, row):
        return { col[0] : row[idx] for idx, col in enumerate(cursor.description)}
    conn = get_conn()
    cursor = conn.cursor() 
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    if to_dict:
        cursor.row_factory = dict_factory
    data =  cursor.fetchall()
    if as_df:
        data = pd.DataFrame(data)
    return data

def is_table_exists(table_name):
    """ 判断表是否存在
    """
    query = f"select count(*) from sqlite_master where type='table' and name='{table_name}'"
    # print(query)
    count= simple_execute(query, to_dict = False)[0][0]
    # print("%s %s" %(table_name, count))
    return count > 0
def clear_table(table_name):
    """ 清空数据表
    """
    if not is_table_exists(table_name):
        return  
    query = f"delete from {table_name}"
    cursor = get_conn().cursor() 
    # print(query)
    cursor.execute(query)
    return 

def get_table_count(table_name):
    query = f"select count(*) from {table_name}"
    return simple_execute(query, to_dict = False)[0][0]

def get_table_columns(table_name, ignore_update_time = True):
    """获取表的所有列名
    """
    conn = get_conn()
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    table_columns = [(c[1], c[2]) for c in cursor.fetchall() if ignore_update_time and c[1] != 'update_time']
    return table_columns

def write_table_with_dataframe(table_name, dataframe, if_exists, add_update_time = True):
    """
      dataframe写到数据表中
      if_exists: 
        - append 添加到后面
        - replace 整个表重写
    """
    assert dataframe.shape[0] < 4001000, 'df太大了!!! %s' %(dataframe.shape[0])
    # logging.info("[sql_api.write_table_with_dataframe] %s 写入(%s)数据: %s" %(table_name, if_exists, dataframe.shape))
    if add_update_time:
        dataframe['update_time'] = dataframe.apply(lambda x:datetime.now(), axis=1)
    
    # 分批次写入数据库
    batch_size = 1000000
    total_rows = dataframe.shape[0]
    num_batches = (total_rows + batch_size - 1) // batch_size  # 向上取整
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_rows)
        batch_df = dataframe.iloc[start_idx:end_idx]
        
        # 写入数据库
        batch_df.to_sql(table_name, get_conn(), if_exists='append' if i > 0 else if_exists, index=False)
    return 

def read_single_stock(ts_code, as_df = True):
    query = """
        select * from 
        stock_basic_table a 
            join stock_daily_table b on a.ts_code  = b.ts_code
            join stock_daily_basic_table c on a.ts_code = c.ts_code 
        where  a.ts_code = '002140.SZ' 
        order by b.trade_date;
    """
    data = simple_execute(query, to_dict = True)
    if as_df:
        data = pd.DataFrame(data)
    return data

def reduce_and_sort_by_date(start_date = '20240923', end_date = '20240926', merge_days = 1):
    """ merge_days: n个交易日合并成一个, k线图
    """
    query = f"""
       with t1 as (
            select 
                *
            from 
                stock_basic_table a 
                    join stock_daily_table b on a.ts_code  = b.ts_code
                    join stock_daily_basic_table c on a.ts_code = c.ts_code  and b.trade_date = c.trade_date 
            where
                b.trade_date >= '{start_date}' and b.trade_date <= '{end_date}'
                -- and industry = '橡胶'
                -- and a.ts_code in ('000001.SZ', '000002.SZ', '000402.SZ') 
        )
        select  
            * 
        from t1 
        limit 
            1000000
    """
    df = simple_execute(query, to_dict = True, as_df=True)
    trade_date_list = df['trade_date'].unique()
    trade_date_list.sort()

    assert len(trade_date_list) % merge_days == 0, "merge_days 必须是交易日的总天数的整数倍， %s vs %s" %(len(trade_date_list), merge_days)
    
    ############## 日期区间映射  ，新增两个字段, start_date + end_date, 即交易日对应到哪个[start_date, end_date]区间
    start_date_map = {}
    end_date_map = {}
    for i, date in enumerate(trade_date_list):
        start_date_map[date] = trade_date_list[i//merge_days * merge_days]
        end_date_map[date] = trade_date_list[i//merge_days * merge_days + merge_days - 1]
    df['start_date'] = df['trade_date'].map(start_date_map)
    df['end_date'] = df['trade_date'].map(end_date_map)
    df['is_ST'] = df['name'].str.contains('ST')

    ###### k线图合并 ########
    ## 新增3个字段: (1) 给df新增: (1) 区间开盘价 start_date_open (2) 区间前的收盘价 start_date_pre_close (3) 区间收盘价 end_date_close
    df_start_date = df[df['trade_date'] == df['start_date']].rename(columns={
        'ts_code': 'ts_code',
        'start_date' : 'start_date',
        'open': 'start_date_open',            # 时间区间首日的开盘价
        'pre_close': 'start_date_pre_close'   # 时间区间首日的前一天收盘价
    })[['ts_code','start_date',  'start_date_open', 'start_date_pre_close']]
    
    df_end_date = df[df['trade_date'] == df['end_date']].rename(columns={
        'ts_code': 'ts_code',
        'end_date' : 'end_date',
        'close': 'end_date_close'             # 时间区间的最后一天的收盘价
    })[['ts_code', 'end_date', 'end_date_close']]

    df = df.merge(df_start_date, on=['ts_code', 'start_date'], how='left')
    df = df.merge(df_end_date, on=['ts_code', 'end_date'], how='left')
    
    ## 多天数据融合, 并合并k线数据 ####
    df2 = df.groupby(['ts_code', 'start_date', 'end_date']).agg(
        # 市盈率、市净率、总市值
        pe = pd.NamedAgg(column='pe', aggfunc='mean'),
        pe_ttm = pd.NamedAgg(column='pe_ttm', aggfunc='mean'),
        pb = pd.NamedAgg(column='pb', aggfunc='mean'),
        turnover_rate_f = pd.NamedAgg(column='turnover_rate_f', aggfunc='sum'),
        turnover_rate = pd.NamedAgg(column='turnover_rate', aggfunc='sum'),
        total_mv = pd.NamedAgg(column='total_mv', aggfunc='mean'),
        # k线图
        pre_close = pd.NamedAgg(column='start_date_pre_close', aggfunc='first'),
        open=pd.NamedAgg(column='start_date_open', aggfunc='first'),
        close=pd.NamedAgg(column='end_date_close', aggfunc='first'),
        high =pd.NamedAgg(column='high', aggfunc='max'),
        low = pd.NamedAgg(column='low', aggfunc='min'),
        name=pd.NamedAgg(column='name', aggfunc='first'),
        industry=pd.NamedAgg(column='industry', aggfunc='first'),
        vol=pd.NamedAgg(column='vol', aggfunc='sum'),
        amount=pd.NamedAgg(column='amount', aggfunc='sum'),
        amount_yi = pd.NamedAgg(column='amount', aggfunc=lambda x : x.sum() / 10**5),  # 亿为单位(原来是1000)
        days = pd.NamedAgg(column='trade_date', aggfunc='count'),
    ).reset_index()
    # print(df2)
    df2['pct_chg'] = (df2['close'] - df2['pre_close']) / df2['pre_close']
    return df, df2

# df, df2 = reduce_and_sort_by_date(merge_days = 2)
# print(df)
# print(df2)
# exit(0)
# def get_stock_rise_between_dates(st_date, ed_date):
#     query = f""" 
#         SELECT 
#             ts_code,
#             count(*) as count,
#             MAX(CASE WHEN trade_date = '{st_date}' THEN open END) as open,
#             MAX(CASE WHEN trade_date = '{ed_date}' THEN close END) as close
#         FROM 
#             stock_daily_table
#         WHERE 
#             trade_date IN ('{st_date}', '{ed_date}')
#         GROUP BY 
#             ts_code
#         having
#             count = 2
#     """
#     # print(query)
#     items = simple_execute(query)

#     tscode2score = {item['ts_code'] : (item['open'], item['close']) for item in items if item['count'] == 2}
#     return tscode2score

def get_stock_daily_info(ts_code, st_date, ed_date = None):
    query = f"""
        SELECT 
            *
        FROM 
            stock_daily_table
        WHERE 
            ts_code = '{ts_code}'
            and trade_date >= '{st_date}' 
            and trade_date <= '{ed_date if ed_date is not None else st_date}'
        order by 
            trade_date
    """
    # print(query)
    data = simple_execute(query)
    # print(data)
    # print(len(data))
    return data

def update_date_avg_label_table(date2label_count):
    """ 更新avg_label_table表
    """
    # 将date2label_count转换为DataFrame
    data = []
    for date_and_key, label_and_count in date2label_count.items():
        date, key = date_and_key
        avg_label, count = label_and_count
        data.append({'date': date, 'key': key, 'avg_label': avg_label, 'count': count})
    df = pd.DataFrame(data)
    logging.info(f"每日平均label样例(总数:{df.shape[0]}):")
    print(df.head(3))
    try:
        write_table_with_dataframe("date_avg_label_table", df, if_exists='replace', add_update_time=True)
        logging.info("[update_date_avg_label_table] success, 写入sql行数: %s" % (df.shape[0]))
    except Exception as e:
        logging.error("[update_date_avg_label_table] failed: %s" % e)
    return

def update_fid_avg_label(fid2label_count, fid2feature):
    """ 更新fid_avg_label_table表
    """
    # 将fid2label_count转换为DataFrame
    data = []
    for fid_and_key, label_and_count in fid2label_count.items():
        fid, key = fid_and_key 
        avg_label,count = label_and_count
        feature = fid2feature[fid]
        item = {
            'slot': fid>>54,
            'fid': fid, 
            'key': key, 
            'avg_label': avg_label, 
            'count': count,
            'raw_feature' : ", ".join(feature.raw_feature),
            'extracted_features' : ", ".join(feature.extracted_features),
        }
        data.append(item)
    df = pd.DataFrame(data)
    try:
        write_table_with_dataframe("fid_avg_label_table", df, if_exists='replace', add_update_time=True)
        logging.info("[update_fid_avg_label] success, 数据量: %s" % (df.shape[0]))
    except Exception as e:
        logging.error("[update_fid_avg_label] failed: %s" % e)
    return
def read_date_avg_label(key=None):
    """ 指定key返回 date -> avg_label """
    if key is None:
        # 获取所有可用的 key
        query = "SELECT DISTINCT key FROM date_avg_label_table"
        keys = simple_execute(query, to_dict=False)
        logging.error("read_date_avg_label: 没有指定key, 可用: %s" %(keys))
        exit(0)
    else:
        # 根据指定的 key 获取 date -> avg_label
        query = "SELECT date, avg_label FROM date_avg_label_table WHERE key = ?"
        result = simple_execute(query, to_dict=True, params=(key,))
        date_avg_label = {row['date']: row['avg_label'] for row in result}
        return date_avg_label

def read_fid_avg_label(key=None, raw = False):
    """ 指定key返回 fid -> avg_label """
    if key is None:
        # 获取所有可用的 key
        query = "SELECT DISTINCT key FROM fid_avg_label_table"
        keys = simple_execute(query, to_dict=False)
        logging.error("read_date_avg_label: 没有指定key, 可用: %s" %(keys))
        exit(0)
    else:
        # 根据指定的 key 获取 fid -> avg_label
        query = "SELECT * FROM fid_avg_label_table WHERE key='train_label' or key = ?"
        result = simple_execute(query, to_dict=True, params=(key,))
        if raw:
            return result
        fid_avg_label = {row['fid']: row['avg_label'] for row in result}
        return fid_avg_label
    
def create_temp_table():
    """创建临时表，每次更新数据库，都全量更新
    """
    query = """
        CREATE TEMP_STOCK_CATEGORY TABLE temp_table AS
        SELECT 
            ts_code,
            trade_date,
            open,
            high,
            low,
            close,
            pre_close,
            change,
            pct_chg,
            vol,
            amount
        FROM stock_daily_table
    """
    return 

if __name__ == "__main__":
    # print(get_table_columns(TABLE.BASIC))
    # data = read_single_stock('002140.SZ')
    # print(data.count())
    # print(data.head())
    # print(reduce_and_sort_by_date("d"))
    pass
    # print(get_stock_rise_between_dates('20221012', '20221027'))
    # print(get_stock_daily_info('002140.SZ', '20221012'))