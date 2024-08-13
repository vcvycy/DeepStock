### 所有操作sql的语句都加在这里 ###
import sqlite3
import pandas as pd
import os
from datetime import datetime
from data.sqlite.define import TABLE
import logging
DATABASE_NAME = os.environ.get('DB_FILE')
assert DATABASE_NAME is not None, "数据库名: DB_FILE未配置(环境变量)"
print("DB: %s" %(DATABASE_NAME))
def get_conn():
    # 获取sqlite3连接对象
    global DATABASE_NAME
    return sqlite3.connect(DATABASE_NAME)

def simple_execute(query, params=None, to_dict=True):
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
    return cursor.fetchall()

def clear_table(table_name):
    """ 清空数据表
    """
    query = f"delete from {table_name}"
    cursor = get_conn().cursor() 
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
    # clear_table(table_name)
    if add_update_time:
        dataframe['update_time'] = dataframe.apply(lambda x:datetime.now(), axis=1)
    return dataframe.to_sql(table_name, get_conn(), if_exists=if_exists, index=False)

def read_single_stock(ts_code):
    query = """
        select * from 
        stock_basic_table a 
            join stock_daily_table b on a.ts_code 
            join stock_daily_basic_table c on a.ts_code = c.ts_code 
        where  a.ts_code = '002140.SZ' 
        order by b.trade_date;
    """
    print(simple_execute("select * from stock_daily_table limit 2", to_dict = True))
    return 

def get_stock_rise_between_dates(st_date, ed_date):
    query = f""" 
        SELECT 
            ts_code,
            count(*) as count,
            MAX(CASE WHEN trade_date = '{st_date}' THEN open END) as open,
            MAX(CASE WHEN trade_date = '{ed_date}' THEN close END) as close
        FROM 
            stock_daily_table
        WHERE 
            trade_date IN ('{st_date}', '{ed_date}')
        GROUP BY 
            ts_code
        having
            count = 2
    """
    # print(query)
    items = simple_execute(query)

    tscode2score = {item['ts_code'] : (item['open'], item['close']) for item in items if item['count'] == 2}
    return tscode2score

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
if __name__ == "__main__":
    # print(get_table_columns(TABLE.BASIC))
    # read_single_stock('002140.SZ')
    # print(get_stock_rise_between_dates('20221012', '20221027'))
    print(get_stock_daily_info('002140.SZ', '20221012'))