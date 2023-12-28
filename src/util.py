import logging
from datetime import datetime, timedelta 
# 在其他包之前配置basicConfig
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# class MYDate():
#     def __init__():

def date_format(date):
    if isinstance(date, datetime):
        return date.strftime('%Y%m%d')
    else:
        return date

def date_add(date_string, delta = 1):
    """ 日期字符串+1, 如20220131变成20220201
    """
    date_format = "%Y%m%d"
    # 将日期字符串解析为日期对象
    date = datetime.strptime(date_string, date_format)
    new_date = date + timedelta(days=delta) 
    new_date_string = new_date.strftime(date_format)
    return new_date_string

def get_date(delta = 0):
    """ 获取当前date: 如返回20231001
    """
    return date_add(datetime.now().strftime('%Y%m%d'), delta)

if __name__ == "__main__":
    print(get_date(-1))