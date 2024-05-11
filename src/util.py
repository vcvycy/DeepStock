import logging
from datetime import datetime, timedelta 
# 在其他包之前配置basicConfig
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# class MYDate():
#     def __init__():

# def date_format(date):
#     if isinstance(date, datetime):
#         return date.strftime('%Y%m%d')
#     else:
#         return date

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
    """ 获取当前date: 如返回20231001, 返回字符串
    """
    return date_add(datetime.now().strftime('%Y%m%d'), delta)

def date_diff(date_str1, date_str2):
    """
    计算两个日期字符串之间的天数差。
    example: date_diff("20220101", "20240101")
    """
    # 定义日期格式
    date_format = "%Y%m%d"
    # 将日期字符串解析为日期对象
    date1 = datetime.strptime(date_str1, date_format)
    date2 = datetime.strptime(date_str2, date_format)
    # 计算两个日期之间的天数差
    date_diff = (date2 - date1).days
    return date_diff

if __name__ == "__main__":
    print()
    # print(get_date(-1))