
import sys 
sys.path.append("/Users/chenjianfeng/code/DeepStock/src/data/sqlite")
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from orm import StockBasicTable, StockCompanyTable, StockDailyBasicTable, StockDailyTable, StockRewardTable
import os
# 创建数据库引擎，这里使用 SQLite
engine = create_engine('sqlite:///%s' %(os.getenv("DB_FILE")))  
# 创建会话
session = sessionmaker(bind=engine)()

# 查询数据
stock_basic = session.query(StockDailyTable).filter_by(ts_code='000001.SZ').all()
for item in stock_basic:
    print(item.trade_date)
# print(stock_basic)  # 输出：Test Name

# 关闭会话
session.close()