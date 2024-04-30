from global_resource import Resource
from datetime import datetime, timedelta
import numpy as np

if __name__ == "__main__":
    stock = Resource.ts_code_to_stock_info['000001.SZ']
    print(stock.daily('20231120', '20231220'))