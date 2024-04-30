from data.sqlite import sql_api 

class Candle:
    def __init__(self):
        pass
class Stock:
    def __init__(self, symbol):
        # symbol为股市代码, 不需要加.SZ后缀
        symbol = symbol.spit(".")[0] 
        sql_api.simple_execute()
        return 


if __name__ == "__main__":
    a = Stock 