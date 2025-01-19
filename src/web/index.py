# 启动网站
from util import *
from data.sqlite.sql_api import * 
import bottle
from web.stock_group import *
# 在127.0.0.1:80上监听，可自行修改开端口
@bottle.route("/<path>")
def files(path): 
    path = './web/'+path
    logging.info("请求访问文件: %s" %(path))
    if os.path.exists(path):
        content = open(path, "rb")   
        return content.read()
    else:
        # print "not exist :%s" %(path) 
        logging.info("文件 %s 不存在" %(path))
        return "文件不存在, 可能是要执行 python web/index.py"

@bottle.route("/api/go", method="POST")
def go():
    data = bottle.request.json
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    merge_days = int(data.get('merge_days'))
    print(merge_days)
    group_keys = [x.strip() for x in data.get('group_keys').split(",")]
    logging.info("params: group_keys: %s" %(group_keys))
    
    ################ 获取股票日线区间数据 #############################
    df, df2 = reduce_and_sort_by_date(start_date=start_date, end_date=end_date, merge_days = merge_days) 
    # 这里可以添加你的业务逻辑
    # 例如：处理日期范围和分组键
    
    df2 = add_all_cat(df2)
    # assert group_keys in ['is_ST', 'industry', 'total_mv_cat', 'pe_cat', 'pb_cat', 'turnover_rate_f']

    df3 = df2.groupby(['start_date', 'end_date']+group_keys).agg(
        stock_num = pd.NamedAgg(column='days', aggfunc='count'),
        pct_chg = pd.NamedAgg(column='pct_chg', aggfunc='mean'),
        亏损=pd.NamedAgg(column='pe', aggfunc=lambda x: x.isna().sum()),
        total_mv_yi = pd.NamedAgg(column='total_mv', aggfunc=lambda x : x.sum()/10**5),
        pe_median = pd.NamedAgg(column='pe', aggfunc='median'),
        pb_median = pd.NamedAgg(column='pb', aggfunc='median'),
        turnover_rate_f = pd.NamedAgg(column='turnover_rate_f', aggfunc='mean'),
        turnover_rate = pd.NamedAgg(column='turnover_rate', aggfunc='mean'),
        amount_yi = pd.NamedAgg(column='amount_yi', aggfunc='sum'),
        days = pd.NamedAgg(column='days', aggfunc='mean'),
        names = pd.NamedAgg(column='name', aggfunc='first')
    ).reset_index()
    df3['amount_per_stock'] = df3['amount_yi'] / df3['stock_num']
    df3 = df3.sort_values(by='pct_chg')

    # 返回一个示例响应
    data = json.loads(df3.to_json(orient="records"))
    data.sort(key = lambda x : x['start_date'])
    response = {
        "df3" : data,
        "df_html" : df[df['name']=='ST易购'].head(100).to_html(classes='table table-striped', index=False),
        "df_html2" :  df2.head(4).to_html(classes='table table-striped', index=False),
        "df_html3" : df3.head(100).to_html(classes='table table-striped', index=False),
        "status": "success",
        "start_date": start_date,
        "end_date": end_date,
        "merge_days": merge_days,
    }
    return response

@bottle.route("/api/update_superset_temp_table",  method="POST")
def update_superset_temp_table():
    """修改superset的临时表
    """
    data = bottle.request.json
    logging.info("update_superset_temp_table: %s" %(data))
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    sample = int(data.get("sample"))
    from data import update_database
    update_nums = update_database.update_superset_temp_table(start_date, end_date, sample)
    return f"success: {start_date} - {end_date}, 更新数量{update_nums}"
@bottle.route("/")
def index():

    return bottle.redirect("index.html")

@bottle.route("/index")
def index2():
    conf = {}
    summary = {
        "stock_num" : simple_execute("select count(*) from stock_basic_table", to_dict=False)[0][0],
        "conf" : conf
    }

    ################ 获取股票日线区间数据 #############################
    df, df2 = reduce_and_sort_by_date(start_date='20150605', end_date='20150611', merge_days = 5) 

    df2 = add_all_cat(df2)
    group_keys = ['pe_cat']
    # assert group_keys in ['is_ST', 'industry', 'total_mv_cat', 'pe_cat', 'pb_cat', 'turnover_rate_f']
    df3 = df2.groupby(['start_date', 'end_date']+group_keys).agg(
        stock_num = pd.NamedAgg(column='days', aggfunc='count'),
        pct_chg = pd.NamedAgg(column='pct_chg', aggfunc='mean'),
        亏损=pd.NamedAgg(column='pe', aggfunc=lambda x: x.isna().sum()),
        total_mv_yi = pd.NamedAgg(column='total_mv', aggfunc=lambda x : x.sum()/10**5),
        pe_median = pd.NamedAgg(column='pe', aggfunc='median'),
        pb_median = pd.NamedAgg(column='pb', aggfunc='median'),
        turnover_rate_f = pd.NamedAgg(column='turnover_rate_f', aggfunc='mean'),
        turnover_rate = pd.NamedAgg(column='turnover_rate', aggfunc='mean'),
        amount_yi = pd.NamedAgg(column='amount_yi', aggfunc='sum'),
        days = pd.NamedAgg(column='days', aggfunc='mean'),
        names = pd.NamedAgg(column='name', aggfunc='first')
    ).reset_index()
    df3['amount_per_stock'] = df3['amount_yi'] / df3['stock_num']
    df3 = df3.sort_values(by='pct_chg')

    data = {
        "summary" : summary,
        "title" : "股票分析",
        "df_html" : df[df['name']=='ST易购'].head(100).to_html(classes='table table-striped', index=False),
        "df_html2" : df2.head(4).to_html(classes='table table-striped', index=False),
        "df_html3" : df3.head(100).to_html(classes='table table-striped', index=False)   
    }

    return bottle.template("./web/index.html", **data)
if __name__ == "__main__":
    bottle.run(host='localhost', port=8082)