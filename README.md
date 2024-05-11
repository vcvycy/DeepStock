# 配置
## 设置环境变量
```bash
# 根目录
export STOCKHOOM=/Users/chenjianfeng/code/DeepStock
# 数据库目录
export DB_FILE=$STOCKHOOM/run/db.stock         
# python import查找路径
export PYTHONPATH=$PYTHONPATH:$STOCKHOOM/src   
# 不生成pyc
export PYTHONDONTWRITEBYTECODE=1   
# 日志
echo "环境变量设置成功"
echo "数据库路径: $DB_FILE"
echo "根目录为$STOCKHOOM"          

```
## 初始化/更新数据库
```bash
$STOCKHOOM/run/update_sql.sh
```
## 更新数据库(需要每日更新)
```bash
python3 src/data/update_database.py
```
# DEBUG
查看sqlite3数据库:  SQLiteStudio

# 项目架构
* src/data: 从tushare读取数据，写到sqlite3数据库。
### 需求
1. 底部涨停的
###