
#!/bin/bash
echo "[*] 准备更新数据库数据到最新..."
# 获取当前脚本的目录路径
script_dir=$(dirname "$0")
# 切换到脚本所在目录
cd "$script_dir"

############ 初始化数据库 ###################
sqlite3 $DB_FILE < init.sql
if [ -f "$DB_FILE" ]; then
    echo "[*] 执行init.sql成功"
else 
    echo "[!] 数据库不存在"
    exit -1
fi


############ 检查环境变量是否存在 ###################
if [ -z "$STOCK_HOME" ]; then
    echo "环境变量 STOCK_HOME 不存在。请先设置环境变量。"
    exit 1
fi


############ 更新SQLAlchemy ###################
# mkdir -p $STOCK_HOME/src/data/sqlite/
# sql_define=$STOCK_HOME/src/data/sqlite/orm.py
# rm $sql_define
# echo "[*] 准备生成sqlalchemy格式的数据, 写到$DB_FILE -> $sql_define"
# sqlacodegen sqlite:///$DB_FILE > $sql_define
# rm db.stock
# echo "[*] 当前路径: $(pwd)"
# # 检查SQLite3数据库文件 "db.stock" 是否存在。如果不存在，则创建数据库并执行 "init.sql" 文件。  
# if [ ! -f "db.stock" ]; then
#     echo "[*] 数据库文件不存在，创建数据库并执行初始化脚本..."
#     # 创建数据库文件
#     touch db.stock
#     # 创建表和执行初始化脚本
#     sqlite3 db.stock < init.sql
#     echo "  [*] 数据库创建完成并初始化成功！"
# else
#     echo "[*] 数据库文件已存在，无需创建。"
# fi