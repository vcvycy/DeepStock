# 需要加到bashrc中

export STOCK_HOME=/Users/jianfeng/Documents/DeepStock
# 数据库目录
export DB_FILE=$STOCK_HOME/run/stock.db         
# python import查找路径
export PYTHONPATH=$PYTHONPATH:$STOCK_HOME/src   
# 不生成pyc
export PYTHONDONTWRITEBYTECODE=1   
# 日志
echo "环境变量设置成功"
echo "数据库路径: $DB_FILE"
echo "根目录为$STOCK_HOME"          
export HTTP_PROXY=
export HTTPS_PROXY=
# 和数据库相关
alias stock_init_sql="$STOCK_HOME/run/update_sql.sh"
# alias stock_update_data="cd $STOCK_HOME; python -m src.data.update_database"

stock_update_data() {
    cd $STOCK_HOME
    # 删除旧的备份文件
    rm -f run/stock.db.backup.*

    # 获取当前日期，格式为 yyyymmdd
    current_date=$(date +%Y%m%d)

    # 创建新的备份文件
    cp run/stock.db run/stock.db.backup.${current_date}

    echo "备份成功: run/stock.db.backup.${current_date}"

    # 更新数据库
    python -m src.data.update_database 
}
# 和训练相关
alias stock_train="cd $STOCK_HOME/src;python -m train.train_v2"
alias stock_update_train_avg_label="cd $STOCK_HOME/src; python -m train.update_avg_label_table"  # fid和date的平均label
alias stock_show_fid_avg_label="cd $STOCK_HOME/src; python -m train.update_avg_label_table --show"
alias stock_train_analyse="cd $STOCK_HOME/src;python -m train.analyse"   # 

alias stock_tensorboard="cd $STOCK_HOME/src; tensorboard --logdir=runs"
alias stock_sqlite3="sqlite3 $DB_FILE"
cat <<EOF
 可使用的命令:
    训练相关: 
    (1) stock_update_train_avg_label 依赖于训练数据, 更新date2avg_label/fid2avg_label
    (2) stock_train: 训练模型
    (3) stock_train_analyse : 分析fid的label均值
    (4) stock_show_fid_avg_label : 展示fid的平均label

    数据库相关: 
    (1) stock_init_sql: 初始化sql表 (如果表存在则不会操作)
    (2) stock_update_data: 更新数据表 (从tushare获取数据)
    (3) stock_sqlite3: 打开数据库文件
EOF