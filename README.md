# 设置环境变量
```bash
export STOCKHOOM=/Users/bytedance/DeepStock    # 根目录
export DB_FILE=$STOCKHOOM/run/db.stock         # 数据库目录
export PYTHONPATH=$PYTHONPATH:$STOCKHOOM/src   # python import查找路径
export PYTHONDONTWRITEBYTECODE=1               # 不生成pyc

更新数据库
python3 src/data/update_database.py
```

echo "数据库路径: $DB_FILE"
echo "根目录为$STOCKHOOM"

### 需求
1. 底部涨停的
###