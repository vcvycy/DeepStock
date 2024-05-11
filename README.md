# 配置
## 在bashrc中调用我们的环境信息
```bash
echo "source $(pwd)/env.sh" >> ~/.bashrc
```
# DEBUG
查看sqlite3数据库:  SQLiteStudio

# 项目架构
* src/data: 从tushare读取数据，写到sqlite3数据库。
### 需求
1. 底部涨停的
###