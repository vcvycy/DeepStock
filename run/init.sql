-- 5000+个股票的基本信息
CREATE TABLE IF NOT EXISTS stock_basic_table (
    ts_code TEXT,
    symbol TEXT,
    name TEXT,
    area TEXT,
    industry TEXT,          -- 行业: 如银行
    market TEXT,
    list_date TEXT,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code) 
);
-- 股票市值等信息
CREATE TABLE IF NOT EXISTS stock_daily_basic_table (
    ts_code TEXT, -- TS股票代码 (主键)
    trade_date TEXT, -- 交易日期
    close REAL, -- 当日收盘价
    turnover_rate REAL, -- 换手率（%）
    turnover_rate_f REAL, -- 换手率（自由流通股）
    volume_ratio REAL, -- 量比
    pe REAL, -- 市盈率（总市值/净利润，亏损的PE为空）
    pe_ttm REAL, -- 市盈率（TTM，亏损的PE为空）
    pb REAL, -- 市净率（总市值/净资产）
    ps REAL, -- 市销率
    ps_ttm REAL, -- 市销率（TTM）
    dv_ratio REAL, -- 股息率（%）
    dv_ttm REAL, -- 股息率（TTM）（%）
    total_share REAL, -- 总股本（万股）
    float_share REAL, -- 流通股本（万股）
    free_share REAL, -- 自由流通股本（万）
    total_mv REAL, -- 总市值（万元）
    circ_mv REAL, -- 流通市值（万元）
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code) 
);
-- 股票的公司信息
CREATE TABLE IF NOT EXISTS  stock_company_table (
    ts_code TEXT,
    chairman TEXT,    -- 法人代表
    manager TEXT,     -- 总经理
    secretary TEXT,   -- 董事长秘书
    reg_capital REAL, -- 注册资本
    setup_date TEXT,  -- 创建时间
    province TEXT,    -- 省份
    city TEXT,        -- 城市
    employees INTEGER, -- 员工数量
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code) 
);
-- 股票薪酬表
CREATE TABLE IF NOT EXISTS stock_reward_table (
    ts_code TEXT, -- TS股票代码
    ann_date TEXT, -- 公告日期
    end_date TEXT, -- 截止日期
    name TEXT, -- 姓名
    title TEXT, -- 职务
    reward REAL, -- 报酬
    hold_vol REAL, -- 持股数
    PRIMARY KEY (ts_code, end_date, name) 
);

-- 股票日线信息
CREATE TABLE IF NOT EXISTS stock_daily_table (
    ts_code TEXT COMMENT '股票代码',
    trade_date TEXT COMMENT '交易日期',
    open REAL COMMENT '开盘价',
    high REAL COMMENT '最高价',
    low REAL COMMENT '最低价',
    close REAL COMMENT '收盘价',
    pre_close REAL COMMENT '昨收价(前复权)',
    change REAL COMMENT '涨跌额',
    pct_chg REAL COMMENT '涨跌幅（未复权，如果是复权请用通用行情接口）',
    vol REAL COMMENT '成交量（手）',
    amount REAL COMMENT '成交额（千元）',
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, trade_date)  -- 主键
);
-- 模型训练
CREATE TABLE  IF NOT EXISTS date_avg_label_table (
    date TEXT,
    key TEXT,       -- label的key
    count INTEGER,
    avg_label FLOAT,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, key)
);
CREATE TABLE IF NOT EXISTS fid_avg_label_table (
    fid INTEGER,
    key TEXT,       -- label的key
    count INTEGER,
    avg_label REAL,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (fid, key)
);