# encoding: UTF-8

# 默认空值
EMPTY_STRING = ''
EMPTY_UNICODE = u''
EMPTY_INT = 0
EMPTY_FLOAT = 0.0

# 方向常量
DIRECTION_NONE = u'无方向'
DIRECTION_LONG = u'多'
DIRECTION_SHORT = u'空'
DIRECTION_UNKNOWN = u'未知'
DIRECTION_NET = u'净'
DIRECTION_SELL = u'卖出'      # IB接口

# 开平常量
OFFSET_NONE = u'无开平'
OFFSET_OPEN = u'开仓'
OFFSET_CLOSE = u'平仓'
OFFSET_CLOSETODAY = u'平今'
OFFSET_CLOSEYESTERDAY = u'平昨'
OFFSET_UNKNOWN = u'未知'

# 数据库名称
TICK_DB_NAME = 'Rickets_Tick_Db'
DAILY_DB_NAME = 'Rickets_Daily_Db'
MINUTE_DB_NAME = 'Rickets_1Min_Db'
RANGEBAR_DB_NAME = 'Rickets_RangeBar_Db'
RAW_DB_NAME = 'Rickets_Raw_Db'