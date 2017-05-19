# encoding: UTF-8

"""
QuantStart网站上向量化回测的Demo双均线策略
https://www.quantstart.com/articles/Backtesting-a-Moving-Average-Crossover-in-Python-with-pandas
"""

from backtestBase import *
from backtestEngine import *


########################################################################
class MovingAverageCrossStrategy(Strategy):
    """双均线策略"""

    # ----------------------------------------------------------------------
    def __init__(self, engine, setting):

        # 基础变量
        self.engine = engine

        # 回测参数
        self.bar_period = setting['bar_period']
        self.short_period = setting['short_period']
        self.long_period = setting['long_period']

    # ----------------------------------------------------------------------
    def loadHistoryData(self):
        """载入历史数据"""
        # 返回数据库查询Cursor，查询的参数在策略执行时指定
        dbCursor = self.engine.loadCursor()
        bars = pd.DataFrame(list(dbCursor), columns=['symbol', 'datetime', 'date', 'open', 'close'])
        bars.set_index('datetime', drop=False, inplace=True)  # 使用datetime作为index，同时保留datetime列，原地替换bars

        # 生成15min K线
        bars = bars.resample(self.bar_period).last().dropna()  # 需要去除NA值

        return bars
    # ----------------------------------------------------------------------
    def generate_signals(self):
        """产生回测信号，返回一个包含signals(1, -1 or 0)列的DataFrame"""

        bars = self.loadHistoryData()
        # signal初始化
        signals = pd.DataFrame(index=bars.index)
        signals['signal'] = 0.0

        # signal信号计算，赋值（1, -1 or 0）
        signals['short_mavg'] = bars['close'].rolling(min_periods=1,window=self.short_period).mean()
        signals['long_mavg'] = bars['close'].rolling(min_periods=1,window=self.long_period).mean()

        signals['signal'][self.short_period:] = np.where(signals['short_mavg'][self.short_period:]
            > signals['long_mavg'][self.short_period:], 1.0, -1.0)

        # 产生交易信号, Take the difference of the signals in order to generate actual trading orders
        signals['positions'] = signals['signal'].diff()
        signals['close'] = bars['close']

        return signals


########################################################################
class MarketOnClosePortfolio(Portfolio):
    """绩效计算"""

    # ----------------------------------------------------------------------
    def __init__(self):
        self.symbol = ''
        self.tradeDict = OrderedDict()      # 成交字典

    # ----------------------------------------------------------------------
    def generate_positions(self):
        """（必须由用户继承实现）"""
        pass

    # ----------------------------------------------------------------------
    def backtest_portfolio(self, signals):
        """生成成交字典"""

        # 处理交易信号数据，生成回测的trades数据
        # 1. 丢弃na行
        trades = signals[signals['positions'] != 0].dropna()
        # 2. 两个信号的差值作为入场信号的策略，需要等待‘穿越’的情形，即signals['signal'].diff()的第二行
        trades = trades[1:]
        # 3. 处理反手策略，第一次开仓时双倍仓位的错误
        trade_1st = trades.iloc[0]['positions']
        if abs(trade_1st) > 1:
            trades.ix[0, 'positions'] = trade_1st / 2
        # print trades

        # 生成tradeDict成交字典
        for tradeID, dt in enumerate(trades.index):
            trade_detail = trades.iloc[tradeID]
            trade = VtTradeData()
            trade.symbol = self.symbol
            trade.tradeID = tradeID
            if trade_detail['positions'] > 0:
                trade.direction = DIRECTION_LONG
            else:
                trade.direction = DIRECTION_SHORT
            trade.price = trade_detail['close']
            trade.volume = abs(trade_detail['positions'])
            trade.tradeTime = dt
            self.tradeDict[tradeID] = trade

        return self.tradeDict

# ----------------------------------------------------------------------
def runBacktesting(engine):
    """开始跑回测"""
    engine.runBacktesting()

    # 显示回测结果
    # spyder或者ipython notebook中运行时，会弹出盈亏曲线图
    # 直接在cmd中回测则只会打印一些回测数值
    engine.showBacktestingResult()

# ----------------------------------------------------------------------
def runOptimization(engine, strategyName, setting):
    """单线程优化"""

    # 性能测试环境：I7-3770，主频3.4G, 8核心，内存16G，Windows 7 专业版
    # 测试时还跑着一堆其他的程序，性能仅供参考
    import time

    start = time.time()
    print u'单进程优化模式'
    # 运行单进程优化函数，自动输出结果，耗时：359秒
    engine.runOptimization(strategyName, setting)
    print u'耗时：%s' % (time.time() - start)

# ----------------------------------------------------------------------
def runParallelOptimization(engine, strategyName, portfolioName, setting):
    """多进程优化"""

    # 性能测试环境：I7-3770，主频3.4G, 8核心，内存16G，Windows 7 专业版
    # 测试时还跑着一堆其他的程序，性能仅供参考
    import time

    start = time.time()
    print u'多进程优化模式'
    # 多进程优化，耗时：89秒
    engine.runParallelOptimization(strategyName, portfolioName, setting)
    print u'耗时：%s' % (time.time() - start)

# ----------------------------------------------------------------------
def backtesting():
    """运行回测"""

    # 参数
    initCapital = 10000
    symbol = 'rb0000'
    slippage = 1
    rate = 1.0 / 10000
    size = 10
    startDate = '20160101'
    initDays = 0
    endDate = ''
    dbName = MINUTE_DB_NAME

    # 创建回测引擎
    engine = BacktestEngine()

    # 设置回测用的数据起始日期
    engine.setStartDate(startDate, initDays)
    engine.setEndDate(endDate)

    # 设置产品相关参数
    engine.setInitCapital(initCapital)  # 设置初始资金
    engine.setSlippage(slippage)  # 设置滑点
    engine.setRate(rate)  # 设置佣金
    engine.setSize(size)  # 设置合约大小

    # 设置使用的历史数据库
    engine.setDatabase(dbName, symbol)

    # 在引擎中创建策略对象
    d = {
        'bar_period': '10min',
        'short_period': 10,
        'long_period': 23,
    }

    engine.initStrategy(MovingAverageCrossStrategy, d)

    # 在引擎中创建Portfolio对象
    engine.initPortfolio(MarketOnClosePortfolio)

    # 跑优化参数
    setting = OptimizationSetting()  # 新建一个优化任务设置对象
    setting.setOptimizeTarget('capital')  # 设置优化排序的目标是策略净盈利
    setting.addParameter('bar_period', '10min')  # 增加固定参数，无需优化
    setting.addParameter('short_period', 6, 10, 1)  # 增加第一个优化参数atrLength，起始11，结束12，步进1
    setting.addParameter('long_period', 18, 20, 1)  # 增加第二个优化参数atrMa，起始20，结束30，步进1

    # 执行回测、单线程优化或者多线程优化(三选一）
    # runBacktesting(engine)
    # runOptimization(engine, MovingAverageCrossStrategy, setting)
    # runParallelOptimization(engine, MovingAverageCrossStrategy, MarketOnClosePortfolio, setting)
    runBacktesting(engine)

if __name__ == "__main__":
    backtesting()

