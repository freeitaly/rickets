# encoding: UTF-8

'''
本文件中包含的是向量化回测模块的回测引擎
'''

from __future__ import division
from datetime import datetime, timedelta
from collections import OrderedDict
from itertools import product
import pandas as pd
import numpy as np
import time
import math
import multiprocessing
import pymongo
from pymongo.errors import ConnectionFailure
from backtestConstant import *


########################################################################
class BacktestEngine(object):
    """向量化回测引擎"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""

        # 回测相关
        self.strategy = None        # 回测策略

        self.initCapital = 0        # 回测时初始资金
        self.slippage = 0           # 回测时假设的滑点
        self.rate = 0               # 回测时假设的佣金比例（适用于百分比佣金）
        self.fixedCommission = 0    # 回测时假设的佣金比例（适用于固定佣金）
        self.size = 1               # 合约大小，默认为1

        self.dbClient = None        # 数据库客户端
        self.dbCursor = None        # 数据库指针

        self.dbName = ''            # 回测数据库名
        self.symbol = ''            # 回测集合名
        
        self.dataStartDate = None       # 回测数据开始日期，datetime对象
        self.dataEndDate = None         # 回测数据结束日期，datetime对象
        self.strategyStartDate = None   # 策略启动日期（即前面的数据用于初始化），datetime对象
        self.initDays = None

        self.tradeDict = OrderedDict()  # 成交字典
        
    #----------------------------------------------------------------------
    def dbConnect(self):
        """连接MongoDB数据库"""
        if not self.dbClient:
            try:
                self.dbClient = pymongo.MongoClient('10.10.40.20:28019', connectTimeoutMS=500)
                # 调用server_info查询服务器状态，防止服务器异常并未连接成功
                self.dbClient.server_info()
            except ConnectionFailure:
                print u'MongoDB连接失败'

    #----------------------------------------------------------------------
    def setStartDate(self, startDate='20100101', initDays=10):
        """设置回测的启动日期"""
        self.startDate = startDate
        self.initDays = initDays
        
        self.dataStartDate = datetime.strptime(startDate, '%Y%m%d')
        
        initTimeDelta = timedelta(initDays)
        self.strategyStartDate = self.dataStartDate + initTimeDelta
        
    #----------------------------------------------------------------------
    def setEndDate(self, endDate=''):
        """设置回测的结束日期"""
        self.endDate = endDate
        if endDate:
            self.dataEndDate= datetime.strptime(endDate, '%Y%m%d')
            # 若不修改时间则会导致不包含dataEndDate当天数据
            self.dataEndDate = self.dataEndDate.replace(hour=23, minute=59)
        
    #----------------------------------------------------------------------
    def setInitCapital(self, initCapital):
        """设置初始资金"""
        self.initCapital = initCapital

    #----------------------------------------------------------------------
    def setDatabase(self, dbName, symbol):
        """设置历史数据所用的数据库"""
        self.dbName = dbName
        self.symbol = symbol

    #----------------------------------------------------------------------
    def loadCursor(self):
        """载入历史数据"""
        self.dbConnect()
        collection = self.dbClient[self.dbName][self.symbol]

        self.output(u'开始载入数据')
        # 载入回测数据
        if not self.dataEndDate:
            flt = {'datetime': {'$gte': self.strategyStartDate}}  # 数据过滤条件
        else:
            flt = {'datetime': {'$gte': self.strategyStartDate,
                                '$lte': self.dataEndDate}}

        self.dbCursor = collection.find(flt)
        self.output(u'载入完成，数据量：%s' %(self.dbCursor.count()))

        return self.dbCursor

    #----------------------------------------------------------------------
    def runBacktesting(self):
        """运行回测"""
        self.output(u'开始回测')
        
        signals = self.strategy.generate_signals()
        self.output(u'策略信号生成完成')

        self.tradeDict = self.portfolio.backtest_portfolio(signals)
        self.output(u'策略成交字典生成完成')
        self.output(u'回测结束')

    #----------------------------------------------------------------------
    def initStrategy(self, strategyClass, setting=None):
        """
        初始化策略
        setting是策略的参数设置，如果使用类中写好的默认设置则可以不传该参数
        """
        self.strategy = strategyClass(self, setting)

    # ----------------------------------------------------------------------
    def initPortfolio(self, portfolioClass):
        """
        初始化策略
        """
        self.portfolio = portfolioClass()
        self.portfolio.symbol = self.symbol

    #----------------------------------------------------------------------
    def output(self, content):
        """输出内容"""
        print str(datetime.now()) + "\t" + content 
    
    #----------------------------------------------------------------------
    def calculateBacktestingResult(self):
        """
        计算回测结果
        """
        self.output(u'计算回测结果')
        
        # 首先基于回测后的成交记录，计算每笔交易的盈亏
        resultList = []             # 交易结果列表

        longTrade = []              # 未平仓的多头交易
        shortTrade = []             # 未平仓的空头交易

        tradeTimeList = []          # 每笔成交时间戳
        posList = [0]               # 每笔成交后的持仓情况        

        for trade in self.tradeDict.values():
            # 多头交易
            if trade.direction == DIRECTION_LONG:
                # 如果尚无空头交易
                if not shortTrade:
                    longTrade.append(trade)
                # 当前多头交易为平空
                else:
                    while True:
                        entryTrade = shortTrade[0]
                        exitTrade = trade
                        
                        # 清算开平仓交易
                        closedVolume = min(exitTrade.volume, entryTrade.volume)
                        result = TradingResult(entryTrade.price, entryTrade.tradeTime,
                                               exitTrade.price, exitTrade.tradeTime,
                                               -closedVolume, self.rate, self.fixedCommission, self.slippage, self.size)
                        resultList.append(result)
                        
                        posList.extend([-1,0])
                        tradeTimeList.extend([result.entryDt, result.exitDt])
                        
                        # 计算未清算部分
                        entryTrade.volume -= closedVolume
                        exitTrade.volume -= closedVolume
                        
                        # 如果开仓交易已经全部清算，则从列表中移除
                        if not entryTrade.volume:
                            shortTrade.pop(0)
                        
                        # 如果平仓交易已经全部清算，则退出循环
                        if not exitTrade.volume:
                            break
                        
                        # 如果平仓交易未全部清算，
                        if exitTrade.volume:
                            # 且开仓交易已经全部清算完，则平仓交易剩余的部分
                            # 等于新的反向开仓交易，添加到队列中
                            if not shortTrade:
                                longTrade.append(exitTrade)
                                break
                            # 如果开仓交易还有剩余，则进入下一轮循环
                            else:
                                pass
                        
            # 空头交易        
            else:
                # 如果尚无多头交易
                if not longTrade:
                    shortTrade.append(trade)
                # 当前空头交易为平多
                else:                    
                    while True:
                        entryTrade = longTrade[0]
                        exitTrade = trade
                        
                        # 清算开平仓交易
                        closedVolume = min(exitTrade.volume, entryTrade.volume)
                        result = TradingResult(entryTrade.price, entryTrade.tradeTime,
                                               exitTrade.price, exitTrade.tradeTime,
                                               closedVolume, self.rate, self.fixedCommission, self.slippage, self.size)
                        resultList.append(result)
                        
                        posList.extend([1,0])
                        tradeTimeList.extend([result.entryDt, result.exitDt])

                        # 计算未清算部分
                        entryTrade.volume -= closedVolume
                        exitTrade.volume -= closedVolume
                        
                        # 如果开仓交易已经全部清算，则从列表中移除
                        if not entryTrade.volume:
                            longTrade.pop(0)
                        
                        # 如果平仓交易已经全部清算，则退出循环
                        if not exitTrade.volume:
                            break
                        
                        # 如果平仓交易未全部清算，
                        if exitTrade.volume:
                            # 且开仓交易已经全部清算完，则平仓交易剩余的部分
                            # 等于新的反向开仓交易，添加到队列中
                            if not longTrade:
                                shortTrade.append(exitTrade)
                                break
                            # 如果开仓交易还有剩余，则进入下一轮循环
                            else:
                                pass                    
                    
        # 检查是否有交易
        if not resultList:
            self.output(u'无交易结果')
            return {}
        
        # 然后基于每笔交易的结果，我们可以计算具体的盈亏曲线和最大回撤等        
        capital = 0             # 资金
        maxCapital = 0          # 资金最高净值
        drawdown = 0            # 回撤
        drawdownRatio = 0       # 回撤率

        totalResult = 0         # 总成交数量
        totalTurnover = 0       # 总成交金额（合约面值）
        totalCommission = 0     # 总手续费
        totalSlippage = 0       # 总滑点
        
        timeList = []           # 时间序列
        pnlList = []            # 每笔盈亏序列
        capitalList = []        # 盈亏汇总的时间序列
        drawdownList = []       # 回撤的时间序列
        drawdownRatioList = []  # 回撤率的时间序列
        dailyList = []          # 每日盈亏序列

        winningResult = 0       # 盈利次数
        losingResult = 0        # 亏损次数		
        totalWinning = 0        # 总盈利金额		
        totalLosing = 0         # 总亏损金额        

        resultPdList = []
        for result in resultList:
            resultPdList.append(result.__dict__)        # 下一步生成pd使用
            capital += result.pnl
            maxCapital = max(capital, maxCapital)
            drawdown = capital - maxCapital
            drawdownRatio = drawdown / (self.initCapital + maxCapital)
            pnlList.append(result.pnl)
            timeList.append(result.exitDt)      # 交易的时间戳使用平仓时间
            capitalList.append(capital)
            drawdownList.append(drawdown)
            drawdownRatioList.append(drawdownRatio)

            totalResult += 1
            totalTurnover += result.turnover
            totalCommission += result.commission
            totalSlippage += result.slippage
            
            if result.pnl >= 0:
                winningResult += 1
                totalWinning += result.pnl
            else:
                losingResult += 1
                totalLosing += result.pnl
                
        # 计算盈亏相关数据

        winningRate = winningResult / totalResult * 100  # 胜率

        averageWinning = 0  # 这里把数据都初始化为0
        averageLosing = 0
        profitLossRatio = 0

        if winningResult:
            averageWinning = totalWinning / winningResult  # 平均每笔盈利
        if losingResult:
            averageLosing = totalLosing / losingResult  # 平均每笔亏损
        if averageLosing:
            profitLossRatio = -averageWinning / averageLosing  # 盈亏比
        if totalLosing:
            profitFactor = -totalWinning/totalLosing         # 获利因子

        # 生成盈亏df，计算相关绩效
        data = pd.DataFrame(resultPdList)
        data['capital'] = self.initCapital + data['pnl'].cumsum()
        return_rate = (data.iloc[-1]['capital'] - self.initCapital) / self.initCapital      # 收益率
        trading_day = (data.iloc[-1]['exitDt'] - data.iloc[0]['entryDt']).days      # 交易日
        annually_return_rate = return_rate / trading_day * 365      # 年化收益率
        data['daily_return_rate'] = data['capital'].pct_change()        # 日收益率
        std = data['daily_return_rate'].std()      # 日收益率标准差
        annually_std = std * math.sqrt(250)     # 年化标准差
        sharp = (annually_return_rate - 0.02) / annually_std        # 夏普比率
        winningCount = len(data[data['pnl'] >= 0].index)       # 盈利次数
        losingCount = len(data[data['pnl'] < 0].index)     # 亏损次数

        # 计算每日绩效
        data0 = data.set_index(data['entryDt'])
        data0 = data0.resample('D').sum()
        winningDailyMax = data0['pnl'].max()
        losingDailyMax = data0['pnl'].min()
        dailyList = data0['pnl'].dropna().tolist()

        # 计算连续亏损、盈利
        maxConWin = 0
        maxConLose = 0
        i = j = 0
        for pnl in data['pnl']:
            if pnl >= 0:
                i += 1
                j = 0
                maxConWin = max(maxConWin, i)
            else:
                i = 0
                j += 1
                maxConLose = max(maxConLose, j)

        # # 规范日期格式
        # data['entryDt'] = data['entryDt'].apply(lambda x: x.strftime('%Y/%m/%d %H:%M:%S'))
        # data['exitDt'] = data['exitDt'].apply(lambda x: x.strftime('%Y/%m/%d %H:%M:%S'))

        # 回测结果输出到csv
        data.to_csv('./results/%s.csv' % time.strftime('%Y-%m-%d %H.%M.%S',time.localtime()), date_format='%Y/%m/%d %H:%M:%S')

        # 返回回测结果
        d = {}
        d['capital'] = capital
        d['maxCapital'] = maxCapital
        d['drawdown'] = drawdown
        d['totalResult'] = totalResult
        d['totalTurnover'] = totalTurnover
        d['totalCommission'] = totalCommission
        d['totalSlippage'] = totalSlippage
        d['timeList'] = timeList
        d['pnlList'] = pnlList
        d['dailyList'] = dailyList
        d['capitalList'] = capitalList
        d['drawdownList'] = drawdownList
        d['drawdownRatio'] = '%.2f' % (min(drawdownRatioList) * 100) + ' %'
        d['winningRate'] = winningRate
        d['averageWinning'] = averageWinning
        d['averageLosing'] = averageLosing
        d['profitLossRatio'] = profitLossRatio
        d['profitFactor'] = profitFactor
        d['return_rate'] = '%.2f' % (return_rate * 100) + ' %'
        d['annually_return_rate'] = '%.2f' % (annually_return_rate * 100) + ' %'
        d['std'] = std
        d['annually_std'] = annually_std
        d['sharp'] = sharp
        d['winningCount'] = winningCount
        d['losingCount'] = losingCount
        d['maxConWin'] = maxConWin
        d['maxConLose'] = maxConLose
        d['winningDailyMax'] = winningDailyMax
        d['losingDailyMax'] = losingDailyMax
        d['posList'] = posList
        d['tradeTimeList'] = tradeTimeList
        
        return d
        
    #----------------------------------------------------------------------
    def showBacktestingResult(self):
        """显示回测结果"""
        d = self.calculateBacktestingResult()
        
        # 输出
        self.output('-' * 30)
        self.output(u'第一笔交易：\t%s' % d['timeList'][0])
        self.output(u'最后一笔交易：\t%s' % d['timeList'][-1])
        
        self.output(u'总交易次数：\t%s' % formatNumber(d['totalResult']))        
        self.output(u'总盈亏：\t%s' % formatNumber(d['capital']))
        self.output(u'总佣金：\t%s' % formatNumber(d['totalCommission']))
        self.output(u'最大回撤: \t%s' % formatNumber(min(d['drawdownList'])))
        self.output(u'最大回撤率: \t%s' % (d['drawdownRatio']))

        self.output(u'平均每笔盈利：\t%s' %formatNumber(d['capital']/d['totalResult']))
        self.output(u'平均每笔滑点：\t%s' %formatNumber(d['totalSlippage']/d['totalResult']))
        self.output(u'平均每笔佣金：\t%s' %formatNumber(d['totalCommission']/d['totalResult']))

        self.output(u'盈利次数\t\t%s' % (d['winningCount']))
        self.output(u'亏损次数\t\t%s' % (d['losingCount']))
        self.output(u'胜率\t\t%s%%' %formatNumber(d['winningRate']))
        self.output(u'盈利交易平均值\t%s' %formatNumber(d['averageWinning']))
        self.output(u'亏损交易平均值\t%s' %formatNumber(d['averageLosing']))
        self.output(u'盈亏比：\t%s' %formatNumber(d['profitLossRatio']))
        self.output(u'获利因子：\t%s' %formatNumber(d['profitFactor']))

        self.output(u'最大连续盈利次数：\t%s' % (d['maxConWin']))
        self.output(u'最大连续亏损次数：\t%s' % (d['maxConLose']))
        self.output(u'最大日盈利：\t%s' % formatNumber(d['winningDailyMax']))
        self.output(u'最大日亏损：\t%s' % formatNumber(d['losingDailyMax']))


        self.output(u'收益率：\t%s' %(d['return_rate']))
        self.output(u'年化收益率：\t%s' %(d['annually_return_rate']))
        self.output(u'日收益率标准差：\t%s' %formatNumber(d['std']))
        self.output(u'年化标准差：\t%s' %formatNumber(d['annually_std']))
        self.output(u'夏普比率：\t%s' %formatNumber(d['sharp']))

        # 绘图
        import matplotlib.pyplot as plt
        import numpy as np
        
        try:
            import seaborn as sns       # 如果安装了seaborn则设置为白色风格
            sns.set_style('whitegrid')  
        except ImportError:
            pass

        pCapital = plt.subplot(5, 1, 1)
        pCapital.set_ylabel("capital")
        pCapital.plot(d['capitalList'], color='r', lw=0.8)
        
        pDD = plt.subplot(5, 1, 2)
        pDD.set_ylabel("DD")
        pDD.bar(range(len(d['drawdownList'])), d['drawdownList'], color='g')
        
        pPnl = plt.subplot(5, 1, 3)
        pPnl.set_ylabel("pnl")
        pPnl.hist(d['pnlList'], bins=50, color='c')


        pDaily = plt.subplot(5, 1, 4)
        pDaily.set_ylabel("daily")
        pDaily.hist(d['dailyList'], bins=50)

        pPos = plt.subplot(5, 1, 5)
        pPos.set_ylabel("Position")
        if d['posList'][-1] == 0:
            del d['posList'][-1]
        tradeTimeIndex = [item.strftime("%m/%d %H:%M:%S") for item in d['tradeTimeList']]
        xindex = np.arange(0, len(tradeTimeIndex), np.int(len(tradeTimeIndex)/10))
        tradeTimeIndex = map(lambda i: tradeTimeIndex[i], xindex)
        pPos.plot(d['posList'], color='k', drawstyle='steps-pre')
        pPos.set_ylim(-1.2, 1.2)
        plt.sca(pPos)
        plt.tight_layout()
        plt.xticks(xindex, tradeTimeIndex, rotation=30)  # 旋转15
        
        plt.show()

    #----------------------------------------------------------------------
    def setSlippage(self, slippage):
        """设置滑点点数"""
        self.slippage = slippage
        
    #----------------------------------------------------------------------
    def setSize(self, size):
        """设置合约大小"""
        self.size = size
        
    #----------------------------------------------------------------------
    def setRate(self, rate, fixedCommission=0):
        """设置佣金比例"""
        self.rate = rate
        self.fixedCommission = fixedCommission

    #----------------------------------------------------------------------
    def runOptimization(self, strategyClass, optimizationSetting):
        """优化参数"""
        # 获取优化设置        
        settingList = optimizationSetting.generateSetting()
        targetName = optimizationSetting.optimizeTarget
        
        # 检查参数设置问题
        if not settingList or not targetName:
            self.output(u'优化设置有问题，请检查')
        
        # 遍历优化
        resultList = []
        for setting in settingList:
            # self.tradeDict.clear()
            self.output('-' * 30)
            self.output('setting: %s' %str(setting))
            self.initStrategy(strategyClass, setting)
            self.runBacktesting()
            d = self.calculateBacktestingResult()
            try:
                targetValue = d[targetName]
            except KeyError:
                targetValue = 0
            resultList.append(([str(setting)], targetValue))
        
        # 显示结果
        resultList.sort(reverse=True, key=lambda result:result[1])
        self.output('-' * 30)
        self.output(u'优化结果：')
        for result in resultList:
            self.output(u'%s: %s' %(result[0], result[1]))
        return result

    #----------------------------------------------------------------------
    def runParallelOptimization(self, strategyClass, portfolioClass, optimizationSetting):
        """并行优化参数"""
        # 获取优化设置        
        settingList = optimizationSetting.generateSetting()
        targetName = optimizationSetting.optimizeTarget
        
        # 检查参数设置问题
        if not settingList or not targetName:
            self.output(u'优化设置有问题，请检查')
        
        # 多进程优化，启动一个对应CPU核心数量的进程池
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        l = []

        for setting in settingList:
            l.append(pool.apply_async(optimize, (strategyClass, portfolioClass, setting, targetName,
                                                 self.startDate, self.initDays, self.endDate,
                                                 self.initCapital, self.slippage, self.rate, self.size,
                                                 self.dbName, self.symbol)))
        pool.close()
        pool.join()
        
        # 显示结果
        resultList = [res.get() for res in l]
        resultList.sort(reverse=True, key=lambda result:result[1])
        self.output('-' * 30)
        self.output(u'优化结果：')
        for result in resultList:
            self.output(u'%s: %s' %(result[0], result[1]))    
            

########################################################################
class VtTradeData(object):
    """成交数据类"""

    # ----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        super(VtTradeData, self).__init__()

        # 代码编号相关
        self.symbol = EMPTY_STRING  # 合约代码

        self.tradeID = EMPTY_STRING  # 成交编号

        # 成交相关
        self.direction = EMPTY_UNICODE  # 成交方向
        self.offset = EMPTY_UNICODE  # 成交开平仓
        self.price = EMPTY_FLOAT  # 成交价格
        self.volume = EMPTY_INT  # 成交数量
        self.tradeTime = EMPTY_STRING  # 成交时间
        self.tradeDate = EMPTY_STRING  # 成交日期


########################################################################
class TradingResult(object):
    """每笔交易的结果"""

    #----------------------------------------------------------------------
    def __init__(self, entryPrice, entryDt, exitPrice, 
                 exitDt, volume, rate, fixedCommission, slippage, size):
        """Constructor"""
        self.entryPrice = entryPrice    # 开仓价格
        self.exitPrice = exitPrice      # 平仓价格
        
        self.entryDt = entryDt          # 开仓时间datetime    
        self.exitDt = exitDt            # 平仓时间
        
        self.volume = volume    # 交易数量（+/-代表方向）
        
        self.turnover = (self.entryPrice+self.exitPrice)*size*abs(volume)   # 成交金额
        self.commission = self.turnover*rate + fixedCommission*2         # 手续费成本
        self.slippage = slippage*2*size*abs(volume)                         # 滑点成本
        self.pnl = ((self.exitPrice - self.entryPrice) * volume * size 
                    - self.commission - self.slippage)                      # 净盈亏



########################################################################
class OptimizationSetting(object):
    """优化设置"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self.paramDict = OrderedDict()
        
        self.optimizeTarget = ''        # 优化目标字段

    # ----------------------------------------------------------------------
    def addInitParameter(self, paras):
        """增加策略初始化参数"""
        for name, value in paras.items():
            self.paramDict[name] = [value]
    #----------------------------------------------------------------------
    def addParameter(self, name, start, end=None, step=None):
        """增加优化参数"""
        if end is None and step is None:
            self.paramDict[name] = [start]
            return 
        
        if end < start:
            print u'参数起始点必须不大于终止点'
            return
        
        if step <= 0:
            print u'参数布进必须大于0'
            return
        
        l = []
        param = start
        
        while param <= end:
            l.append(param)
            param += step
        
        self.paramDict[name] = l
        
    #----------------------------------------------------------------------
    def generateSetting(self):
        """生成优化参数组合"""
        # 参数名的列表
        nameList = self.paramDict.keys()
        paramList = self.paramDict.values()
        
        # 使用迭代工具生产参数对组合
        productList = list(product(*paramList))
        
        # 把参数对组合打包到一个个字典组成的列表中
        settingList = []
        for p in productList:
            d = dict(zip(nameList, p))
            settingList.append(d)
    
        return settingList
    
    #----------------------------------------------------------------------
    def setOptimizeTarget(self, target):
        """设置优化目标字段"""
        self.optimizeTarget = target


#----------------------------------------------------------------------
def formatNumber(n):
    """格式化数字到字符串"""
    rn = round(n, 2)        # 保留两位小数
    return format(rn, ',')  # 加上千分符
    
#----------------------------------------------------------------------
def optimize(strategyClass, portfolioClass, setting, targetName,
             startDate, initDays, endDate,
             initCapital, slippage, rate, size,
             dbName, symbol):
    """多进程优化时跑在每个进程中运行的函数"""
    engine = BacktestEngine()
    # engine.setBacktestingMode(mode)
    engine.setStartDate(startDate, initDays)
    engine.setInitCapital(initCapital)
    engine.setEndDate(endDate)
    engine.setSlippage(slippage)
    engine.setRate(rate)
    engine.setSize(size)
    engine.setDatabase(dbName, symbol)
    
    engine.initStrategy(strategyClass, setting)
    engine.initPortfolio(portfolioClass)
    engine.runBacktesting()
    d = engine.calculateBacktestingResult()
    try:
        targetValue = d[targetName]
    except KeyError:
        targetValue = 0            
    return (str(setting), targetValue)    



if __name__ == '__main__':
    # 以下内容是一段回测脚本的演示，用户可以根据自己的需求修改
    # 建议使用ipython notebook或者spyder来做回测
    # 同样可以在命令模式下进行回测（一行一行输入运行）
    pass