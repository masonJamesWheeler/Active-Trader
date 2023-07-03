from time import sleep

from ib_insync import *

class IB_CLIENT:
    def __init__(self, host="127.0.0.1", port = 7497, client_id=0):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.ib.connect(self.host, self.port, self.client_id)
        accountSummary = self.ib.accountSummary()
        self.accountSummaryDict = {}
        for i in range(len(accountSummary)):
            self.accountSummaryDict[accountSummary[i].tag] = i

    def buy_shares_mkt(self, contract, quantity, algo_type='Adaptive', algo_params=None):
        baseOrder = MarketOrder('BUY', quantity)
        baseOrder.algoStrategy = "Adaptive"
        baseOrder.algoParams = [TagValue("adaptivePriority", "Normal")]
        trade = self.ib.placeOrder(contract, baseOrder)
        return trade

    def sell_shares_mkt(self, contract, quantity):
        baseOrder = MarketOrder('SELL', quantity)
        baseOrder.algoStrategy = "Adaptive"
        baseOrder.algoParams = [TagValue("adaptivePriority", "Normal")]
        trade = self.ib.placeOrder(contract, baseOrder)
        return trade

    def create_contract(self, ticker, exchange='SMART', currency='USD'):
        contract = Stock(ticker, exchange, currency)
        return contract

    def get_orders(self):
        return self.ib.reqOpenOrders()

    def cancel_all_trades(self):
        self.ib.reqGlobalCancel()

    def get_cash_balance(self):
        return float(self.ib.accountSummary()[self.accountSummaryDict['AvailableFunds']].value)

    def get_shares(self, ticker):
        if ticker not in self.ib.positions():
            return 0
        else:
            return self.ib.positions()[ticker].position

    def get_positions(self):
        return self.ib.positions()

    def get_portfolio_value(self):
        return self.ib.accountSummary()[self.accountSummaryDict['NetLiquidation']].value


client = IB_CLIENT()
sleep(1)
# contract = client.create_contract('AAPL')
# trade = client.buy_shares_mkt(contract, 5)
# client.cancel_all_trades()
print(client.get_cash_balance())
print(client.get_positions())
print(client.get_shares('AAPL'))
print(client.get_portfolio_value())



