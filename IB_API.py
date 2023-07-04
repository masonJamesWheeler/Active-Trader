from time import sleep
from ib_insync import *

class IB_CLIENT:
    def __init__(self, host="127.0.0.1", port = 7497, client_id=0):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.ib.connect(self.host, self.port, self.client_id)
        self.accountSummary = self.ib.accountSummary()
        self.accountSummaryDict = {}
        for i in range(len(self.accountSummary)):
            self.accountSummaryDict[self.accountSummary[i].tag] = i

    def buy_shares_mkt(self, ticker, quantity, algo_type='Adaptive', algo_params=None):
        contract = self.create_contract(ticker)
        baseOrder = MarketOrder('BUY', quantity)
        trade = self.ib.placeOrder(contract, baseOrder)
        return trade

    def sell_shares_mkt(self, ticker, quantity):
        contract = self.create_contract(ticker)
        baseOrder = MarketOrder('SELL', quantity)
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
        return float(self.accountSummary[self.accountSummaryDict['AvailableFunds']].value)

    def get_shares(self, ticker):
        for position in self.ib.positions():
            if position.contract.symbol == ticker:
                return int(position.position)
        return 0

    def get_ticker_price(self, ticker):
        contract = self.create_contract(ticker)
        ticker = self.ib.reqMktData(contract)
        # Wait for the ticker to update
        while not ticker.marketPrice():
            sleep(0.1)
        return float(ticker.marketPrice())

    def get_positions(self):
        positions = {self.ib.positions()[i].contract.symbol: self.ib.positions()[i].position for i in range(len(self.ib.positions()))}
        return positions

    def get_portfolio_value(self):
        #  Force client to update portfolio value
        return float(self.accountSummary[self.accountSummaryDict['NetLiquidation']].value)

if __name__ == "__main__":
    ib_client = IB_CLIENT()
    for i in range(1000):
        print(ib_client.get_portfolio_value())


