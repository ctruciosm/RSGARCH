import pandas as pd
import os.path
import math
import time
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser

### API
binance_api_key = 'oGNvbXiviy06tz68men6hxr21D7weJQ32G2aunYy348aQdowPbTLLPrK2dJRUM1w'    #Enter your own API-key here
binance_api_secret = '0U3QvigIe9MaBI6uGfSpUofS9fCzyPl0Wwg2HGZL28hF6t2yVIBLeeHOASQgTbCR' #Enter your own API-secret here

### CONSTANTS
binsizes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)


### Main Function
def get_hfd_binance(symbol, frequency, start = '1 Jan 2017', end = 'last'):
    filename = '%s-%s-data.csv' % (symbol, frequency)
    if os.path.isfile(filename): 
        data_df = pd.read_csv(filename)
    else: 
        data_df = pd.DataFrame()
    start = datetime.strptime(start, '%d %b %Y')
    if end == 'last':
        end = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=frequency)[-1][0], unit='ms')
    else:
        end = datetime.strptime(end, '%d %b %Y')
    print('Downloading %s data for %s from %s to %s ...' % (frequency, symbol, start, end))
    candles = binance_client.get_historical_klines(symbol, frequency, start.strftime("%d %b %Y %H:%M:%S"), end.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(candles,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data_df = data
    data_df.set_index('timestamp', inplace=True)
    data_df.to_csv(filename)
    return data_df


# On 27/05/2020 the Top 10 MarketCap was
# Fees:
# 30d Trade Volume (BTC) < 50BTC  and/or BNB Balance >0
# Maker/Taker 0.1%, with BNB 0.075%
# Top 10 on June 2, 2020
BTC = get_hfd_binance("BTCUSDT", '1d', start = '1 Jan 2017')
ETH = get_hfd_binance("ETHUSDT", '1d', start = '1 Jan 2017')
import pandas as pd
import os.path
import math
import time
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser

### API
binance_api_key = 'oGNvbXiviy06tz68men6hxr21D7weJQ32G2aunYy348aQdowPbTLLPrK2dJRUM1w'    #Enter your own API-key here
binance_api_secret = '0U3QvigIe9MaBI6uGfSpUofS9fCzyPl0Wwg2HGZL28hF6t2yVIBLeeHOASQgTbCR' #Enter your own API-secret here

### CONSTANTS
binsizes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)


### Main Function
def get_hfd_binance(symbol, frequency, start = '1 Jan 2017', end = 'last'):
    filename = '%s-%s-data.csv' % (symbol, frequency)
    if os.path.isfile(filename): 
        data_df = pd.read_csv(filename)
    else: 
        data_df = pd.DataFrame()
    start = datetime.strptime(start, '%d %b %Y')
    if end == 'last':
        end = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=frequency)[-1][0], unit='ms')
    else:
        end = datetime.strptime(end, '%d %b %Y')
    print('Downloading %s data for %s from %s to %s ...' % (frequency, symbol, start, end))
    candles = binance_client.get_historical_klines(symbol, frequency, start.strftime("%d %b %Y %H:%M:%S"), end.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(candles,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data_df = data
    data_df.set_index('timestamp', inplace=True)
    data_df.to_csv(filename)
    return data_df


# On 27/05/2020 the Top 10 MarketCap was
# Fees:
# 30d Trade Volume (BTC) < 50BTC  and/or BNB Balance >0
# Maker/Taker 0.1%, with BNB 0.075%
# Top 10 on June 2, 2020
BTC = get_hfd_binance("BTCUSDT", '1d', start = '1 Jan 2017')
ETH = get_hfd_binance("ETHUSDT", '1d', start = '1 Jan 2017')
import pandas as pd
import os.path
import math
import time
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser

### API
binance_api_key = 'oGNvbXiviy06tz68men6hxr21D7weJQ32G2aunYy348aQdowPbTLLPrK2dJRUM1w'    #Enter your own API-key here
binance_api_secret = '0U3QvigIe9MaBI6uGfSpUofS9fCzyPl0Wwg2HGZL28hF6t2yVIBLeeHOASQgTbCR' #Enter your own API-secret here

### CONSTANTS
binsizes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)


### Main Function
def get_hfd_binance(symbol, frequency, start = '1 Jan 2017', end = 'last'):
    filename = '%s-%s-data.csv' % (symbol, frequency)
    if os.path.isfile(filename): 
        data_df = pd.read_csv(filename)
    else: 
        data_df = pd.DataFrame()
    start = datetime.strptime(start, '%d %b %Y')
    if end == 'last':
        end = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=frequency)[-1][0], unit='ms')
    else:
        end = datetime.strptime(end, '%d %b %Y')
    print('Downloading %s data for %s from %s to %s ...' % (frequency, symbol, start, end))
    candles = binance_client.get_historical_klines(symbol, frequency, start.strftime("%d %b %Y %H:%M:%S"), end.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(candles,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data_df = data
    data_df.set_index('timestamp', inplace=True)
    data_df.to_csv(filename)
    return data_df


# On 27/05/2020 the Top 10 MarketCap was
# Fees:
# 30d Trade Volume (BTC) < 50BTC  and/or BNB Balance >0
# Maker/Taker 0.1%, with BNB 0.075%
# Top 10 on June 2, 2020
BTC = get_hfd_binance("BTCUSDT", '1d', start = '1 Jan 2017')
ETH = get_hfd_binance("ETHUSDT", '1d', start = '1 Jan 2017')
BNB = get_hfd_binance("BNBUSDT", '1d', start = '1 Jan 2017')
XRP = get_hfd_binance("XRPUSDT", '1d', start = '1 Jan 2017')
ADA = get_hfd_binance("ADAUSDT", '1d', start = '1 Jan 2017')
LTC = get_hfd_binance("LTCUSDT", '1d', start = '1 Jan 2017')
DOGE = get_hfd_binance("DOGEUSDT", '1d', start = '1 Jan 2017')
