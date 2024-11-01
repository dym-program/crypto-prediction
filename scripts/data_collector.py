import pandas as pd
from binance.client import Client
import configparser
import schedule
import time
import os
import sys
import datetime
import pytz  # 添加这一行

# 从配置文件读取配置
config = configparser.ConfigParser()
config.read('config/config.ini')

api_key = config['binance']['api_key']
api_secret = config['binance']['api_secret']
data_save_dir = config['settings']['data_save_dir']

# 设置代理
proxies = {
    "http": "http://127.0.0.1:10809",
    "https": "http://127.0.0.1:10809"
}

client = Client(api_key, api_secret, requests_params={"proxies": proxies})

china_tz = pytz.timezone('Asia/Shanghai')  # 定义中国时区

def get_historical_klines(symbol, interval, start_str, end_str=None):
    """获取历史K线数据"""
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    df = pd.DataFrame(klines, columns=['开盘时间', '开盘价', '最高价', '最低价', '收盘价', '成交量', '收盘时间', '成交额', '成交笔数', '主动买入成交量', '主动买入成交额', '忽略'])
    df['开盘时间'] = pd.to_datetime(df['开盘时间'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(china_tz)
    df['收盘时间'] = pd.to_datetime(df['收盘时间'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(china_tz)
    return df

# 获取 BTCUSDT 的初始 K 线数据
if len(sys.argv) > 1 and sys.argv[1] == "init":
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    file_path_btc = os.path.join(data_save_dir, 'BTCUSDT_15m.csv')
    btc_df = get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_15MINUTE, '1 Jan 2024')
    btc_df.to_csv(file_path_btc, index=False)

def get_latest_klines(symbol, interval, start_timestamp):
    """获取从指定时间点开始的最新K线数据"""
    klines = client.get_klines(symbol=symbol, interval=interval, startTime=start_timestamp)
    df = pd.DataFrame(klines, columns=['开盘时间', '开盘价', '最高价', '最低价', '收盘价', '成交量', '收盘时间', '成交额', '成交笔数', '主动买入成交量', '主动买入成交额', '忽略'])
    df['开盘时间'] = pd.to_datetime(df['开盘时间'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(china_tz)
    df['收盘时间'] = pd.to_datetime(df['收盘时间'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(china_tz)
    return df

def update_data(symbol, interval):
    """更新数据并追加到CSV文件，补全中断期间的缺失数据"""
    file_path = os.path.join(data_save_dir, f'{symbol}_{interval}.csv')
    
    if os.path.exists(file_path):
        # 加载已有的数据，获取最后的时间戳
        existing_df = pd.read_csv(file_path)
        existing_df['收盘时间'] = pd.to_datetime(existing_df['收盘时间']).dt.tz_localize(china_tz)  # 确保收盘时间是中国时区
        last_timestamp = int(existing_df['收盘时间'].max().timestamp() * 1000)
    else:
        # 如果没有文件，从头开始获取数据
        existing_df = pd.DataFrame()
        last_timestamp = None

    # 当前时间（转换为中国时区）
    current_timestamp = int(datetime.datetime.now(tz=china_tz).timestamp() * 1000)

    # 获取缺失的K线数据，直到最新时间
    all_new_data = []

    while last_timestamp is None or last_timestamp < current_timestamp:
        new_data_df = get_latest_klines(symbol, interval, last_timestamp)
        
        if new_data_df.empty:
            break

        # 更新最新的时间戳
        last_timestamp = new_data_df['收盘时间'].max().timestamp() * 1000
        all_new_data.append(new_data_df)

        # 防止短时间内请求过多，稍作休眠
        time.sleep(1)

    if all_new_data:
        new_data_df_combined = pd.concat(all_new_data, ignore_index=True)

        # 合并已有数据和新数据
        updated_df = pd.concat([existing_df, new_data_df_combined], ignore_index=True)

        # 删除重复的行（如果因重叠获取到部分已存在的数据）
        updated_df = updated_df.drop_duplicates(subset='收盘时间')

        # 保存到CSV文件
        updated_df.to_csv(file_path, index=False)
        print(f"{symbol} 数据已更新，共 {len(updated_df)} 行数据。")

# 每15分钟运行一次更新任务
schedule.every(15).minutes.do(update_data, symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_15MINUTE)

while True:
    schedule.run_pending()
    time.sleep(1)
