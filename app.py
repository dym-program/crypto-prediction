import pandas as pd
from binance.client import Client
import configparser
import schedule
import time
import os
import datetime
import numpy as np
import tensorflow as tf

# 从配置文件读取配置
config = configparser.ConfigParser()
config.read('config/config.ini')

api_key = config['binance']['api_key']
api_secret = config['binance']['api_secret']
data_save_dir = config['settings']['data_save_dir']

# 设置代理
proxies = {
    "http": "http://127.0.0.1:10808",
    "https": "http://127.0.0.1:10808"
}

client = Client(api_key, api_secret, requests_params={"proxies": proxies})


def get_historical_klines(symbol, interval, start_str, end_str=None):
    """获取历史K线数据"""
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    df = pd.DataFrame(klines, columns=['开盘时间', '开盘价', '最高价', '最低价', '收盘价', '成交量', '收盘时间', '成交额', '成交笔数', '主动买入成交量', '主动买入成交额', '忽略'])
    df['开盘时间'] = pd.to_datetime(df['开盘时间'], unit='ms')
    df['收盘时间'] = pd.to_datetime(df['收盘时间'], unit='ms')
    return df


def get_latest_klines(symbol, interval, start_timestamp):
    """获取从指定时间点开始的最新K线数据"""
    klines = client.get_klines(symbol=symbol, interval=interval, startTime=start_timestamp)
    df = pd.DataFrame(klines, columns=['开盘时间', '开盘价', '最高价', '最低价', '收盘价', '成交量', '收盘时间', '成交额', '成交笔数', '主动买入成交量', '主动买入成交额', '忽略'])
    df['开盘时间'] = pd.to_datetime(df['开盘时间'], unit='ms')
    df['收盘时间'] = pd.to_datetime(df['收盘时间'], unit='ms')
    return df


def update_data(symbol, interval):
    """更新数据并追加到CSV文件，补全中断期间的缺失数据"""
    file_path = os.path.join(data_save_dir, f'{symbol}_{interval}.csv')
    
    if os.path.exists(file_path):
        # 加载已有的数据，获取最后的时间戳
        existing_df = pd.read_csv(file_path)
        last_timestamp = int(pd.to_datetime(existing_df['收盘时间']).max().timestamp() * 1000)
    else:
        # 如果没有文件，从头开始获取数据
        existing_df = pd.DataFrame()
        last_timestamp = None

    # 当前时间
    current_timestamp = int(datetime.datetime.now().timestamp() * 1000)

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


def extract_features():
    """从收集的数据中提取特征，保存到新的 CSV 文件中"""
    file_path_btc = os.path.join(data_save_dir, 'BTCUSDT_15m.csv')
    if not os.path.exists(file_path_btc):
        print("数据文件不存在，无法提取特征。")
        return

    df = pd.read_csv(file_path_btc)
    df['收盘价'] = df['收盘价'].astype(float)

    # 示例特征提取：计算均线
    df['SMA'] = df['收盘价'].rolling(window=20).mean()
    df['RSI'] = 100 - (100 / (1 + df['收盘价'].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() / df['收盘价'].diff().apply(lambda x: abs(x)).rolling(window=14).mean()))
    df['MACD'] = df['收盘价'].ewm(span=12, adjust=False).mean() - df['收盘价'].ewm(span=26, adjust=False).mean()

    # 保存特征数据
    features_path = os.path.join(data_save_dir, 'processed_data.csv')
    df.to_csv(features_path, index=False)
    print("特征提取完成。")


def train_model():
    """训练 LSTM 模型"""
    features_path = os.path.join(data_save_dir, 'processed_data.csv')
    if not os.path.exists(features_path):
        print("特征数据不存在，无法训练模型。")
        return

    # 加载处理过的数据
    df = pd.read_csv(features_path)
    df = df.dropna()  # 删除包含 NaN 的行

    # 数据准备
    sequence_length = 50
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[['收盘价', 'SMA', 'RSI', 'MACD']].iloc[i:i + sequence_length].values)
        y.append(df['收盘价'].iloc[i + sequence_length])

    X = np.array(X)
    y = np.array(y)
    X = np.expand_dims(X, axis=2)

    # 创建模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    model.fit(X, y, epochs=10, batch_size=32)
    # 保存模型
    model.save('models/BTCUSDT_15m_lstm_model.h5')
    print("模型已保存")


def predict():
    """使用最新数据进行预测"""
    model_path = 'models/BTCUSDT_15m_lstm_model.h5'
    if not os.path.exists(model_path):
        print("模型不存在，无法进行预测。")
        return

    # 加载模型
    model = tf.keras.models.load_model(model_path)
    features_path = os.path.join(data_save_dir, 'processed_data.csv')
    df = pd.read_csv(features_path).dropna()

    # 使用最新的 50 条数据进行预测
    X = df[['收盘价', 'SMA', 'RSI', 'MACD']].iloc[-50:].values
    X = np.expand_dims(X, axis=(0, 2))

    # 进行预测
    prediction = model.predict(X)
    print("预测值：", prediction[-1])


# 每 15 分钟运行数据更新、特征提取、模型训练和预测
schedule.every(15).minutes.do(update_data, symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_15MINUTE)
schedule.every(15).minutes.do(extract_features)
schedule.every(15).minutes.do(train_model)
schedule.every(15).minutes.do(predict)

while True:
    schedule.run_pending()
    time.sleep(1)
