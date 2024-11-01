import numpy as np
import pandas as pd
import tensorflow as tf
import schedule
import os
import time

data_save_dir = 'data/raw_data'  # 数据保存目录

def train_model():
    """训练 LSTM 模型"""
    file_path = os.path.join(data_save_dir, 'BTCUSDT_15m.csv')
    if not os.path.exists(file_path):
        print("数据文件不存在，无法训练模型。")
        return

    # 加载最新数据
    df = pd.read_csv(file_path)
    df['收盘价'] = df['收盘价'].astype(float)
    sequence_length = 50

    X = []
    y = []
    for i in range(len(df) - sequence_length):
        X.append(df['收盘价'].iloc[i:i + sequence_length].values)
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

# 每天定时训练模型
schedule.every().day.at("00:00").do(train_model)

while True:
    train_model()
    schedule.run_pending()
    time.sleep(1)