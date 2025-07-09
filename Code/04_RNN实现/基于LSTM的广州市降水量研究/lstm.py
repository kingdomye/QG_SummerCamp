# ================================
# @File         : lstm.py
# @Time         : 2025/07/09
# @Author       : Yingrui Chen
# @description  : 利用pytorch构建RNN循环神经网络模型
#                 并且利用爬虫爬取广州市天气数据
#                 对广州市的降水量数据进行预测
# ================================


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        return out

def prepare_data(data, target_col, sequence_length=30, test_size=0.2):
    """
    Prepare data for LSTM training with robust handling of all data issues
    
    Args:
        data: DataFrame with the time series data (including date column)
        target_col: name of the target column
        sequence_length: number of time steps to use for prediction
        test_size: proportion of data to use for testing
        
    Returns:
        X_train, y_train, X_test, y_test: training and testing data
        train_dates, test_dates: corresponding dates for training and testing
        scalers: dictionary containing feature and target scalers
    """
    # 1. 处理日期列
    if '日期' not in data.columns:
        raise ValueError("Data must contain a '日期' column")
    
    # 创建数据副本避免修改原始数据
    data = data.copy()
    
    # 2. 处理无效日期
    def parse_date(date_str):
        try:
            for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y年%m月%d日', '%m/%d/%Y', '%d-%m-%Y'):
                try:
                    dt = datetime.strptime(str(date_str), fmt)
                    # 验证日期有效性（例如2月30日）
                    if dt.month == 2 and dt.day > 28:
                        if not (dt.year % 4 == 0 and (dt.year % 100 != 0 or dt.year % 400 == 0) and dt.day == 29):
                            return pd.NaT
                    return dt
                except ValueError:
                    continue
            return pd.NaT
        except:
            return pd.NaT
    
    date_objects = data['日期'].apply(parse_date)
    
    # 删除无效日期行
    invalid_dates = date_objects.isna()
    if invalid_dates.any():
        print(f"删除 {invalid_dates.sum()} 个无效日期行")
        data = data[~invalid_dates].copy()
        date_objects = date_objects[~invalid_dates]
    
    # 3. 彻底清理所有数据列
    # 替换所有空白字符串和不可解析值为NaN
    data = data.replace(r'^\s*$', np.nan, regex=True)
    
    # 4. 分离特征和目标
    # 确保只选择数值列
    numeric_cols = data.select_dtypes(include=['number']).columns
    features = data[numeric_cols.drop(target_col, errors='ignore')]
    target = data[[target_col]] if target_col in numeric_cols else None
    
    if target is None:
        raise ValueError(f"Target column '{target_col}' not found or not numeric")
    
    # 5. 处理缺失值
    print("\n缺失值统计 (清理前):")
    print("特征缺失:", features.isna().sum())
    print("目标缺失:", target.isna().sum())
    
    # 填充缺失值 - 可根据数据特点调整
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    target = target.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print("\n缺失值统计 (清理后):")
    print("特征缺失:", features.isna().sum().sum(), "总缺失")
    print("目标缺失:", target.isna().sum().sum(), "总缺失")
    
    # 6. 数据标准化
    try:
        scaler_features = MinMaxScaler()
        features_scaled = scaler_features.fit_transform(features)
        
        scaler_target = MinMaxScaler()
        target_scaled = scaler_target.fit_transform(target)
    except Exception as e:
        print("\n标准化失败 - 检查数据:")
        print("特征数据类型:", features.dtypes)
        print("目标数据类型:", target.dtypes)
        print("非数值值示例:")
        for col in features.columns:
            non_numeric = pd.to_numeric(features[col], errors='coerce').isna()
            if non_numeric.any():
                print(f"列 '{col}':", features[col][non_numeric].head().tolist())
        raise
    
    # 7. 创建序列
    X, y, date_seq = [], [], []
    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:i+sequence_length, :])
        y.append(target_scaled[i+sequence_length, 0])
        date_seq.append(date_objects.iloc[i+sequence_length])
    
    # 转换为数组
    X = np.array(X)
    y = np.array(y)
    date_seq = np.array(date_seq)
    
    # 8. 分割数据集
    X_train, X_test, y_train, y_test, train_dates, test_dates = train_test_split(
        X, y, date_seq, test_size=test_size, shuffle=False)
    
    # 9. 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    # 10. 返回结果
    scalers = {
        'features': scaler_features,
        'target': scaler_target
    }
    
    return X_train, y_train, X_test, y_test, train_dates, test_dates, scalers

def train_model(model, X_train, y_train, X_test, y_test, 
                num_epochs=100, learning_rate=0.001, batch_size=32):
    """
    Train the LSTM model
    
    Args:
        model: LSTM model instance
        X_train, y_train: training data
        X_test, y_test: testing data
        num_epochs: number of training epochs
        learning_rate: learning rate for optimizer
        batch_size: size of training batches
        
    Returns:
        model: trained model
        train_losses: list of training losses
        test_losses: list of test losses
    """
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create DataLoader
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
    
    train_losses = []
    test_losses = []
    
    # Train the model
    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        # Calculate average loss for the epoch
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_losses.append(test_loss)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    return model, train_losses, test_losses

def predict(model, X, scalers, original_data, target_col, sequence_length):
    """
    Make predictions with the trained model and return with dates
    
    Args:
        model: trained LSTM model
        X: input data (numpy array)
        scalers: dictionary containing feature and target scalers
        original_data: original DataFrame with dates
        target_col: name of target column
        sequence_length: length of input sequences used in training
        
    Returns:
        result_df: DataFrame with dates, actual and predicted values
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        predictions = model(X_tensor).numpy()
    
    # Inverse transform predictions
    predictions = scalers['target'].inverse_transform(predictions).flatten()
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        '日期': original_data['日期'],
        f'实际_{target_col}': original_data[target_col],
        f'预测_{target_col}': np.nan  # Initialize with NaN
    })
    
    # Calculate the correct indices for predictions
    # The first (sequence_length) days don't have predictions
    # Then we have predictions for the remaining days
    pred_indices = range(sequence_length, sequence_length + len(predictions))
    
    # Ensure we don't exceed DataFrame length
    pred_indices = [i for i in pred_indices if i < len(result_df)]
    predictions = predictions[:len(pred_indices)]
    
    # Assign predictions to the correct dates
    result_df.loc[pred_indices, f'预测_{target_col}'] = predictions
    
    return result_df

def plot_results_with_dates(result_df, target_col, title='Actual vs Predicted'):
    """
    Plot actual vs predicted values with dates on x-axis
    
    Args:
        result_df: DataFrame containing dates, actual and predicted values
        target_col: name of the target column
        title: plot title
    """
    plt.figure(figsize=(15, 6))
    plt.plot(result_df['日期'], result_df[f'实际_{target_col}'], label='Actual')
    plt.plot(result_df['日期'], result_df[f'预测_{target_col}'], label='Predicted')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Load your data
    # Replace this with your actual data loading code
    data = pd.read_excel('../data/data_new_output.xlsx', sheet_name='Sheet1')
    
    # Ensure date column is in correct format
    data['日期'] = data['日期'].astype(str)
    
    # Parameters (adjustable)
    params = {
        'target_col': '平均气温',  # Column to predict
        'sequence_length': 30,     # Number of past time steps to use
        'hidden_size': 64,         # Number of features in hidden state
        'num_layers': 2,           # Number of LSTM layers
        'output_size': 1,          # Number of output features
        'dropout': 0.2,            # Dropout rate
        'num_epochs': 20,         # Number of training epochs
        'learning_rate': 0.001,    # Learning rate
        'batch_size': 32,          # Batch size
        'test_size': 0.2           # Proportion of data for testing
    }
    
    # Prepare data (now returns dates as well)
    X_train, y_train, X_test, y_test, train_dates, test_dates, scalers = prepare_data(
        data, 
        params['target_col'], 
        params['sequence_length'], 
        params['test_size']
    )
    print(y_train)
    # Get input size (number of features)
    input_size = X_train.shape[2]
    
    # Initialize model
    model = LSTMModel(
        input_size=input_size,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        output_size=params['output_size'],
        dropout=params['dropout']
    )
    
    # Train model
    print("Training model...")
    model, train_losses, test_losses = train_model(
        model, 
        X_train, y_train, X_test, y_test,
        params['num_epochs'], 
        params['learning_rate'], 
        params['batch_size']
    )
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Make predictions on training set
    train_result = predict(model, X_train, scalers, data, params['target_col'], params['sequence_length'])
    
    # Make predictions on test set
    test_result = predict(model, X_test, scalers, data, params['target_col'], params['sequence_length'])
    
    # Combine results for plotting
    full_result = pd.concat([train_result, test_result])
    full_result = full_result[~full_result.index.duplicated(keep='last')]
    full_result = full_result.sort_values('日期')

    full_result = full_result[full_result.预测_平均气温.notna()]
    print(full_result)
    exit(0)
    
    # Plot results with dates
    plot_results_with_dates(full_result, params['target_col'], 
                           f'{params["target_col"]} - Actual vs Predicted')

if __name__ == '__main__':
    main()