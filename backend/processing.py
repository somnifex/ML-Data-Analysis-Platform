import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
import platform
import os

def configure_matplotlib_chinese_font():
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统常用中文字体
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong']
    elif system == 'Darwin':  # macOS
        chinese_fonts = ['PingFang SC', 'Hiragino Sans GB', 'Heiti SC']
    else:  # 其他系统
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']
    
    available_font = None
    for font in chinese_fonts:
        for f in fontManager.ttflist:
            if font.lower() in f.name.lower():
                available_font = font
                break
        if available_font:
            break
    
    if available_font:
        matplotlib.rcParams['font.family'] = available_font
    else:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    
    matplotlib.rcParams['axes.unicode_minus'] = False

configure_matplotlib_chinese_font()

def preprocess_data(df, x_columns, y_column, test_size=0.2, random_state=42):
    """
    预处理数据，包括缺失值处理、特征编码、标准化、训练测试集分割
    
    参数:
    - df: 原始数据DataFrame
    - x_columns: 特征列名列表
    - y_column: 目标列名
    - test_size: 测试集比例
    - random_state: 随机种子
    
    返回:
    - X_train, X_test, y_train, y_test: 处理后的训练测试数据
    - preprocessing_info: 预处理信息字典
    """
    preprocessing_info = {}
    
    X = df[x_columns].copy()
    y = df[y_column].copy()
    
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    preprocessing_info['numeric_cols'] = numeric_cols
    preprocessing_info['categorical_cols'] = categorical_cols
    
    if X.isnull().sum().sum() > 0:
        if numeric_cols:
            imputer = SimpleImputer(strategy='mean')
            X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        
        if categorical_cols:
            imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = imputer.fit_transform(X[categorical_cols])
    
    encoded_features = {}
    if categorical_cols:
        for col in categorical_cols:
            if X[col].nunique() < 10:
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded = encoder.fit_transform(X[[col]])
                encoded_df = pd.DataFrame(
                    encoded, 
                    columns=[f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                )
                
                X = pd.concat([X, encoded_df], axis=1)
                X.drop(columns=[col], inplace=True)
                
                encoded_features[col] = {
                    'type': 'onehot',
                    'encoder': encoder,
                    'categories': encoder.categories_[0].tolist()
                }
            else:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col])
                encoded_features[col] = {
                    'type': 'label',
                    'encoder': encoder,
                    'classes': encoder.classes_.tolist()
                }
    
    preprocessing_info['encoded_features'] = encoded_features
    
    scaler = StandardScaler()
    current_cols = X.columns.tolist()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=current_cols)
    
    preprocessing_info['scaler'] = scaler
    
    is_classification = False
    if pd.api.types.is_categorical_dtype(y) or y.dtype == 'object' or y.nunique() < 10:
        is_classification = True
        le = LabelEncoder()
        y = le.fit_transform(y)
        preprocessing_info['target_encoder'] = le
        preprocessing_info['target_classes'] = le.classes_.tolist()
    
    preprocessing_info['is_classification'] = is_classification
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, preprocessing_info

def calculate_feature_importance(X_train, y_train, feature_names, model_name):
    """
    计算特征重要性
    
    参数:
    - X_train: 训练特征
    - y_train: 训练标签
    - feature_names: 特征名列表
    - model_name: 使用的模型名称
    
    返回:
    - importance_data: 特征重要性数据
    """
    is_classification = len(np.unique(y_train)) < 10
    
    # 使用随机森林计算特征重要性
    if is_classification:
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    rf_model.fit(X_train, y_train)
    
    # 获取特征重要性
    importances = rf_model.feature_importances_
    
    # 排序特征重要性
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    return {
        'feature_names': sorted_features,
        'importance_values': sorted_importances.tolist(),
        'model_used': 'random_forest'
    }

def evaluate_model(model, X_test, y_test, is_classification):
    """
    评估模型性能
    
    参数:
    - model: 训练好的模型
    - X_test: 测试特征
    - y_test: 测试标签
    - is_classification: 是否是分类任务
    
    返回:
    - metrics: 评估指标字典
    """
    y_pred = model.predict(X_test)
    
    if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    
    metrics = {}
    
    if is_classification:
        metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
        
        if len(np.unique(y_test)) == 2:
            metrics['precision'] = float(precision_score(y_test, y_pred, zero_division=0))
            metrics['recall'] = float(recall_score(y_test, y_pred, zero_division=0))
            metrics['f1'] = float(f1_score(y_test, y_pred, zero_division=0))
            
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
        else:
            metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    else:
        metrics['r2'] = float(r2_score(y_test, y_pred))
        metrics['mse'] = float(mean_squared_error(y_test, y_pred))
        metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
    
    return metrics

def generate_plots(model, X_train, X_test, y_train, y_test, training_history, importance_data, feature_names, is_classification):
    """
    生成评估图表
    
    参数:
    - model: 训练好的模型
    - X_train, X_test, y_train, y_test: 训练和测试数据
    - training_history: 训练历史
    - importance_data: 特征重要性数据
    - feature_names: 特征名称列表
    - is_classification: 是否是分类模型
    
    返回:
    - plots: 以base64编码的图表字典
    """
    configure_matplotlib_chinese_font()
    
    plots = {}
    
    if importance_data:
        plt.figure(figsize=(10, 6))
        plt.barh(
            importance_data['feature_names'][:10],  # 只显示前10个特征
            importance_data['importance_values'][:10]
        )
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.title('特征重要性')
        plt.tight_layout()
        plots['feature_importance'] = fig_to_base64(plt.gcf())
        plt.close()
    
    if 'train_loss' in training_history and training_history['train_loss']:
        plt.figure(figsize=(10, 6))
        plt.plot(training_history['train_loss'], label='训练损失')
        if 'val_loss' in training_history:
            plt.plot(training_history['val_loss'], label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('学习曲线')
        plt.legend()
        plt.tight_layout()
        plots['learning_curve'] = fig_to_base64(plt.gcf())
        plt.close()
        
        if is_classification and 'train_acc' in training_history and training_history['train_acc']:
            plt.figure(figsize=(10, 6))
            plt.plot(training_history['train_acc'], label='训练准确率')
            if 'val_acc' in training_history:
                plt.plot(training_history['val_acc'], label='验证准确率')
            plt.xlabel('轮次')
            plt.ylabel('准确率')
            plt.title('准确率曲线')
            plt.legend()
            plt.tight_layout()
            plots['accuracy_curve'] = fig_to_base64(plt.gcf())
            plt.close()
    
    y_pred = model.predict(X_test)
    
    if is_classification:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plots['confusion_matrix'] = fig_to_base64(plt.gcf())
        plt.close()
        
        if len(np.unique(y_test)) == 2 and hasattr(model, 'net') and hasattr(model.net, 'forward'):
            try:
                X_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test)
                model.net.eval()
                with torch.no_grad():
                    outputs = model.net(X_tensor)
                    probas = torch.softmax(outputs, 1).numpy()[:, 1]
                
                fpr, tpr, _ = roc_curve(y_test, probas)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('假阳性率')
                plt.ylabel('真阳性率')
                plt.title('ROC曲线')
                plt.legend(loc='lower right')
                plt.tight_layout()
                plots['roc_curve'] = fig_to_base64(plt.gcf())
                plt.close()
            except:
                pass
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测 vs 真实值')
        plt.tight_layout()
        plots['prediction_vs_actual'] = fig_to_base64(plt.gcf())
        plt.close()
    
    return plots

def fig_to_base64(fig):
    """
    将matplotlib图表转换为base64编码的字符串
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str