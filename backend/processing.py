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
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import torch
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
import platform
import os


def configure_matplotlib_chinese_font():
    system = platform.system()

    if system == "Windows":
        chinese_fonts = ["Microsoft YaHei", "SimHei", "SimSun", "NSimSun", "FangSong"]
    elif system == "Darwin":
        chinese_fonts = ["PingFang SC", "Hiragino Sans GB", "Heiti SC"]
    else:
        chinese_fonts = [
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "Droid Sans Fallback",
        ]

    available_font = None
    for font in chinese_fonts:
        for f in fontManager.ttflist:
            if font.lower() in f.name.lower():
                available_font = font
                break
        if available_font:
            break

    if available_font:
        matplotlib.rcParams["font.family"] = available_font
    else:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [
            "DejaVu Sans",
            "Arial Unicode MS",
            "sans-serif",
        ]

    matplotlib.rcParams["axes.unicode_minus"] = False


configure_matplotlib_chinese_font()


def preprocess_data(df, x_columns, y_column, test_size=0.2, random_state=42):
    """
    预处理数据：处理缺失值、特征编码、标准化和训练测试集分割
    """
    preprocessing_info = {}

    X = df[x_columns].copy()
    y = df[y_column].copy()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessing_info["numeric_cols"] = numeric_cols
    preprocessing_info["categorical_cols"] = categorical_cols

    if X.isnull().sum().sum() > 0:
        if numeric_cols:
            imputer = SimpleImputer(strategy="mean")
            X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

        if categorical_cols:
            imputer = SimpleImputer(strategy="most_frequent")
            X[categorical_cols] = imputer.fit_transform(X[categorical_cols])

    encoded_features = {}
    if categorical_cols:
        for col in categorical_cols:
            if X[col].nunique() < 10:
                encoder = OneHotEncoder(
                    sparse_output=False, drop="first", handle_unknown="ignore"
                )
                encoded = encoder.fit_transform(X[[col]])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{col}_{cat}" for cat in encoder.categories_[0][1:]],
                )

                X = pd.concat([X, encoded_df], axis=1)
                X.drop(columns=[col], inplace=True)

                encoded_features[col] = {
                    "type": "onehot",
                    "encoder": encoder,
                    "categories": encoder.categories_[0].tolist(),
                }
            else:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col])
                encoded_features[col] = {
                    "type": "label",
                    "encoder": encoder,
                    "classes": encoder.classes_.tolist(),
                }

    preprocessing_info["encoded_features"] = encoded_features

    scaler = StandardScaler()
    current_cols = X.columns.tolist()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=current_cols)

    preprocessing_info["scaler"] = scaler

    is_classification = False
    if pd.api.types.is_categorical_dtype(y) or y.dtype == "object" or y.nunique() < 10:
        is_classification = True
        le = LabelEncoder()
        y = le.fit_transform(y)
        preprocessing_info["target_encoder"] = le
        preprocessing_info["target_classes"] = le.classes_.tolist()

    preprocessing_info["is_classification"] = is_classification

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if is_classification and len(np.unique(y)) > 1 else None,
    )

    preprocessing_info["feature_names_after_preprocessing"] = X_train.columns.tolist()

    return X_train, X_test, y_train, y_test, preprocessing_info


def apply_saved_preprocessing(
    df_new, x_columns_new, y_column_new, saved_preprocessing_info
):
    """
    对新数据应用已保存的预处理转换。
    """
    X_new = df_new[x_columns_new].copy()
    y_new = (
        df_new[y_column_new].copy()
        if y_column_new and y_column_new in df_new.columns
        else None
    )

    numeric_cols_original = saved_preprocessing_info.get("numeric_cols", [])
    categorical_cols_original = saved_preprocessing_info.get("categorical_cols", [])

    current_numeric_cols = [
        col for col in numeric_cols_original if col in X_new.columns
    ]
    current_categorical_cols = [
        col for col in categorical_cols_original if col in X_new.columns
    ]

    if X_new.isnull().sum().sum() > 0:
        if current_numeric_cols:
            num_imputer = saved_preprocessing_info.get("numeric_imputer")
            if num_imputer:
                X_new[current_numeric_cols] = num_imputer.transform(
                    X_new[current_numeric_cols]
                )

        if current_categorical_cols:
            cat_imputer = saved_preprocessing_info.get("categorical_imputer")
            if cat_imputer:
                X_new[current_categorical_cols] = cat_imputer.transform(
                    X_new[current_categorical_cols]
                )

    encoded_features_info = saved_preprocessing_info.get("encoded_features", {})
    temp_encoded_dfs = []

    processed_categorical_cols = []

    for col in current_categorical_cols:
        if col in encoded_features_info:
            encoder_info = encoded_features_info[col]
            encoder = encoder_info["encoder"]

            if encoder_info["type"] == "onehot":
                encoded_data = encoder.transform(X_new[[col]])
                feature_names_onehot = [
                    f"{col}_{cat}"
                    for cat in encoder.categories_[0][1:]
                    if cat in encoder.get_feature_names_out([col])
                ]
                output_cols = encoder.get_feature_names_out([col])

                encoded_df = pd.DataFrame(
                    encoded_data, columns=output_cols, index=X_new.index
                )
                temp_encoded_dfs.append(encoded_df)
                processed_categorical_cols.append(col)
            elif encoder_info["type"] == "label":
                X_new[col] = encoder.transform(X_new[col])

    if temp_encoded_dfs:
        X_new = pd.concat([X_new] + temp_encoded_dfs, axis=1)

    X_new.drop(
        columns=[
            col
            for col in processed_categorical_cols
            if col in X_new.columns
            and col in encoded_features_info
            and encoded_features_info[col]["type"] == "onehot"
        ],
        inplace=True,
    )

    expected_cols_before_scaling = saved_preprocessing_info.get(
        "feature_names_after_preprocessing", []
    )

    for col in expected_cols_before_scaling:
        if col not in X_new.columns:
            X_new[col] = 0

    X_new = X_new[expected_cols_before_scaling]

    scaler = saved_preprocessing_info.get("scaler")
    if scaler:
        X_scaled = scaler.transform(X_new)
        X_processed = pd.DataFrame(X_scaled, columns=X_new.columns, index=X_new.index)
    else:
        X_processed = X_new

    y_processed = None
    is_classification = saved_preprocessing_info.get("is_classification", False)
    if y_new is not None:
        if is_classification:
            target_encoder = saved_preprocessing_info.get("target_encoder")
            if target_encoder:
                try:
                    y_processed = target_encoder.transform(y_new)
                except ValueError as e:
                    unknown_labels = set(y_new) - set(target_encoder.classes_)
                    if unknown_labels:
                        raise ValueError(
                            f"目标列包含未知标签: {unknown_labels}. 模型无法处理。"
                        )
                    else:
                        raise e
            else:
                y_processed = y_new.values
        else:
            y_processed = y_new.values

    return X_processed, y_processed, is_classification


def calculate_feature_importance(X_train, y_train, feature_names, model_name):
    """
    计算特征重要性并生成多种可视化图表
    """
    is_classification = len(np.unique(y_train)) < 10

    if is_classification:
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)

    rf_model.fit(X_train, y_train)

    importances = rf_model.feature_importances_
    
    # 计算标准差，用于误差条显示
    if hasattr(rf_model, 'estimators_'):
        std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
    else:
        std = np.zeros_like(importances)
    
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    sorted_std = std[indices]
    
    # 生成多种特征重要性可视化图
    visualizations = {}
    
    # 1. 水平条形图 - 增强版本
    plt.figure(figsize=(10, 8))
    top_n = min(15, len(sorted_features))
    plt.barh(range(top_n), sorted_importances[:top_n], align='center', 
             color='skyblue', edgecolor='navy', alpha=0.8,
             xerr=sorted_std[:top_n], capsize=5)
    plt.yticks(range(top_n), sorted_features[:top_n])
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title('Feature Importance Ranking')
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    visualizations['horizontal_bar'] = fig_to_base64(plt.gcf())
    plt.close()
    
    # 2. 垂直条形图 - 带颜色渐变
    plt.figure(figsize=(12, 8))
    top_n = min(15, len(sorted_features))
    bars = plt.bar(range(top_n), sorted_importances[:top_n], 
           yerr=sorted_std[:top_n], align='center', alpha=0.8,
           color=plt.cm.viridis(np.linspace(0, 1, top_n)))
    plt.xticks(range(top_n), sorted_features[:top_n], rotation=45, ha='right')
    plt.ylabel('Importance Score')
    plt.xlabel('Features')
    plt.title('Top Features by Importance')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    visualizations['vertical_bar'] = fig_to_base64(plt.gcf())
    plt.close()
    
    # 3. 饼图 - 显示最重要的特征比例
    plt.figure(figsize=(10, 10))
    top_n = min(8, len(sorted_features))
    # 确保重要性值为正
    pie_values = np.maximum(0, sorted_importances[:top_n])
    # 添加其他类别
    if len(sorted_features) > top_n:
        pie_values = np.append(pie_values, sum(sorted_importances[top_n:]))
        labels = sorted_features[:top_n] + ['Other Features']
    else:
        labels = sorted_features[:top_n]
    
    plt.pie(pie_values, labels=labels, autopct='%1.1f%%', 
            shadow=True, startangle=90, 
            colors=plt.cm.tab10(np.linspace(0, 1, len(pie_values))))
    plt.axis('equal')  # 保持饼图为圆形
    plt.title('Feature Importance Distribution')
    plt.tight_layout()
    visualizations['pie'] = fig_to_base64(plt.gcf())
    plt.close()
    
    # 4. 热图 - 重要特征之间的相关性矩阵
    if len(sorted_features) > 2:
        plt.figure(figsize=(10, 8))
        top_n = min(12, len(sorted_features))
        top_features = [feature_names[i] for i in indices[:top_n]]
        X_top = X_train[top_features]
        
        # 计算相关矩阵
        corr = X_top.corr()
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
                    annot=True, fmt='.2f', square=True, linewidths=.5)
        plt.title('Correlation Matrix of Top Features')
        plt.tight_layout()
        visualizations['correlation'] = fig_to_base64(plt.gcf())
        plt.close()
    
    # 5. 累积重要性图
    plt.figure(figsize=(10, 6))
    cumulative_importance = np.cumsum(sorted_importances)
    plt.plot(range(1, len(sorted_features) + 1), cumulative_importance, 
             marker='o', linestyle='-', color='#1f77b4', markersize=5)
    
    # 添加90%和95%重要性的参考线
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Importance')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% Importance')
    
    # 找到达到90%和95%重要性所需的特征数量
    features_90 = np.where(cumulative_importance >= 0.9)[0][0] + 1
    features_95 = np.where(cumulative_importance >= 0.95)[0][0] + 1
    
    plt.axvline(x=features_90, color='r', linestyle=':', alpha=0.7)
    plt.axvline(x=features_95, color='g', linestyle=':', alpha=0.7)
    
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title(f'Cumulative Feature Importance\n(90% at {features_90} features, 95% at {features_95} features)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    visualizations['cumulative'] = fig_to_base64(plt.gcf())
    plt.close()

    return {
        "feature_names": sorted_features,
        "importance_values": sorted_importances.tolist(),
        "std_values": sorted_std.tolist(),
        "model_used": "random_forest",
        "visualizations": visualizations
    }


def evaluate_model(model, X_test, y_test, is_classification):
    """
    评估模型性能
    """
    y_pred = model.predict(X_test)

    if hasattr(y_pred, "shape") and len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    metrics = {}

    if is_classification:
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        
        classes = np.unique(y_test)
        n_classes = len(classes)
        
        # 检测类别不平衡
        class_counts = np.bincount(y_test.astype(int))
        class_distribution = class_counts / class_counts.sum()
        metrics["class_distribution"] = class_distribution.tolist()
        
        # 检查是否严重不平衡（任一类别占比低于10%）
        imbalance_detected = any(dist < 0.1 for dist in class_distribution)
        metrics["class_imbalance_detected"] = imbalance_detected
        
        if n_classes == 2:
            # 二分类指标 - 使用多种平均方式
            metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
            metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
            metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
            
            # 为每个类别添加单独的指标
            metrics["per_class_metrics"] = {}
            for i, class_label in enumerate(classes):
                metrics["per_class_metrics"][str(class_label)] = {
                    "precision": float(precision_score(y_test, y_pred, pos_label=class_label, average="binary", zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, pos_label=class_label, average="binary", zero_division=0)),
                    "f1": float(f1_score(y_test, y_pred, pos_label=class_label, average="binary", zero_division=0)),
                    "support": int(np.sum(y_test == class_label))
                }

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics["confusion_matrix"] = cm.tolist()
            
            # 添加困惑矩阵衍生指标
            metrics["balanced_accuracy"] = float((tp/(tp+fn) + tn/(tn+fp))/2) if (tp+fn)*(tn+fp) > 0 else 0.0
            metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

            # 计算 ROC 曲线相关指标
            y_scores = None
            
            if hasattr(model, "predict_proba"):
                try:
                    y_scores = model.predict_proba(X_test)[:, 1]
                except Exception as e:
                    print(f"使用predict_proba计算分数时出错: {e}")
            
            elif hasattr(model, "model") and hasattr(model.model, "predict_proba"):
                try:
                    y_scores = model.model.predict_proba(X_test)[:, 1]
                except Exception as e:
                    print(f"使用model.predict_proba计算分数时出错: {e}")
            
            elif hasattr(model, "decision_function"):
                try:
                    y_scores = model.decision_function(X_test)
                    if y_scores.ndim > 1:
                        y_scores = y_scores[:, 1]
                except Exception as e:
                    print(f"使用decision_function计算分数时出错: {e}")
            
            elif hasattr(model, "model") and hasattr(model.model, "decision_function"):
                try:
                    y_scores = model.model.decision_function(X_test)
                    if y_scores.ndim > 1:
                        y_scores = y_scores[:, 1]
                except Exception as e:
                    print(f"使用model.decision_function计算分数时出错: {e}")
            
            elif hasattr(model, "net") and hasattr(model, "device"):
                try:
                    X_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, "values") else X_test).to(model.device)
                    model.net.eval()
                    with torch.no_grad():
                        outputs = model.net(X_tensor)
                        if outputs.shape[1] >= 2:
                            y_scores = torch.softmax(outputs, 1).cpu().numpy()[:, 1]
                        else:
                            y_scores = torch.sigmoid(outputs).cpu().numpy().flatten()
                except Exception as e:
                    print(f"使用PyTorch模型计算分数时出错: {e}")
            
            if y_scores is None:
                print(f"警告：模型{type(model).__name__}不支持predict_proba或decision_function，将使用预测标签作为分数")
                try:
                    y_scores = np.array(y_pred, dtype=float)
                except:
                    y_scores = np.array([1.0 if pred else 0.0 for pred in y_pred])
            
            try:
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                metrics["auc"] = float(auc(fpr, tpr))

                metrics["roc_curve_data"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist()
                }

                precision, recall, _ = precision_recall_curve(y_test, y_scores)
                metrics["average_precision"] = float(average_precision_score(y_test, y_scores))
                metrics["pr_curve_data"] = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist()
                }
            except Exception as e:
                print(f"计算ROC/AUC时出错: {e}")
                metrics["auc"] = 0.5
                    
        else:
            # 多分类指标 - 使用多种平均方式
            metrics["precision_weighted"] = float(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            metrics["precision_macro"] = float(
                precision_score(y_test, y_pred, average="macro", zero_division=0)
            )
            metrics["precision_micro"] = float(
                precision_score(y_test, y_pred, average="micro", zero_division=0)
            )
            
            metrics["recall_weighted"] = float(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            metrics["recall_macro"] = float(
                recall_score(y_test, y_pred, average="macro", zero_division=0)
            )
            metrics["recall_micro"] = float(
                recall_score(y_test, y_pred, average="micro", zero_division=0)
            )
            
            metrics["f1_weighted"] = float(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            metrics["f1_macro"] = float(
                f1_score(y_test, y_pred, average="macro", zero_division=0)
            )
            metrics["f1_micro"] = float(
                f1_score(y_test, y_pred, average="micro", zero_division=0)
            )
            
            # 兼容旧版本API
            metrics["precision"] = metrics["precision_weighted"]
            metrics["recall"] = metrics["recall_weighted"] 
            metrics["f1"] = metrics["f1_weighted"]
            
            # 为每个类别添加单独的指标
            metrics["per_class_metrics"] = {}
            for i, class_label in enumerate(classes):
                y_true_binary = (y_test == class_label).astype(int)
                y_pred_binary = (y_pred == class_label).astype(int)
                metrics["per_class_metrics"][str(class_label)] = {
                    "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
                    "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
                    "f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
                    "support": int(np.sum(y_test == class_label))
                }
            
            # 计算多分类ROC曲线和AUC
            y_test_bin = label_binarize(y_test, classes=classes)
            
            all_fpr = {}
            all_tpr = {}
            all_auc = {}
            
            y_scores = None
            
            if hasattr(model, "predict_proba"):
                try:
                    y_scores = model.predict_proba(X_test)
                except Exception as e:
                    print(f"获取多分类概率分数时出错: {e}")
            
            elif hasattr(model, "decision_function"):
                try:
                    decision_scores = model.decision_function(X_test)
                    if decision_scores.ndim > 1:
                        y_scores = decision_scores
                    else:
                        y_scores = np.zeros((len(y_test), n_classes))
                        for i, score in enumerate(decision_scores):
                            y_scores[i, 1] = score
                except Exception as e:
                    print(f"使用decision_function获取多分类分数时出错: {e}")
            
            elif hasattr(model, "net") and hasattr(model, "device"):
                try:
                    X_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, "values") else X_test).to(model.device)
                    model.net.eval()
                    with torch.no_grad():
                        outputs = model.net(X_tensor)
                        y_scores = torch.softmax(outputs, 1).cpu().numpy()
                except Exception as e:
                    print(f"使用PyTorch模型获取多分类分数时出错: {e}")
            
            if y_scores is not None:
                try:
                    try:
                        metrics["auc"] = float(roc_auc_score(y_test_bin, y_scores, average="macro", multi_class="ovr"))
                    except ValueError:
                        try:
                            metrics["auc"] = float(roc_auc_score(y_test, y_scores, average="weighted", multi_class="ovr"))
                        except:
                            metrics["auc"] = float(roc_auc_score(y_test_bin, y_scores, average="macro", multi_class="ovr", labels=range(y_scores.shape[1])))
                    
                    metrics["multi_class_roc_data"] = {}
                    for i, class_name in enumerate(classes):
                        if i < y_scores.shape[1]:
                            fpr, tpr, _ = roc_curve(
                                y_test_bin[:, i] if y_test_bin.shape[1] > 1 else (y_test == class_name).astype(int), 
                                y_scores[:, i]
                            )
                            roc_auc = auc(fpr, tpr)
                            
                            all_fpr[str(class_name)] = fpr.tolist()
                            all_tpr[str(class_name)] = tpr.tolist()
                            all_auc[str(class_name)] = float(roc_auc)
                    
                    metrics["multi_class_roc_data"] = {
                        "fpr": all_fpr,
                        "tpr": all_tpr,
                        "auc": all_auc,
                        "classes": [str(c) for c in classes]
                    }
                except Exception as e:
                    print(f"计算多分类ROC曲线时出错: {e}")
    else:
        # 保存原有回归指标
        metrics["r2"] = float(r2_score(y_test, y_pred))
        metrics["mse"] = float(mean_squared_error(y_test, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        
        # 为回归模型添加基于多阈值的分类指标
        thresholds = {
            "median": np.median(y_test),
            "mean": np.mean(y_test),
            "q1": np.percentile(y_test, 25),
            "q3": np.percentile(y_test, 75)
        }
        
        metrics["binary_classification_metrics"] = {}
        
        for threshold_name, threshold in thresholds.items():
            # 转换真实值和预测值为二分类
            y_test_binary = (y_test > threshold).astype(int)
            y_pred_binary = (y_pred > threshold).astype(int)
            
            # 计算分类指标
            threshold_metrics = {
                "threshold_value": float(threshold),
                "accuracy": float(accuracy_score(y_test_binary, y_pred_binary)),
                "precision": float(precision_score(y_test_binary, y_pred_binary, zero_division=0)),
                "recall": float(recall_score(y_test_binary, y_pred_binary, zero_division=0)),
                "f1": float(f1_score(y_test_binary, y_pred_binary, zero_division=0)),
                "support_above_threshold": int(np.sum(y_test_binary)),
                "support_below_threshold": int(np.sum(1 - y_test_binary))
            }
            
            metrics["binary_classification_metrics"][threshold_name] = threshold_metrics
        
        # 使用中位数阈值作为默认分类指标（保持向后兼容）
        metrics["accuracy"] = metrics["binary_classification_metrics"]["median"]["accuracy"]
        metrics["precision"] = metrics["binary_classification_metrics"]["median"]["precision"]
        metrics["recall"] = metrics["binary_classification_metrics"]["median"]["recall"]
        metrics["f1"] = metrics["binary_classification_metrics"]["median"]["f1"]
        metrics["threshold_used"] = float(thresholds["median"])
        
        # 计算混淆矩阵以获取特异性
        cm = confusion_matrix(y_test_binary, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["confusion_matrix"] = cm.tolist()
        
        # 添加阈值信息
        metrics["threshold_used"] = float(threshold)
        
        # 计算ROC曲线和AUC
        try:
            # 对连续值预测使用ROC
            fpr, tpr, _ = roc_curve(y_test_binary, y_pred)
            metrics["auc"] = float(auc(fpr, tpr))
            
            metrics["roc_curve_data"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            }
            
            # 精确率-召回率曲线
            precision, recall, _ = precision_recall_curve(y_test_binary, y_pred)
            metrics["average_precision"] = float(average_precision_score(y_test_binary, y_pred))
            metrics["pr_curve_data"] = {
                "precision": precision.tolist(),
                "recall": recall.tolist()
            }
        except Exception as e:
            print(f"计算回归模型ROC/AUC时出错: {e}")
            metrics["auc"] = 0.5

    return metrics


def generate_plots(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    training_history,
    importance_data,
    feature_names,
    is_classification,
    preprocessing_info=None,
):
    """
    生成评估图表
    """
    configure_matplotlib_chinese_font()

    plots = {}

    if importance_data:
        # 添加特征重要性可视化图表
        if "visualizations" in importance_data:
            # 直接从原始数据中添加可视化
            for viz_name, viz_data in importance_data["visualizations"].items():
                plots[f"feature_importance_{viz_name}"] = viz_data
                
        # 生成标准的特征重要性图
        plt.figure(figsize=(10, 6))
        num_features = min(10, len(importance_data["feature_names"]))
        plt.barh(
            importance_data["feature_names"][:num_features],
            importance_data["importance_values"][:num_features],
        )
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("Feature Importance")
        plt.tight_layout()
        plots["feature_importance"] = fig_to_base64(plt.gcf())
        plt.close()

    if "train_loss" in training_history and training_history["train_loss"]:
        plt.figure(figsize=(10, 6))
        plt.plot(training_history["train_loss"], label="Training Loss")
        if "val_loss" in training_history:
            plt.plot(training_history["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plots["learning_curve"] = fig_to_base64(plt.gcf())
        plt.close()

        if (
            is_classification
            and "train_acc" in training_history
            and training_history["train_acc"]
        ):
            plt.figure(figsize=(10, 6))
            plt.plot(training_history["train_acc"], label="Training Accuracy")
            if "val_acc" in training_history:
                plt.plot(training_history["val_acc"], label="Validation Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title("Accuracy Curve")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plots["accuracy_curve"] = fig_to_base64(plt.gcf())
            plt.close()

    y_pred = model.predict(X_test)

    if is_classification:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        tick_labels = None
        if preprocessing_info and "target_encoder" in preprocessing_info:
            target_encoder = preprocessing_info["target_encoder"]
            if hasattr(target_encoder, "classes_"):
                tick_labels = target_encoder.classes_

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=tick_labels if tick_labels is not None else "auto",
            yticklabels=tick_labels if tick_labels is not None else "auto",
            cbar=True,
            linewidths=0.5,
            linecolor='gray',
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plots["confusion_matrix"] = fig_to_base64(plt.gcf())
        plt.close()

        classes = np.unique(y_test)
        if len(classes) > 2:
            metrics_data = evaluate_model(model, X_test, y_test, is_classification)
            if "multi_class_roc_data" in metrics_data:
                roc_data = metrics_data["multi_class_roc_data"]
                
                plt.figure(figsize=(10, 8))
                colors = plt.cm.get_cmap('tab10', len(classes))
                
                for i, class_name in enumerate(roc_data["classes"]):
                    if class_name in roc_data["fpr"] and class_name in roc_data["tpr"] and class_name in roc_data["auc"]:
                        fpr = roc_data["fpr"][class_name]
                        tpr = roc_data["tpr"][class_name]
                        roc_auc = roc_data["auc"][class_name]
                        
                        display_name = class_name
                        if preprocessing_info and "target_encoder" in preprocessing_info:
                            try:
                                idx = int(class_name)
                                if idx < len(preprocessing_info["target_encoder"].classes_):
                                    display_name = str(preprocessing_info["target_encoder"].classes_[idx])
                            except (ValueError, IndexError):
                                pass
                        
                        plt.plot(
                            fpr, tpr, 
                            color=colors(i), lw=2,
                            label=f'Class {display_name} (AUC = {roc_auc:.3f})'
                        )
                
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Multi-class ROC Curves (One-vs-Rest)')
                plt.legend(loc="lower right", fontsize=9)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plots["roc_curve"] = fig_to_base64(plt.gcf())
                plt.close()

        elif len(np.unique(y_test)) == 2:
            y_probas = None
            if hasattr(model, "predict_proba"):
                y_probas = model.predict_proba(X_test)[:, 1]
            elif (
                hasattr(model, "net")
                and hasattr(model, "device")
                and hasattr(model, "is_classification")
                and model.is_classification
            ):
                X_tensor = torch.FloatTensor(
                    X_test.values if hasattr(X_test, "values") else X_test
                ).to(model.device)
                model.net.eval()
                with torch.no_grad():
                    outputs = model.net(X_tensor)
                    if outputs.shape[1] >= 2:
                        y_probas = torch.softmax(outputs, 1).cpu().numpy()[:, 1]
                    else:
                        y_probas = torch.sigmoid(outputs).cpu().numpy().flatten()

            if y_probas is not None:
                try:
                    fpr, tpr, _ = roc_curve(y_test, y_probas)
                    roc_auc = auc(fpr, tpr)

                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", color='#1f77b4', linewidth=2)
                    plt.plot([0, 1], [0, 1], "k--", label="Random Guess", alpha=0.8)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel("False Positive Rate (1-Specificity)")
                    plt.ylabel("True Positive Rate (Sensitivity/Recall)")
                    plt.title("Receiver Operating Characteristic (ROC) Curve")
                    plt.legend(loc="lower right")
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plots["roc_curve"] = fig_to_base64(plt.gcf())
                    plt.close()

                    precision, recall, _ = precision_recall_curve(y_test, y_probas)
                    avg_precision = average_precision_score(y_test, y_probas)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(recall, precision, label=f"PR Curve (AP = {avg_precision:.3f})", color='#ff7f0e', linewidth=2)
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title("Precision-Recall Curve")
                    plt.legend(loc="lower left")
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plots["pr_curve"] = fig_to_base64(plt.gcf())
                    plt.close()
                    
                except Exception as e:
                    print(f"生成ROC曲线时出错: {e}")
                    pass
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs Actual Values")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plots["prediction_vs_actual"] = fig_to_base64(plt.gcf())
        plt.close()

        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plots["residuals_plot"] = fig_to_base64(plt.gcf())
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Residuals Distribution")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plots["residuals_hist"] = fig_to_base64(plt.gcf())
        plt.close()
        
    return plots


def fig_to_base64(fig):
    """
    将matplotlib图表转换为base64编码的字符串
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    return img_str
