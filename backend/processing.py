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
    elif system == "Darwin":  # macOS
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
            if X[col].nunique() < 10:  # 对类别数较少的分类特征进行独热编码
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
    计算特征重要性
    """
    is_classification = len(np.unique(y_train)) < 10

    if is_classification:
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)

    rf_model.fit(X_train, y_train)

    importances = rf_model.feature_importances_

    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    return {
        "feature_names": sorted_features,
        "importance_values": sorted_importances.tolist(),
        "model_used": "random_forest",
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

        if n_classes == 2:  # 二分类情况
            metrics["precision"] = float(
                precision_score(y_test, y_pred, zero_division=0)
            )
            metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
            metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))

            # 计算混淆矩阵以获取特异性
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics["confusion_matrix"] = cm.tolist()

            # 尝试多种方法获取分数用于ROC计算
            y_scores = None
            
            # 方法1: 使用predict_proba (大多数scikit-learn模型)
            if hasattr(model, "predict_proba"):
                try:
                    y_scores = model.predict_proba(X_test)[:, 1]
                except Exception as e:
                    print(f"使用predict_proba计算分数时出错: {e}")
            
            # 处理sklearn包装模型
            elif hasattr(model, "model") and hasattr(model.model, "predict_proba"):
                try:
                    y_scores = model.model.predict_proba(X_test)[:, 1]
                except Exception as e:
                    print(f"使用model.predict_proba计算分数时出错: {e}")
            
            # 方法2: 使用decision_function (某些线性模型如SVM)
            elif hasattr(model, "decision_function"):
                try:
                    y_scores = model.decision_function(X_test)
                    # 如果返回的是多类决策分数，选择正类的分数
                    if y_scores.ndim > 1:
                        y_scores = y_scores[:, 1]
                except Exception as e:
                    print(f"使用decision_function计算分数时出错: {e}")
            
            # 处理sklearn包装模型
            elif hasattr(model, "model") and hasattr(model.model, "decision_function"):
                try:
                    y_scores = model.model.decision_function(X_test)
                    # 如果返回的是多类决策分数，选择正类的分数
                    if y_scores.ndim > 1:
                        y_scores = y_scores[:, 1]
                except Exception as e:
                    print(f"使用model.decision_function计算分数时出错: {e}")
            
            # 方法3: PyTorch模型
            elif hasattr(model, "net") and hasattr(model, "device"):
                try:
                    X_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, "values") else X_test).to(model.device)
                    model.net.eval()
                    with torch.no_grad():
                        outputs = model.net(X_tensor)
                        if outputs.shape[1] >= 2:  # 多分类
                            y_scores = torch.softmax(outputs, 1).cpu().numpy()[:, 1]
                        else:  # 二分类
                            y_scores = torch.sigmoid(outputs).cpu().numpy().flatten()
                except Exception as e:
                    print(f"使用PyTorch模型计算分数时出错: {e}")
            
            # 方法4: 对于不支持上述方法的模型，使用预测结果作为代理分数
            if y_scores is None:
                print(f"警告：模型{type(model).__name__}不支持predict_proba或decision_function，将使用预测标签作为分数")
                # 将类别标签转换为连续分数 (0或1) - 这是一个近似方法
                try:
                    y_scores = np.array(y_pred, dtype=float)
                except:
                    y_scores = np.array([1.0 if pred else 0.0 for pred in y_pred])
            
            # 如果成功获取分数，计算ROC/AUC
            try:
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                metrics["auc"] = float(auc(fpr, tpr))

                # 保存ROC曲线数据点
                metrics["roc_curve_data"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist()
                }

                # 精确率-召回率曲线
                precision, recall, _ = precision_recall_curve(y_test, y_scores)
                metrics["average_precision"] = float(average_precision_score(y_test, y_scores))
                metrics["pr_curve_data"] = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist()
                }
            except Exception as e:
                print(f"计算ROC/AUC时出错: {e}")
                metrics["auc"] = 0.5  # 默认值 (随机猜测)
                    
        else:  # 多分类情况
            metrics["precision"] = float(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            metrics["f1"] = float(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            
            # 多分类ROC曲线和AUC计算 (一对多方式)
            y_test_bin = label_binarize(y_test, classes=classes)
            
            # 为存储每个类的ROC曲线数据创建容器
            all_fpr = {}
            all_tpr = {}
            all_auc = {}
            
            # 尝试获取多分类的分数
            y_scores = None
            
            # 方法1: 使用predict_proba
            if hasattr(model, "predict_proba"):
                try:
                    y_scores = model.predict_proba(X_test)
                except Exception as e:
                    print(f"获取多分类概率分数时出错: {e}")
            
            # 方法2: 使用decision_function
            elif hasattr(model, "decision_function"):
                try:
                    decision_scores = model.decision_function(X_test)
                    # 如果是二维数组，可以直接使用
                    if decision_scores.ndim > 1:
                        y_scores = decision_scores
                    # 如果是一维数组，需要转换为OvR格式
                    else:
                        # 创建n_classes维度的零矩阵
                        y_scores = np.zeros((len(y_test), n_classes))
                        for i, score in enumerate(decision_scores):
                            y_scores[i, 1] = score  # 假设score是正类的分数
                except Exception as e:
                    print(f"使用decision_function获取多分类分数时出错: {e}")
            
            # 方法3: PyTorch模型
            elif hasattr(model, "net") and hasattr(model, "device"):
                try:
                    X_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, "values") else X_test).to(model.device)
                    model.net.eval()
                    with torch.no_grad():
                        outputs = model.net(X_tensor)
                        y_scores = torch.softmax(outputs, 1).cpu().numpy()
                except Exception as e:
                    print(f"使用PyTorch模型获取多分类分数时出错: {e}")
            
            # 如果成功获取到分数，计算多分类AUC
            if y_scores is not None:
                try:
                    # 计算宏平均ROC AUC
                    try:
                        metrics["auc"] = float(roc_auc_score(y_test_bin, y_scores, average="macro", multi_class="ovr"))
                    except ValueError:
                        # 如果有些类别在测试集中没有样本，可能会出错
                        try:
                            metrics["auc"] = float(roc_auc_score(y_test, y_scores, average="weighted", multi_class="ovr"))
                        except:
                            metrics["auc"] = float(roc_auc_score(y_test_bin, y_scores, average="macro", multi_class="ovr", labels=range(y_scores.shape[1])))
                    
                    # 计算每个类别的ROC曲线
                    metrics["multi_class_roc_data"] = {}
                    for i, class_name in enumerate(classes):
                        if i < y_scores.shape[1]:  # 确保类索引在y_scores的范围内
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
        metrics["r2"] = float(r2_score(y_test, y_pred))
        metrics["mse"] = float(mean_squared_error(y_test, y_pred))
        # 添加均方根误差(RMSE)
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))

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
        plt.figure(figsize=(10, 6))
        plt.barh(
            importance_data["feature_names"][:10],  # 只显示前10个特征
            importance_data["importance_values"][:10],
        )
        plt.xlabel("重要性")
        plt.ylabel("特征")
        plt.title("特征重要性")
        plt.tight_layout()
        plots["feature_importance"] = fig_to_base64(plt.gcf())
        plt.close()

    if "train_loss" in training_history and training_history["train_loss"]:
        plt.figure(figsize=(10, 6))
        plt.plot(training_history["train_loss"], label="训练损失")
        if "val_loss" in training_history:
            plt.plot(training_history["val_loss"], label="验证损失")
        plt.xlabel("轮次")
        plt.ylabel("损失")
        plt.title("学习曲线")
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
            plt.plot(training_history["train_acc"], label="训练准确率")
            if "val_acc" in training_history:
                plt.plot(training_history["val_acc"], label="验证准确率")
            plt.xlabel("轮次")
            plt.ylabel("准确率")
            plt.title("准确率曲线")
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

        # 美化混淆矩阵显示
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
        plt.xlabel("预测类别")
        plt.ylabel("真实类别")
        plt.title("混淆矩阵")
        plt.tight_layout()
        plots["confusion_matrix"] = fig_to_base64(plt.gcf())
        plt.close()

        # 多分类问题 ROC 曲线
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
                        
                        # 获取类别名称（如果有预处理信息）
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
                            label=f'类别 {display_name} (AUC = {roc_auc:.3f})'
                        )
                
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('假阳性率 (False Positive Rate)')
                plt.ylabel('真阳性率 (True Positive Rate)')
                plt.title('多分类ROC曲线 (一对多方法)')
                plt.legend(loc="lower right", fontsize=9)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plots["roc_curve"] = fig_to_base64(plt.gcf())
                plt.close()

        elif len(np.unique(y_test)) == 2:  # 仅二分类绘制ROC
            y_probas = None
            if hasattr(model, "predict_proba"):
                y_probas = model.predict_proba(X_test)[:, 1]
            elif (
                hasattr(model, "net")
                and hasattr(model, "device")
                and hasattr(model, "is_classification")
                and model.is_classification
            ):  # PyTorch 模型
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

                    # 生成增强的ROC曲线
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f"ROC曲线 (AUC = {roc_auc:.3f})", color='#1f77b4', linewidth=2)
                    plt.plot([0, 1], [0, 1], "k--", label="随机猜测", alpha=0.8)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel("假阳性率 (1-特异性)")
                    plt.ylabel("真阳性率 (敏感性/召回率)")
                    plt.title("接收者操作特征(ROC)曲线")
                    plt.legend(loc="lower right")
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plots["roc_curve"] = fig_to_base64(plt.gcf())
                    plt.close()

                    # 添加精确率-召回率曲线
                    precision, recall, _ = precision_recall_curve(y_test, y_probas)
                    avg_precision = average_precision_score(y_test, y_probas)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(recall, precision, label=f"PR曲线 (AP = {avg_precision:.3f})", color='#ff7f0e', linewidth=2)
                    plt.xlabel("召回率")
                    plt.ylabel("精确率")
                    plt.title("精确率-召回率曲线")
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
        plt.xlabel("真实值")
        plt.ylabel("预测值")
        plt.title("预测 vs 真实值")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plots["prediction_vs_actual"] = fig_to_base64(plt.gcf())
        plt.close()

        # 添加残差图
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("预测值")
        plt.ylabel("残差")
        plt.title("残差图")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plots["residuals_plot"] = fig_to_base64(plt.gcf())
        plt.close()

        # 添加残差分布直方图
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel("残差")
        plt.ylabel("频率")
        plt.title("残差分布")
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
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    return img_str
