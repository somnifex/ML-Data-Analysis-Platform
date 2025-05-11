import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import copy
import joblib
import os

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class PyTorchModel:
    def __init__(self, net, criterion, optimizer, is_classification=True, device="cpu"):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.is_classification = is_classification
        self.device = device
        self.history = {"train_loss": [], "val_loss": []}
        if self.is_classification:
            self.history["train_acc"] = []
            self.history["val_acc"] = []

    def fit(
        self, X_train, y_train, val_split=0.2, batch_size=32, epochs=100, patience=10
    ):
        """
        训练模型

        参数:
        - X_train: 训练特征
        - y_train: 训练标签
        - val_split: 验证集比例
        - batch_size: 批量大小
        - epochs: 训练轮数
        - patience: 早停轮数

        返回:
        - history: 训练历史
        """
        X = torch.FloatTensor(X_train.values if hasattr(X_train, "values") else X_train)
        y = torch.FloatTensor(y_train.values if hasattr(y_train, "values") else y_train)

        if self.is_classification:
            y = y.long()

        # 分割训练集和验证集
        val_size = int(len(X) * val_split)
        indices = torch.randperm(len(X))

        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        X_train_t = X[train_indices]
        y_train_t = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_model = None
        no_improve_count = 0

        self.net.to(self.device)

        for epoch in range(epochs):
            self.net.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net(inputs)

                if self.is_classification:
                    loss = self.criterion(outputs, targets)
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += targets.size(0)
                    train_correct += (predicted == targets).sum().item()
                else:
                    targets = targets.view(-1, 1)
                    loss = self.criterion(outputs, targets)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            self.net.eval()
            with torch.no_grad():
                X_val_tensor = X_val.to(self.device)
                y_val_tensor = y_val.to(self.device)

                val_outputs = self.net(X_val_tensor)
                if self.is_classification:
                    _, predicted = torch.max(val_outputs.data, 1)
                    val_acc = (
                        predicted == y_val_tensor
                    ).sum().item() / y_val_tensor.size(0)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()

                    self.history["val_acc"].append(val_acc)
                    self.history["train_acc"].append(train_correct / train_total)
                else:
                    y_val_tensor = y_val_tensor.view(-1, 1)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()

            self.history["train_loss"].append(train_loss / len(train_loader))
            self.history["val_loss"].append(val_loss)

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.net.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(f"早停：{epoch+1}/{epochs}")
                break

        if best_model:
            self.net.load_state_dict(best_model)

        return self.history

    def predict(self, X):
        """
        预测
        """
        X_tensor = torch.FloatTensor(X.values if hasattr(X, "values") else X).to(
            self.device
        )
        self.net.eval()

        with torch.no_grad():
            outputs = self.net(X_tensor)

            if self.is_classification:
                _, predicted = torch.max(outputs, 1)
                return predicted.cpu().numpy()
            else:
                return outputs.cpu().numpy()

    def save_model(self, directory, filename="model.pt"):
        """保存PyTorch模型的状态字典"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, filename)
        torch.save(self.net.state_dict(), path)
        return path

    def load_model(self, path):
        """加载PyTorch模型的状态字典"""
        if self.device == "cpu":
            self.net.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        else:
            self.net.load_state_dict(torch.load(path))
        self.net.to(self.device)
        self.net.eval()


class NeuralClassifier(nn.Module):
    """
    神经网络分类器
    """

    def __init__(self, input_size, hidden_size=64, num_classes=2):
        super(NeuralClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class NeuralRegressor(nn.Module):
    """
    神经网络回归器
    """

    def __init__(self, input_size, hidden_size=64):
        super(NeuralRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class SklearnModelWrapper:
    """
    封装Sklearn模型，提供与PyTorch模型相似的API
    """

    def __init__(self, model):
        self.model = model
        self.history = {"train_loss": [], "val_loss": []}

    def fit(self, X_train, y_train):
        """
        训练Sklearn模型
        """
        self.model.fit(X_train, y_train)
        return self.history

    def predict(self, X):
        """
        预测
        """
        return self.model.predict(X)

    def save_model(self, directory, filename="model.joblib"):
        """保存Sklearn模型"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, filename)
        joblib.dump(self.model, path)
        return path

    def load_model(self, path):
        """加载Sklearn模型"""
        self.model = joblib.load(path)


def get_model(model_name, input_size=None, is_classification_hint=None):
    """
    根据模型名获取相应模型实例或空的模型结构用于加载。

    参数:
    - model_name: 模型名称
    - input_size: 输入特征维度 (主要用于PyTorch模型实例化)
    - is_classification_hint: 布尔值，当模型名称不明确区分分类/回归时提供提示
                                (例如, 'random_forest' 可能需要这个)

    返回:
    - model: 模型实例
    - is_classification: 是否是分类模型 (布尔值)
    """
    if "nn_" in model_name and input_size is None:
        pass

    if model_name == "nn_classifier":
        net = NeuralClassifier(
            input_size if input_size else 1
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        model = PyTorchModel(net, criterion, optimizer, is_classification=True)
        return model, True

    elif model_name == "nn_regressor":
        net = NeuralRegressor(input_size if input_size else 1) 
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        model = PyTorchModel(net, criterion, optimizer, is_classification=False)
        return model, False

    elif model_name == "linear_regression":
        model = SklearnModelWrapper(LinearRegression())
        return model, False

    elif model_name == "logistic_regression":
        model = SklearnModelWrapper(LogisticRegression(max_iter=1000))
        return model, True

    elif model_name == "random_forest_classifier":
        model = SklearnModelWrapper(RandomForestClassifier(n_estimators=100))
        return model, True

    elif model_name == "random_forest_regressor":
        model = SklearnModelWrapper(RandomForestRegressor(n_estimators=100))
        return model, False

    # 支持向量机 - 分类
    elif model_name == "svm_classifier":
        model = SklearnModelWrapper(SVC(probability=True))
        return model, True

    # 支持向量机 - 回归
    elif model_name == "svm_regressor":
        model = SklearnModelWrapper(SVR())
        return model, False

    # K近邻 - 分类
    elif model_name == "knn_classifier":
        model = SklearnModelWrapper(KNeighborsClassifier(n_neighbors=5))
        return model, True

    # K近邻 - 回归
    elif model_name == "knn_regressor":
        model = SklearnModelWrapper(KNeighborsRegressor(n_neighbors=5))
        return model, False

    # 决策树 - 分类
    elif model_name == "decision_tree_classifier":
        model = SklearnModelWrapper(DecisionTreeClassifier())
        return model, True

    # 决策树 - 回归
    elif model_name == "decision_tree_regressor":
        model = SklearnModelWrapper(DecisionTreeRegressor())
        return model, False

    # 梯度提升树 - 分类
    elif model_name == "gradient_boosting_classifier":
        model = SklearnModelWrapper(GradientBoostingClassifier())
        return model, True

    # 梯度提升树 - 回归
    elif model_name == "gradient_boosting_regressor":
        model = SklearnModelWrapper(GradientBoostingRegressor())
        return model, False

    # 弹性网络回归
    elif model_name == "elastic_net":
        model = SklearnModelWrapper(ElasticNet())
        return model, False

    # XGBoost - 条件导入
    elif model_name == "xgboost_classifier" and XGBOOST_AVAILABLE:
        model = SklearnModelWrapper(xgb.XGBClassifier())
        return model, True

    elif model_name == "xgboost_regressor" and XGBOOST_AVAILABLE:
        model = SklearnModelWrapper(xgb.XGBRegressor())
        return model, False

    # LightGBM - 条件导入
    elif model_name == "lightgbm_classifier" and LIGHTGBM_AVAILABLE:
        model = SklearnModelWrapper(lgb.LGBMClassifier())
        return model, True

    elif model_name == "lightgbm_regressor" and LIGHTGBM_AVAILABLE:
        model = SklearnModelWrapper(lgb.LGBMRegressor())
        return model, False

    elif model_name == "random_forest":
        if is_classification_hint is False:
            model = SklearnModelWrapper(RandomForestRegressor(n_estimators=100))
            return model, False
        model = SklearnModelWrapper(RandomForestClassifier(n_estimators=100))
        return model, True

    else:
        raise ValueError(f"不支持的模型: {model_name}")
