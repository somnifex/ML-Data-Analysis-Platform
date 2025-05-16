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
import torch.nn.functional as F

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
        - patience: 早停轮数，设为None时完全禁用早停

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

        # 明确检查早停是否启用
        use_early_stopping = patience is not None

        print(f"PyTorch模型训练: epochs={epochs}, {'启用' if use_early_stopping else '禁用'}早停")
        
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

            # 记录本轮的损失值
            self.history["train_loss"].append(train_loss / len(train_loader))
            self.history["val_loss"].append(val_loss)

            # 始终保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.net.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 只有在启用早停的情况下才检查是否需要提前终止
            if use_early_stopping and no_improve_count >= patience:
                print(f"早停触发：{epoch+1}/{epochs}，验证损失已连续{patience}轮未改善")
                break
            
            # 每10轮显示一次进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"训练进度: [{epoch+1}/{epochs}] 训练损失: {train_loss/len(train_loader):.4f} 验证损失: {val_loss:.4f}")

        # 加载最佳模型参数
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
                # 确保回归输出是一维的
                outputs_np = outputs.cpu().numpy()
                if outputs_np.ndim > 1:  # 如果是多维的
                    if outputs_np.shape[1] == 1:  # 通常回归模型输出形状是(batch_size, 1)
                        return outputs_np.flatten()  # 压缩为一维数组
                    # 如果是矩阵输出，取对角线元素（可能是注意力机制导致的）
                    elif outputs_np.shape[0] == outputs_np.shape[1]:  
                        return np.diag(outputs_np).flatten()
                    # 如果是其他形状，取第一列
                    else:  
                        return outputs_np[:, 0]
                return outputs_np

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


# 添加EMAX注意力模块 - 原始版本（2D版本，用于图像数据）
class EMAX(nn.Module):
    def __init__(self, channels, c2=None, factor=32, reduction=16):
        super(EMAX, self).__init__()
        self.groups = factor
        self.channels = channels
        assert channels // self.groups > 0

        # Spatial attention branch
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

        # Channel attention branch
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        # Input Feature Distribution Adaptive Dynamic Grouping Controller
        self.group_controller = nn.Linear(channels, self.groups)

        # Gating residual connection
        self.gamma = nn.Parameter(torch.zeros(1))

        # Global attention branch
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Dynamic grouping
        group_logits = self.agp(x).view(b, c)  # [b, c]
        group_weights = self.group_controller(group_logits).softmax(dim=-1)  # [b, groups]
        group_indices = torch.multinomial(group_weights, 1).squeeze()  # [b]
        groups = torch.clamp(group_indices, min=1, max=self.groups).mode().values.item()
        group_x = x.reshape(b * groups, -1, h, w)

        # Spatial attention
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(self.agp(x1).reshape(b * groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * groups, c // groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * groups, c // groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * groups, 1, h, w)
        spatial_out = (group_x * weights.sigmoid()).reshape(b, c, h, w)

        # Channel attention
        channel_weights = self.channel_gate(x)
        channel_out = x * channel_weights

        # Fusing spatial information with channel attention
        fused = spatial_out + self.gamma * channel_out

        # Global attention enhancement
        global_weights = self.global_attn(fused)
        out = fused * global_weights

        return out


# 为表格数据添加一维版本的EMAX注意力模块，修复形状兼容性问题
class EMAX1D(nn.Module):
    def __init__(self, features_dim, factor=8, reduction=4):
        super(EMAX1D, self).__init__()
        self.features_dim = features_dim
        
        # 简化的特征注意力机制
        self.feature_attention = nn.Sequential(
            nn.Linear(features_dim, features_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(features_dim // reduction, features_dim),
            nn.Sigmoid()
        )
        
        # 全局注意力分支
        self.global_attention = nn.Sequential(
            nn.Linear(features_dim, features_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(features_dim // reduction, features_dim),
            nn.Sigmoid()
        )
        
        # 残差连接权重
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 添加特征交互学习层
        self.feature_interaction = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # 输入x形状为 [batch_size, features_dim]
        batch_size = x.size(0)
        
        # 应用特征注意力
        feature_weights = self.feature_attention(x)
        feature_enhanced = x * feature_weights
        
        # 特征交互学习
        feature_interaction = self.feature_interaction(feature_enhanced)
        
        # 全局注意力增强
        global_weights = self.global_attention(feature_interaction)
        enhanced_output = feature_interaction * global_weights
        
        # 残差连接
        output = enhanced_output + self.gamma * x
        
        return output


# 使用EMAX1D的分类器
class EMAXClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=2):
        super(EMAXClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.attention = EMAX1D(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.attention(x)  # 应用注意力
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 使用EMAX1D的回归器
class EMAXRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(EMAXRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.attention = EMAX1D(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.attention(x)  # 应用注意力
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
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
        net = NeuralClassifier(input_size if input_size else 1)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        model = PyTorchModel(net, criterion, optimizer, is_classification=True)
        return model, True
    
    elif model_name == "emax_classifier":
        net = EMAXClassifier(input_size if input_size else 1)
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
    
    elif model_name == "emax_regressor":
        net = EMAXRegressor(input_size if input_size else 1)
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

    # 支持向量机 - 分类，确保启用概率输出
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
