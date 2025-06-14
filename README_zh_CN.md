# 机器学习数据分析平台

*[English Version](README.md)*

一个不算太强大的机器学习数据分析平台Demo，旨在无需编程即可使用常用ML模型进行训练和评估。上传数据、选择模型，几秒钟内即可获得专业的可视化和指标分析。这个Demo无法满足所有需求，我们鼓励用户自行修改以适用自己的使用场景。
## 功能特性

- 🚀 **无代码界面**：用户友好的模型选择和配置界面
- 📊 **数据可视化**：自动生成模型性能可视化图表
- 📈 **特征分析**：提供特征重要性分析
- 🔄 **多种ML模型**：支持各种分类和回归模型
- 📁 **数据灵活性**：支持Excel（.xlsx、.xls）和CSV文件
- 📱 **响应式设计**：同时适用于桌面和移动设备

## 快速开始

### 前提条件

- 开发时使用了Python 3.12及pip（其他环境未测试）
- 现代网页浏览器（Chrome、Firefox、Edge等）

### 安装步骤

1. 克隆仓库

   ```bash
   git clone https://github.com/yourusername/ml-data-analysis.git
   cd ml-data-analysis
   ```
2. 安装所需的Python包

   ```bash
   pip install -r requirements.txt
   ```
3. 启动后端服务器

   ```bash
   cd backend
   python main.py
   ```
4. 在浏览器中打开前端页面

   - 在文件浏览器中导航到 `frontend/index.html`并用浏览器打开
   - 或者运行一个简单的HTTP服务器：
     ```bash
     cd frontend
     python -m http.server 8000
     ```

     然后在浏览器中访问 `http://localhost:8000`

## 使用方法

1. **上传数据**：使用文件上传按钮上传Excel或CSV数据文件
2. **选择模型**：根据您的任务类型选择适合的ML模型
3. **配置特征**：选择特征列和目标变量
4. **训练模型**：点击"开始训练"按钮并等待结果
5. **查看结果**：探索生成的指标和可视化图表

## 支持的模型

### 分类模型

- 神经网络分类器
- 逻辑回归
- 随机森林分类器
- 决策树分类器
- 梯度提升树分类器
- 支持向量机分类器
- K近邻分类器
- XGBoost分类器（若已安装）
- LightGBM分类器（若已安装）

### 回归模型

- 神经网络回归器
- 线性回归
- 随机森林回归器
- 决策树回归器
- 梯度提升树回归器
- 支持向量机回归器
- K近邻回归器
- 弹性网络回归
- XGBoost回归器（若已安装）
- LightGBM回归器（若已安装）

## 项目结构

```
ml-data-analysis/
├── backend/               # 后端API服务器
│   ├── main.py            # FastAPI服务器入口点
│   ├── models.py          # ML模型实现
│   └── processing.py      # 数据处理工具
├── frontend/              # 前端Web界面
│   ├── index.html         # 主HTML页面
│   ├── script.js          # JavaScript逻辑
│   └── assets/            # 图片和资源
└── gendata.py             # 数据生成工具脚本
```

## 可选：生成测试数据

您可以使用包含的脚本为各种ML场景生成测试数据：

```bash
python gendata.py
```

这将在 `ml_test_data`目录中创建包含分类、回归和聚类任务的合成数据的Excel文件。

## 许可证

本项目采用GNU通用公共许可证v3.0授权 - 详见[LICENSE](LICENSE)文件。

## 致谢

- 使用[FastAPI](https://fastapi.tiangolo.com/)构建
- 使用[scikit-learn](https://scikit-learn.org/)和[PyTorch](https://pytorch.org/)进行机器学习
- 可视化由[Matplotlib](https://matplotlib.org/)和[Seaborn](https://seaborn.pydata.org/)提供支持
