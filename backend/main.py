from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import pandas as pd
import base64
import io
from starlette.responses import JSONResponse
import uvicorn
import json

from models import get_model
from processing import (
    preprocess_data,
    calculate_feature_importance,
    evaluate_model,
    generate_plots,
)

app = FastAPI(title="ML模型分析API", description="用于数据分析的机器学习API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "机器学习模型分析API服务已启动"}


@app.get("/available-models")
async def get_available_models():
    """
    获取所有可用的模型列表
    """
    # 基础模型
    models = {
        # 神经网络模型
        "nn_classifier": "神经网络分类器",
        "nn_regressor": "神经网络回归器",
        # 线性模型
        "linear_regression": "线性回归",
        "logistic_regression": "逻辑回归",
        "elastic_net": "弹性网络回归",
        # 树模型
        "decision_tree_classifier": "决策树分类器",
        "decision_tree_regressor": "决策树回归器",
        "random_forest_classifier": "随机森林分类器",
        "random_forest_regressor": "随机森林回归器",
        "gradient_boosting_classifier": "梯度提升树分类器",
        "gradient_boosting_regressor": "梯度提升树回归器",
        # SVM模型
        "svm_classifier": "支持向量机分类器",
        "svm_regressor": "支持向量机回归器",
        # KNN模型
        "knn_classifier": "K近邻分类器",
        "knn_regressor": "K近邻回归器",
    }

    # 兼容旧版本
    models["random_forest"] = "随机森林 (默认分类)"

    from models import XGBOOST_AVAILABLE

    if XGBOOST_AVAILABLE:
        models["xgboost_classifier"] = "XGBoost分类器"
        models["xgboost_regressor"] = "XGBoost回归器"

    from models import LIGHTGBM_AVAILABLE

    if LIGHTGBM_AVAILABLE:
        models["lightgbm_classifier"] = "LightGBM分类器"
        models["lightgbm_regressor"] = "LightGBM回归器"

    return {"models": models}


@app.post("/get-columns")
async def get_columns(file: UploadFile = File(...)):
    """
    获取上传文件的列名

    参数:
    - file: 上传的Excel或CSV文件

    返回:
    - columns: 文件的列名列表
    """
    try:
        # 读取上传的文件
        content = await file.read()

        # 根据文件类型读取数据
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400, detail="不支持的文件格式，请上传.csv或.xlsx文件"
            )

        # 返回列名列表
        return {"columns": df.columns.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取文件失败: {str(e)}")


@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    x_columns: str = Form(...),
    y_column: str = Form(...),
    model_name: str = Form(...),
):
    """
    接收数据文件和参数，训练模型并返回结果

    参数:
    - file: 上传的Excel或CSV文件
    - x_columns: JSON格式的特征列名列表
    - y_column: 目标列名
    - model_name: 选择的模型名称

    返回:
    - metrics: 评估指标字典
    - plots: base64编码的图表字典
    - message: 处理状态消息
    """
    try:
        x_columns_list = json.loads(x_columns)

        content = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400, detail="不支持的文件格式，请上传.csv或.xlsx文件"
            )

        X_train, X_test, y_train, y_test, preprocessing_info = preprocess_data(
            df, x_columns_list, y_column
        )

        model, is_classification = get_model(model_name, X_train.shape[1])

        importance_data = calculate_feature_importance(
            X_train, y_train, x_columns_list, model_name
        )

        training_history = model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, is_classification)

        plots = generate_plots(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            training_history,
            importance_data,
            x_columns_list,
            is_classification,
        )

        return {"metrics": metrics, "plots": plots, "message": "模型训练与评估成功完成"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理过程中出错: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=19123, reload=True)
