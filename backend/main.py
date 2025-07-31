from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import pandas as pd
import base64
import io
from starlette.responses import JSONResponse, FileResponse
import uvicorn
import json
import joblib
import zipfile
import tempfile
import os
import shutil
import torch
import sys
import webbrowser
from threading import Timer
from fastapi.staticfiles import StaticFiles

from models import (
    get_model,
    PyTorchModel,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    NeuralClassifier,
    NeuralRegressor,
    SklearnModelWrapper,
)
from processing import (
    preprocess_data,
    apply_saved_preprocessing,
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

current_model_package: Dict[str, Any] = {
    "model_object": None,
    "preprocessing_info": None,
    "metadata": None,
    "is_imported": False,
}


def get_path(relative_path):
    """获取资源的绝对路径，兼容开发环境和PyInstaller环境"""
    if hasattr(sys, '_MEIPASS'):
        # In a bundled app, the base path is the temp directory _MEIPASS
        # We assume the 'frontend' folder is copied to the root of the bundle.
        return os.path.join(sys._MEIPASS, relative_path)
    # In a development environment, we go up one level from 'backend' to the project root
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), relative_path)

# 静态文件目录
static_files_path = get_path("frontend")
app.mount("/static", StaticFiles(directory=static_files_path), name="static")

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(static_files_path, 'index.html'))

@app.get("/script.js")
async def get_script():
    return FileResponse(os.path.join(static_files_path, 'script.js'), media_type="application/javascript")

@app.get("/api")
async def api_root():
    return {"message": "ML Data Analysis Platform API"}

@app.get("/api/available-models")
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

    if XGBOOST_AVAILABLE:
        models["xgboost_classifier"] = "XGBoost分类器"
        models["xgboost_regressor"] = "XGBoost回归器"

    if LIGHTGBM_AVAILABLE:
        models["lightgbm_classifier"] = "LightGBM分类器"
        models["lightgbm_regressor"] = "LightGBM回归器"

    return {"models": models}


@app.post("/api/get-columns")
async def get_columns(file: UploadFile = File(...)):
    """
    获取上传文件的列名

    参数:
    - file: 上传的Excel或CSV文件

    返回:
    - columns: 文件的列名列表
    """
    try:
        content = await file.read()

        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400, detail="不支持的文件格式，请上传.csv或.xlsx文件"
            )
        return {"columns": df.columns.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取文件失败: {str(e)}")


@app.post("/api/train")
async def train_model(
    file: UploadFile = File(...),
    x_columns: str = Form(...),
    y_column: str = Form(...),
    model_name: str = Form(...),
    epochs: int = Form(100),  # 默认100轮
    use_early_stopping: str = Form("true"),  # 默认使用早停
):
    """
    接收数据文件和参数，训练模型并返回结果

    参数:
    - file: 上传的Excel或CSV文件
    - x_columns: JSON格式的特征列名列表
    - y_column: 目标列名
    - model_name: 选择的模型名称
    - epochs: 训练轮数(对于神经网络模型)
    - use_early_stopping: 是否使用早停机制(对于神经网络模型)

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

        global current_model_package
        current_model_package = {
            "model_object": None,
            "preprocessing_info": None,
            "metadata": None,
            "is_imported": False,
        }

        X_train, X_test, y_train, y_test, fitted_preprocessing_info = preprocess_data(
            df, x_columns_list, y_column
        )

        input_size_for_model = X_train.shape[1]

        model, is_classification = get_model(
            model_name,
            input_size=input_size_for_model,
            is_classification_hint=fitted_preprocessing_info.get("is_classification"),
        )

        if (
            isinstance(model, PyTorchModel)
            and model.net.fc1.in_features != input_size_for_model
        ):
            if model_name == "nn_classifier":
                from models import NeuralClassifier

                num_classes = (
                    len(fitted_preprocessing_info.get("target_encoder").classes_)
                    if fitted_preprocessing_info.get("target_encoder")
                    else 2
                )
                model.net = NeuralClassifier(
                    input_size_for_model, num_classes=num_classes
                )
            elif model_name == "nn_regressor":
                from models import NeuralRegressor

                model.net = NeuralRegressor(input_size_for_model)
                
            model.optimizer = torch.optim.Adam(model.net.parameters(), lr=0.001)

        importance_data = calculate_feature_importance(
            X_train, y_train, X_train.columns.tolist(), model_name
        )
        
        # 处理训练参数
        use_early_stopping_bool = use_early_stopping.lower() == "true"
        
        # 如果是PyTorch模型，应用自定义训练参数
        if isinstance(model, PyTorchModel):
            if use_early_stopping_bool:
                # 使用早停，设置合理的patience值
                patience = 10
                print(f"使用早停训练，epochs={epochs}, patience={patience}")
                training_history = model.fit(X_train, y_train, epochs=epochs, patience=patience)
            else:
                # 禁用早停，明确设置patience=None
                print(f"不使用早停训练，epochs={epochs}, patience=None")
                training_history = model.fit(X_train, y_train, epochs=epochs, patience=None)
        else:
            # 非PyTorch模型使用默认训练参数
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
            X_train.columns.tolist(),
            is_classification,
            preprocessing_info=fitted_preprocessing_info,
        )

        # 保存训练好的模型和信息
        current_model_package["model_object"] = model
        current_model_package["preprocessing_info"] = fitted_preprocessing_info
        current_model_package["metadata"] = {
            "model_name_backend": model_name,
            "is_classification": is_classification,
            "input_size": input_size_for_model,
            "original_x_columns": x_columns_list,
            "original_y_column": y_column,
            "feature_names_after_preprocessing": X_train.columns.tolist(),
        }
        current_model_package["is_imported"] = False

        return {"metrics": metrics, "plots": plots, "message": "模型训练与评估成功完成"}

    except Exception as e:
        import traceback

        tb_str = traceback.format_exc()
        print(f"Error during training: {e}\n{tb_str}")
        raise HTTPException(
            status_code=500, detail=f"处理过程中出错: {str(e)}\n{tb_str}"
        )


@app.post("/api/export-model")
async def export_model(background_tasks: BackgroundTasks):
    global current_model_package
    if not current_model_package or not current_model_package["model_object"]:
        raise HTTPException(
            status_code=404, detail="没有可导出的模型。请先训练或导入模型。"
        )

    model_object = current_model_package["model_object"]
    preprocessing_info = current_model_package["preprocessing_info"]
    metadata = current_model_package["metadata"]

    temp_dir = tempfile.mkdtemp()

    try:
        model_file_path_in_zip = "model_content"

        if isinstance(model_object, PyTorchModel):
            saved_model_filename = model_object.save_model(temp_dir, "model.pt")
            model_file_path_in_zip = "model.pt"
        else:
            saved_model_filename = model_object.save_model(temp_dir, "model.joblib")
            model_file_path_in_zip = "model.joblib"

        # 保存预处理信息
        preprocessing_info_path = os.path.join(temp_dir, "preprocessing_info.joblib")
        joblib.dump(preprocessing_info, preprocessing_info_path)

        # 保存元数据
        metadata_path = os.path.join(temp_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # 创建zip文件
        zip_file_path = os.path.join(temp_dir, "exported_model_package.zip")
        with zipfile.ZipFile(zip_file_path, "w") as zipf:
            zipf.write(saved_model_filename, arcname=model_file_path_in_zip)
            zipf.write(preprocessing_info_path, arcname="preprocessing_info.joblib")
            zipf.write(metadata_path, arcname="metadata.json")

        # 清理临时目录
        def cleanup_temp_dir():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"清理临时目录时发生错误: {e}")

        background_tasks.add_task(cleanup_temp_dir)

        return FileResponse(
            zip_file_path,
            media_type="application/zip",
            filename="exported_model_package.zip",
        )
    except Exception as e:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"导出模型过程中发生错误: {str(e)}")


@app.post("/api/import-model")
async def import_model_package(file: UploadFile = File(...)):
    global current_model_package
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="请上传 .zip 格式的模型包文件。")

    with tempfile.TemporaryDirectory() as temp_dir:
        uploaded_zip_path = os.path.join(temp_dir, file.filename)
        with open(uploaded_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extract_dir = os.path.join(temp_dir, "extracted_package")
        os.makedirs(extract_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(uploaded_zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="上传的ZIP文件无效或已损坏。")

        metadata_path = os.path.join(extract_dir, "metadata.json")
        preprocessing_info_path = os.path.join(extract_dir, "preprocessing_info.joblib")

        model_content_path_pt = os.path.join(extract_dir, "model.pt")
        model_content_path_joblib = os.path.join(extract_dir, "model.joblib")

        if not os.path.exists(metadata_path) or not os.path.exists(
            preprocessing_info_path
        ):
            raise HTTPException(
                status_code=400,
                detail="模型包缺少 metadata.json 或 preprocessing_info.joblib。",
            )

        model_content_actual_path = None
        if os.path.exists(model_content_path_pt):
            model_content_actual_path = model_content_path_pt
        elif os.path.exists(model_content_path_joblib):
            model_content_actual_path = model_content_path_joblib
        else:
            raise HTTPException(
                status_code=400,
                detail="模型包缺少模型文件 (model.pt 或 model.joblib)。",
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        preprocessing_info = joblib.load(preprocessing_info_path)

        model_name_backend = metadata["model_name_backend"]
        input_size = metadata.get("input_size")
        is_classification_hint = metadata.get("is_classification")

        # Get a model shell
        model_shell, is_classification_check = get_model(
            model_name_backend,
            input_size=input_size,
            is_classification_hint=is_classification_hint,
        )

        if is_classification_check != metadata["is_classification"]:
            print(
                f"Warning: is_classification mismatch. Metadata: {metadata['is_classification']}, get_model: {is_classification_check}"
            )

            metadata["is_classification"] = is_classification_check
        if (
            isinstance(model_shell, PyTorchModel)
            and model_shell.net.fc1.in_features != input_size
        ):
            if model_name_backend == "nn_classifier":
                num_classes = (
                    len(preprocessing_info.get("target_encoder").classes_)
                    if preprocessing_info.get("target_encoder")
                    else 2
                )
                model_shell.net = type(model_shell.net)(
                    input_size, num_classes=num_classes
                )  # Recreate with correct input_size
            elif model_name_backend == "nn_regressor":
                model_shell.net = type(model_shell.net)(input_size)

        model_shell.load_model(model_content_actual_path)

        current_model_package["model_object"] = model_shell
        current_model_package["preprocessing_info"] = preprocessing_info
        current_model_package["metadata"] = metadata
        current_model_package["is_imported"] = True

        return {"message": f"模型 '{model_name_backend}' 导入成功。"}


@app.post("/api/predict-with-imported-model")
async def predict_with_imported_model_endpoint(
    file: UploadFile = File(...),
    x_columns: str = Form(...),
    y_column: str = Form(...),
):
    global current_model_package
    if not current_model_package or not current_model_package["is_imported"]:
        raise HTTPException(
            status_code=400, detail="没有导入的模型可用于预测。请先导入模型。"
        )

    try:
        x_columns_list_new = json.loads(x_columns)

        content = await file.read()
        if file.filename.endswith(".csv"):
            df_new = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith((".xlsx", ".xls")):
            df_new = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400, detail="不支持的文件格式，请上传.csv或.xlsx文件。"
            )

        imported_model = current_model_package["model_object"]
        saved_preprocessing_info = current_model_package["preprocessing_info"]

        X_processed_new, y_processed_new, is_classification = apply_saved_preprocessing(
            df_new, x_columns_list_new, y_column, saved_preprocessing_info
        )

        if y_processed_new is None and y_column in df_new.columns:
            raise HTTPException(
                status_code=400,
                detail=f"无法处理目标列 '{y_column}'. 可能包含未知标签。",
            )

        metrics = {}
        if y_processed_new is not None:
            metrics = evaluate_model(
                imported_model, X_processed_new, y_processed_new, is_classification
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="使用导入模型预测时，需要提供有效的Y列用于评估。",
            )

        plots = generate_plots(
            model=imported_model,
            X_train=None,
            X_test=X_processed_new,
            y_train=None,
            y_test=y_processed_new,
            training_history={},
            importance_data=None,
            feature_names=X_processed_new.columns.tolist(),
            is_classification=is_classification,
            preprocessing_info=saved_preprocessing_info,
        )

        return {
            "metrics": metrics,
            "plots": plots,
            "message": "使用导入模型预测和评估完成。",
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback

        tb_str = traceback.format_exc()
        print(f"Error during prediction with imported model: {e}\n{tb_str}")
        raise HTTPException(
            status_code=500, detail=f"使用导入模型预测时出错: {str(e)}\n{tb_str}"
        )

def open_browser():
    webbrowser.open_new("http://127.0.0.1:19123")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    
    # Disable colored logging for PyInstaller compatibility in windowed mode
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["use_colors"] = False
    log_config["formatters"]["access"]["use_colors"] = False
    
    uvicorn.run(app, host="127.0.0.1", port=19123, reload=False, log_config=log_config)

