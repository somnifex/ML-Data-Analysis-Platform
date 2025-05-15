const API_BASE_URL = 'http://localhost:19123';

// 保存文件中的列名
let fileColumns = [];
// 保存已选择的特征列
let xColumns = [];
// 保存选择的目标列
let yColumn = '';
// 当前是否有导入的模型
let isModelImported = false;

document.addEventListener('DOMContentLoaded', function () {
    fetchAvailableModels();
    document.getElementById('file').addEventListener('change', handleFileChange);
    document.getElementById('train-button').addEventListener('click', handleTrainButtonClick);
    document.getElementById('export-model-button').addEventListener('click', handleExportModelClick);
    document.getElementById('import-model-file').addEventListener('change', handleImportModelFileChange);
    document.getElementById('predict-imported-button').addEventListener('click', handlePredictWithImportedModelClick);

    // 添加模型选择变化监听
    document.getElementById('model').addEventListener('change', function () {
        if (this.value) {
            updateStepIndicator(2);
        }
    });
});

/**
 * 更新步骤指示器状态
 */
function updateStepIndicator(currentStep) {
    const steps = document.querySelectorAll('.step');

    steps.forEach((step, index) => {
        const stepNum = index + 1;

        // 如果模型已导入，针对步骤3(配置特征)和步骤4(查看结果)特殊处理
        if (isModelImported && currentStep === 4) {
            // 导入模型预测时，直接跳到结果步骤
            if (stepNum === 3) {
                step.classList.add('completed');
                step.classList.remove('active');
            } else if (stepNum === 4) {
                step.classList.add('active');
                step.classList.remove('completed');
            }
        } else {
            if (stepNum < currentStep) {
                step.classList.remove('active');
                step.classList.add('completed');
            } else if (stepNum === currentStep) {
                step.classList.add('active');
                step.classList.remove('completed');
            } else {
                step.classList.remove('active', 'completed');
            }
        }
    });
}

/**
 * 从API获取可用模型列表
 */
async function fetchAvailableModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/available-models`);
        if (!response.ok) {
            throw new Error(`HTTP错误 ${response.status}`);
        }

        const data = await response.json();
        populateModelSelect(data.models);
    } catch (error) {
        console.error('获取模型列表失败:', error);
        const modelSelect = document.getElementById('model');
        const option = document.createElement('option');
        option.value = '';
        option.textContent = '无法加载模型列表';
        modelSelect.appendChild(option);
    }
}

/**
 * 填充模型选择下拉框
 */
function populateModelSelect(models) {
    const modelSelect = document.getElementById('model');

    // 清空现有选项（保留"请选择模型"提示）
    while (modelSelect.options.length > 1) {
        modelSelect.remove(1);
    }

    // 添加模型选项
    for (const [id, name] of Object.entries(models)) {
        const option = document.createElement('option');
        option.value = id;
        option.textContent = name;
        modelSelect.appendChild(option);
    }
}

/**
 * 处理文件上传变化
 */
async function handleFileChange(event) {
    const fileInput = event.target;

    if (!fileInput.files || fileInput.files.length === 0) {
        document.getElementById('columns-selection').style.display = 'none';
        return;
    }

    const file = fileInput.files[0];

    // 更新步骤指示器
    updateStepIndicator(1);

    // 检查文件格式
    if (!file.name.endsWith('.xlsx') && !file.name.endsWith('.xls') && !file.name.endsWith('.csv')) {
        showError('file-error', '请上传.xlsx、.xls或.csv格式的文件');
        return;
    }

    clearErrors();
    showLoading(true);

    try {
        // 创建表单数据
        const formData = new FormData();
        formData.append('file', file);

        // 发送请求获取文件列名
        const response = await fetch(`${API_BASE_URL}/get-columns`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP错误 ${response.status}`);
        }

        const data = await response.json();

        // 保存列名并显示选择界面
        fileColumns = data.columns;
        displayColumnSelections(fileColumns);
        document.getElementById('columns-selection').style.display = 'block';

        // 更新步骤指示器
        updateStepIndicator(2);

    } catch (error) {
        console.error('读取文件列名失败:', error);
        showError('file-error', '无法读取文件列名，请检查文件格式是否正确');
    } finally {
        showLoading(false);
    }
}

/**
 * 显示列名选择界面
 */
function displayColumnSelections(columns) {
    // 清空现有选项
    xColumns = [];
    yColumn = '';

    // 显示特征列选择
    const xColumnsContainer = document.getElementById('x-columns-container');
    xColumnsContainer.innerHTML = '';

    // 显示目标列选择
    const yColumnContainer = document.getElementById('y-column-container');
    yColumnContainer.innerHTML = '';

    // 更新计数器初始状态
    updateColumnsCounter('x', 0);
    updateColumnsCounter('y', '');

    // 设置搜索框和全选按钮事件
    setupSearchFilter('x-search', 'x-columns-container');
    setupSearchFilter('y-search', 'y-column-container');

    document.getElementById('select-all-x').addEventListener('click', () => selectAllColumns(true));
    document.getElementById('deselect-all-x').addEventListener('click', () => selectAllColumns(false));

    // 填充列选项
    columns.forEach(column => {
        // 为X列创建复选框
        const xColumnItem = document.createElement('div');
        xColumnItem.className = 'column-item';
        xColumnItem.dataset.columnName = column;

        const xCheckbox = document.createElement('input');
        xCheckbox.type = 'checkbox';
        xCheckbox.className = 'column-checkbox';
        xCheckbox.id = `x-${column}`;
        xCheckbox.value = column;

        const xLabel = document.createElement('label');
        xLabel.htmlFor = `x-${column}`;
        xLabel.textContent = column;

        xColumnItem.appendChild(xCheckbox);
        xColumnItem.appendChild(xLabel);
        xColumnsContainer.appendChild(xColumnItem);

        // 为整个列项添加点击事件
        xColumnItem.addEventListener('click', function (e) {
            // 如果点击的是复选框本身，不需要额外处理
            if (e.target !== xCheckbox) {
                xCheckbox.checked = !xCheckbox.checked;
                // 手动触发change事件
                const event = new Event('change');
                xCheckbox.dispatchEvent(event);
            }
        });

        xCheckbox.addEventListener('change', function () {
            if (this.checked) {
                // 将列添加到特征列数组
                if (!xColumns.includes(column)) {
                    xColumns.push(column);
                    xColumnItem.classList.add('selected');
                }
                // 如果该列被选为目标列，则取消选择
                if (yColumn === column) {
                    document.getElementById(`y-${column}`).checked = false;
                    document.querySelector(`.column-item[data-column-name="${column}"]`).classList.remove('selected');
                    yColumn = '';
                    updateColumnsCounter('y', '');
                }

                // 更新计数器
                updateColumnsCounter('x', xColumns.length);

                // 如果至少选择了一个特征列和一个目标列，更新步骤
                if (xColumns.length > 0 && yColumn) {
                    updateStepIndicator(3);
                }
            } else {
                // 从特征列数组中移除
                xColumns = xColumns.filter(col => col !== column);
                xColumnItem.classList.remove('selected');

                // 更新计数器
                updateColumnsCounter('x', xColumns.length);

                // 如果没有选择特征列，退回步骤
                if (xColumns.length === 0) {
                    updateStepIndicator(2);
                }
            }
        });

        // 为Y列创建单选按钮
        const yColumnItem = document.createElement('div');
        yColumnItem.className = 'column-item';
        yColumnItem.dataset.columnName = column;

        const yRadio = document.createElement('input');
        yRadio.type = 'radio';
        yRadio.className = 'column-radio';
        yRadio.name = 'y-column';
        yRadio.id = `y-${column}`;
        yRadio.value = column;

        const yLabel = document.createElement('label');
        yLabel.htmlFor = `y-${column}`;
        yLabel.textContent = column;

        yColumnItem.appendChild(yRadio);
        yColumnItem.appendChild(yLabel);
        yColumnContainer.appendChild(yColumnItem);

        // 为整个列项添加点击事件
        yColumnItem.addEventListener('click', function (e) {
            // 如果点击的是单选框本身，不需要额外处理
            if (e.target !== yRadio) {
                yRadio.checked = true;
                // 手动触发change事件
                const event = new Event('change');
                yRadio.dispatchEvent(event);
            }
        });

        yRadio.addEventListener('change', function () {
            if (this.checked) {
                // 先移除之前选中项的样式
                if (yColumn) {
                    const prevItem = document.querySelector(`.column-item[data-column-name="${yColumn}"]`);
                    if (prevItem) prevItem.classList.remove('selected');
                }

                // 设置目标列
                yColumn = column;
                yColumnItem.classList.add('selected');
                updateColumnsCounter('y', yColumn);

                // 如果该列被选为特征列，则取消选择
                const xCheckbox = document.getElementById(`x-${column}`);
                if (xCheckbox && xCheckbox.checked) {
                    xCheckbox.checked = false;
                    document.querySelector(`#x-columns-container .column-item[data-column-name="${column}"]`).classList.remove('selected');
                    xColumns = xColumns.filter(col => col !== column);
                    updateColumnsCounter('x', xColumns.length);
                }

                // 如果至少选择了一个特征列和一个目标列，更新步骤
                if (xColumns.length > 0 && yColumn) {
                    updateStepIndicator(3);
                }
            }
        });
    });
}

/**
 * 设置搜索过滤功能
 */
function setupSearchFilter(searchId, containerId) {
    const searchInput = document.getElementById(searchId);

    searchInput.addEventListener('input', function () {
        const searchTerm = this.value.toLowerCase();
        const columnsContainer = document.getElementById(containerId);
        const columnItems = columnsContainer.querySelectorAll('.column-item');

        columnItems.forEach(item => {
            const columnName = item.dataset.columnName.toLowerCase();
            if (columnName.includes(searchTerm)) {
                item.classList.remove('hidden-column');
            } else {
                item.classList.add('hidden-column');
            }
        });
    });
}

/**
 * 全选或取消全选特征列
 */
function selectAllColumns(select) {
    const xColumnsContainer = document.getElementById('x-columns-container');
    const checkboxes = xColumnsContainer.querySelectorAll('.column-checkbox:not([disabled])');
    const visibleItems = xColumnsContainer.querySelectorAll('.column-item:not(.hidden-column)');

    visibleItems.forEach(item => {
        const checkbox = item.querySelector('.column-checkbox');
        if (checkbox && checkbox.checked !== select) {
            checkbox.checked = select;

            // 手动触发change事件
            const event = new Event('change');
            checkbox.dispatchEvent(event);
        }
    });
}

/**
 * 更新列选择计数器
 */
function updateColumnsCounter(type, count) {
    if (type === 'x') {
        document.querySelector('#x-columns-counter span').textContent = count;
    } else if (type === 'y') {
        document.querySelector('#y-columns-counter span').textContent = count || '无';
    }
}

/**
 * 处理开始训练按钮点击
 */
async function handleTrainButtonClick() {
    // 重置错误信息
    clearErrors();
    hideImportSuccessMessage(); // 隐藏导入成功消息

    // 验证输入
    if (!validateInputs()) {
        return;
    }

    // 显示加载状态
    showLoading(true);

    // 隐藏相关按钮
    document.getElementById('export-model-button').style.display = 'none';
    document.getElementById('predict-imported-button').style.display = 'none';
    isModelImported = false; // 重置导入模型状态

    // 准备表单数据
    const formData = new FormData();
    formData.append('file', document.getElementById('file').files[0]);
    formData.append('x_columns', JSON.stringify(xColumns));
    formData.append('y_column', yColumn);
    formData.append('model_name', document.getElementById('model').value);

    try {
        // 发送API请求
        const response = await fetch(`${API_BASE_URL}/train`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP错误 ${response.status}`);
        }

        const result = await response.json();

        // 处理结果
        displayResults(result);
        // 更新步骤指示器
        updateStepIndicator(4);
        // 显示导出模型按钮
        document.getElementById('export-model-button').style.display = 'inline-block';
        // 更新UI状态
        updateUIAfterModelStatusChange();

        // 平滑滚动到结果区域
        document.getElementById('results').scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });

    } catch (error) {
        console.error('训练请求失败:', error);
        showError('server-error', `服务请求失败: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

/**
 * 验证所有输入
 */
function validateInputs() {
    let isValid = true;

    // 验证文件上传
    const fileInput = document.getElementById('file');
    if (!fileInput.files || fileInput.files.length === 0) {
        showError('file-error', '请上传数据文件');
        isValid = false;
    } else {
        const fileName = fileInput.files[0].name;
        if (!fileName.endsWith('.xlsx') && !fileName.endsWith('.xls') && !fileName.endsWith('.csv')) {
            showError('file-error', '请上传.xlsx、.xls或.csv格式的文件');
            isValid = false;
        }
    }

    // 验证模型选择
    const modelSelect = document.getElementById('model');
    if (!modelSelect.value) {
        showError('model-error', '请选择一个模型');
        isValid = false;
    }

    // 验证特征列
    if (xColumns.length === 0) {
        showError('x-columns-error', '请至少选择一个特征列');
        isValid = false;
    }

    // 验证目标列
    if (!yColumn) {
        showError('y-column-error', '请选择一个目标列');
        isValid = false;
    }

    return isValid;
}

/**
 * 显示错误信息
 */
function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = message;
        element.style.display = 'block';
        // 添加错误动画
        element.classList.add('animate-error');
        setTimeout(() => {
            element.classList.remove('animate-error');
        }, 500);
    }
}

/**
 * 清除所有错误信息
 */
function clearErrors() {
    const errorElements = document.querySelectorAll('.error');
    errorElements.forEach(element => {
        element.textContent = '';
        element.style.display = 'none';
    });
}

/**
 * 显示/隐藏加载状态
 */
function showLoading(isLoading) {
    const loadingElement = document.getElementById('loading');
    const trainButton = document.getElementById('train-button');

    if (isLoading) {
        loadingElement.style.display = 'block';
        trainButton.disabled = true;
    } else {
        loadingElement.style.display = 'none';
        trainButton.disabled = false;
    }
}

/**
 * 显示结果
 */
function displayResults(result) {
    const resultsElement = document.getElementById('results');
    resultsElement.style.display = 'block';

    // 清空旧数据
    document.getElementById('metrics-table').getElementsByTagName('tbody')[0].innerHTML = '';

    // 显示指标
    if (result.metrics) {
        displayMetrics(result.metrics);
    }

    // 显示图表
    if (result.plots) {
        displayPlots(result.plots);
    }
}

/**
 * 显示评估指标
 */
function displayMetrics(metrics) {
    const metricsTable = document.getElementById('metrics-table').getElementsByTagName('tbody')[0];
    metricsTable.innerHTML = '';

    // 指标名称映射
    const metricNames = {
        'accuracy': '准确率',
        'precision': '精确率',
        'recall': '召回率',
        'f1': 'F1分数',
        'auc': 'AUC值(ROC曲线下面积)',
        'specificity': '特异性',
        'r2': 'R²决定系数',
        'mse': '均方误差(MSE)',
        'rmse': '均方根误差(RMSE)',
        'mae': '平均绝对误差(MAE)'
    };

    // 显示顺序
    const displayOrder = [
        'accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity',
        'r2', 'rmse', 'mse', 'mae'
    ];

    // 按显示顺序添加指标
    displayOrder.forEach(key => {
        if (metrics.hasOwnProperty(key)) {
            const value = metrics[key];
            const row = metricsTable.insertRow();
            const nameCell = row.insertCell(0);
            const valueCell = row.insertCell(1);

            nameCell.textContent = metricNames[key] || key;
            // 格式化数值（保留4位小数）
            valueCell.textContent = typeof value === 'number' ? value.toFixed(4) : value;

            // 为良好的指标值添加颜色
            if ((key === 'accuracy' || key === 'precision' || key === 'recall' || key === 'f1' || key === 'auc' || key === 'r2') && value > 0.7) {
                valueCell.style.color = 'var(--success-color)';
                valueCell.style.fontWeight = 'bold';
            } else if ((key === 'mse' || key === 'mae' || key === 'rmse') && value < 0.3) {
                valueCell.style.color = 'var(--success-color)';
                valueCell.style.fontWeight = 'bold';
            } else if ((key === 'accuracy' || key === 'precision' || key === 'recall' || key === 'f1' || key === 'auc' || key === 'r2') && value < 0.5) {
                valueCell.style.color = 'var(--danger-color)';
                valueCell.style.fontWeight = 'bold';
            }
        }
    });

    // 添加其他未在显示顺序中的指标（如果有的话）
    for (const [key, value] of Object.entries(metrics)) {
        if (key === 'confusion_matrix') continue; // 混淆矩阵在图表中展示
        if (displayOrder.includes(key)) continue; // 已经显示的指标跳过

        const row = metricsTable.insertRow();
        const nameCell = row.insertCell(0);
        const valueCell = row.insertCell(1);

        nameCell.textContent = metricNames[key] || key;
        // 格式化数值（保留4位小数）
        valueCell.textContent = typeof value === 'number' ? value.toFixed(4) : value;
    }
}

/**
 * 显示图表
 */
function displayPlots(plots) {
    // 特征重要性图
    const featureImportancePlot = document.getElementById('feature-importance-plot');
    const featureImportanceCard = featureImportancePlot.closest('.card');
    const featureImportanceTitle = featureImportanceCard.querySelector('h2');

    if (plots.feature_importance) {
        featureImportancePlot.src = `data:image/png;base64,${plots.feature_importance}`;
        featureImportanceTitle.textContent = '特征重要性';
        featureImportanceCard.style.display = 'block';
    } else {
        featureImportanceCard.style.display = 'none';
    }

    // 学习曲线图
    const learningCurvePlot = document.getElementById('learning-curve-plot');
    const learningCurveCard = learningCurvePlot.closest('.card');
    const learningCurveTitle = learningCurveCard.querySelector('h2');

    let learningCurveSrc = null;
    if (plots.learning_curve) {
        learningCurveSrc = `data:image/png;base64,${plots.learning_curve}`;
        learningCurveTitle.textContent = '学习曲线';
    } else if (plots.accuracy_curve) {
        learningCurveSrc = `data:image/png;base64,${plots.accuracy_curve}`;
        learningCurveTitle.textContent = '准确率曲线';
    }

    if (learningCurveSrc) {
        learningCurvePlot.src = learningCurveSrc;
        learningCurvePlot.alt = learningCurveTitle.textContent;
        learningCurveCard.style.display = 'block';
    } else {
        learningCurveCard.style.display = 'none';
    }

    // 预测结果图 (优先级: ROC曲线 > 混淆矩阵 > 预测vs真实)
    const predictionPlot = document.getElementById('prediction-plot');
    const predictionCard = predictionPlot.closest('.card');
    const predictionTitle = predictionCard.querySelector('h2');

    let predictionPlotSrc = null;
    
    if (plots.roc_curve) {
        predictionPlotSrc = `data:image/png;base64,${plots.roc_curve}`;
        predictionTitle.textContent = 'ROC曲线';
        predictionPlot.alt = 'ROC曲线图';
    } else if (plots.confusion_matrix) {
        predictionPlotSrc = `data:image/png;base64,${plots.confusion_matrix}`;
        predictionTitle.textContent = '混淆矩阵';
        predictionPlot.alt = '混淆矩阵图';
    } else if (plots.prediction_vs_actual) {
        predictionPlotSrc = `data:image/png;base64,${plots.prediction_vs_actual}`;
        predictionTitle.textContent = '预测值与实际值对比';
        predictionPlot.alt = '预测值与实际值对比图';
    }

    if (predictionPlotSrc) {
        predictionPlot.src = predictionPlotSrc;
        predictionCard.style.display = 'block';
    } else {
        predictionCard.style.display = 'none';
    }

    // 检查是否有额外图表需要显示
    const extraPlotsContainer = document.getElementById('extra-plots-container');
    if (extraPlotsContainer) {
        extraPlotsContainer.innerHTML = ''; // 清空容器
        
        // 添加PR曲线
        if (plots.pr_curve) {
            addExtraPlot(extraPlotsContainer, plots.pr_curve, '精确率-召回率曲线', 'graph-up');
        }
        
        // 添加残差图
        if (plots.residuals_plot) {
            addExtraPlot(extraPlotsContainer, plots.residuals_plot, '残差分布图', 'graph-down');
        }
        
        // 添加残差分布图
        if (plots.residuals_hist) {
            addExtraPlot(extraPlotsContainer, plots.residuals_hist, '残差直方图', 'bar-chart');
        }
        
        // 如果有额外图表显示容器
        if (extraPlotsContainer.children.length > 0) {
            extraPlotsContainer.style.display = 'block';
        } else {
            extraPlotsContainer.style.display = 'none';
        }
    }
}

/**
 * 添加额外的图表到容器
 */
function addExtraPlot(container, plotData, title, iconName) {
    const cardDiv = document.createElement('div');
    cardDiv.className = 'card';
    cardDiv.innerHTML = `
        <div class="section-title">
            <i class="bi bi-${iconName}"></i>
            <h2>${title}</h2>
        </div>
        <div class="plot-container">
            <img class="plot-image" src="data:image/png;base64,${plotData}" alt="${title}">
        </div>
    `;
    container.appendChild(cardDiv);
}

/**
 * 处理导出模型按钮点击
 */
async function handleExportModelClick() {
    showLoading(true);
    clearErrors();

    try {
        const response = await fetch(`${API_BASE_URL}/export-model`, {
            method: 'POST'
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP错误 ${response.status}`);
        }

        // 创建下载链接
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'exported_model_package.zip';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();

        // 显示成功消息
        showImportSuccessMessage('模型导出成功！');

    } catch (error) {
        console.error('导出模型失败:', error);
        showError('import-model-error', `导出模型失败: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

/**
 * 处理导入模型文件选择变化
 */
async function handleImportModelFileChange(event) {
    const fileInput = event.target;
    if (!fileInput.files || fileInput.files.length === 0) {
        return;
    }

    const file = fileInput.files[0];
    if (!file.name.endsWith('.zip')) {
        showError('import-model-error', '请上传 .zip 格式的模型包文件。');
        return;
    }

    clearErrors();
    hideImportSuccessMessage();
    showLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE_URL}/import-model`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || `HTTP错误 ${response.status}`);
        }

        // 显示成功消息
        showImportSuccessMessage(result.message);
        // 设置导入模型状态
        isModelImported = true;
        updateUIAfterModelStatusChange();
        updateStepIndicator(3); // 导入后准备配置新数据的特征

    } catch (error) {
        console.error('导入模型失败:', error);
        showError('import-model-error', `导入模型失败: ${error.message}`);
        isModelImported = false;
        updateUIAfterModelStatusChange();
    } finally {
        showLoading(false);
        fileInput.value = ''; // 重置文件输入框
    }
}

/**
 * 处理使用导入模型预测按钮点击
 */
async function handlePredictWithImportedModelClick() {
    clearErrors();
    hideImportSuccessMessage();

    // 验证新数据文件和列选择
    const fileInput = document.getElementById('file');
    if (!fileInput.files || fileInput.files.length === 0) {
        showError('file-error', '请为导入的模型上传新的数据文件进行预测');
        return;
    }

    const fileName = fileInput.files[0].name;
    if (!fileName.endsWith('.xlsx') && !fileName.endsWith('.xls') && !fileName.endsWith('.csv')) {
        showError('file-error', '请上传.xlsx、.xls或.csv格式的文件');
        return;
    }

    if (xColumns.length === 0) {
        showError('x-columns-error', '请为新数据选择至少一个特征列');
        return;
    }

    if (!yColumn) {
        showError('y-column-error', '请为新数据选择一个目标列用于评估');
        return;
    }

    showLoading(true);

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('x_columns', JSON.stringify(xColumns));
    formData.append('y_column', yColumn);

    try {
        const response = await fetch(`${API_BASE_URL}/predict-with-imported-model`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || `HTTP错误 ${response.status}`);
        }

        // 显示预测结果
        displayResults(result);
        updateStepIndicator(4); // 显示结果步骤

        // 平滑滚动到结果区域
        document.getElementById('results').scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });

    } catch (error) {
        console.error('使用导入模型预测失败:', error);
        showError('server-error', `预测失败: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

/**
 * 显示导入成功消息
 */
function showImportSuccessMessage(message) {
    const successElement = document.getElementById('import-model-success');
    successElement.textContent = message;
    successElement.style.display = 'block';
}

/**
 * 隐藏导入成功消息
 */
function hideImportSuccessMessage() {
    document.getElementById('import-model-success').style.display = 'none';
}

/**
 * 更新UI以反映模型状态变化
 */
function updateUIAfterModelStatusChange() {
    const trainButton = document.getElementById('train-button');
    const exportButton = document.getElementById('export-model-button');
    const predictImportedButton = document.getElementById('predict-imported-button');

    if (isModelImported) {
        trainButton.innerHTML = '<i class="bi bi-play-fill"></i> 开始新训练';
        exportButton.style.display = 'inline-block'; // 导入的模型也可以导出
        predictImportedButton.style.display = 'inline-block';
    } else {
        trainButton.innerHTML = '<i class="bi bi-play-fill"></i> 开始训练';
        predictImportedButton.style.display = 'none';
        // 不要在这里隐藏导出按钮，因为训练成功后也需要显示
        // 仅在调用handleTrainButtonClick时才隐藏，然后在训练成功后再显示
    }
}