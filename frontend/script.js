const API_BASE_URL = 'http://localhost:19123';

let fileColumns = [];
let xColumns = [];
let yColumn = '';
let isModelImported = false;

document.addEventListener('DOMContentLoaded', function () {
    fetchAvailableModels();
    document.getElementById('file').addEventListener('change', handleFileChange);
    document.getElementById('train-button').addEventListener('click', handleTrainButtonClick);
    document.getElementById('export-model-button').addEventListener('click', handleExportModelClick);
    document.getElementById('import-model-file').addEventListener('change', handleImportModelFileChange);
    document.getElementById('predict-imported-button').addEventListener('click', handlePredictWithImportedModelClick);

    document.getElementById('model').addEventListener('change', function () {
        if (this.value) {
            updateStepIndicator(2);
            
            const modelValue = this.value;
            const trainingParamsDiv = document.getElementById('training-params');
            if (modelValue.includes('nn_') || modelValue.includes('emax_')) {
                trainingParamsDiv.style.display = 'block';
            } else {
                trainingParamsDiv.style.display = 'none';
            }
        }
    });
});

function updateStepIndicator(currentStep) {
    const steps = document.querySelectorAll('.step');

    steps.forEach((step, index) => {
        const stepNum = index + 1;

        if (isModelImported && currentStep === 4) {
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

async function fetchAvailableModels() {
    try {
        console.log('正在获取模型列表...');
        const response = await fetch(`${API_BASE_URL}/api/available-models`);
        console.log('响应状态:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP错误 ${response.status}`);
        }

        const data = await response.json();
        console.log('获取到的数据:', data);
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

function populateModelSelect(models) {
    console.log('正在填充模型选择框:', models);
    const modelSelect = document.getElementById('model');

    while (modelSelect.options.length > 1) {
        modelSelect.remove(1);
    }

    for (const [id, name] of Object.entries(models)) {
        console.log(`添加模型: ${id} -> ${name}`);
        const option = document.createElement('option');
        option.value = id;
        option.textContent = name;
        modelSelect.appendChild(option);
    }
    
    console.log(`总共添加了 ${Object.keys(models).length} 个模型`);
}

async function handleFileChange(event) {
    const fileInput = event.target;

    if (!fileInput.files || fileInput.files.length === 0) {
        document.getElementById('columns-selection').style.display = 'none';
        return;
    }

    const file = fileInput.files[0];

    updateStepIndicator(1);

    if (!file.name.endsWith('.xlsx') && !file.name.endsWith('.xls') && !file.name.endsWith('.csv')) {
        showError('file-error', '请上传.xlsx、.xls或.csv格式的文件');
        return;
    }

    clearErrors();
    showLoading(true);

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/api/get-columns`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP错误 ${response.status}`);
        }

        const data = await response.json();

        fileColumns = data.columns;
        displayColumnSelections(fileColumns);
        document.getElementById('columns-selection').style.display = 'block';

        updateStepIndicator(2);

    } catch (error) {
        console.error('读取文件列名失败:', error);
        showError('file-error', '无法读取文件列名，请检查文件格式是否正确');
    } finally {
        showLoading(false);
    }
}

function displayColumnSelections(columns) {
    xColumns = [];
    yColumn = '';

    const xColumnsContainer = document.getElementById('x-columns-container');
    xColumnsContainer.innerHTML = '';

    const yColumnContainer = document.getElementById('y-column-container');
    yColumnContainer.innerHTML = '';

    updateColumnsCounter('x', 0);
    updateColumnsCounter('y', '');

    setupSearchFilter('x-search', 'x-columns-container');
    setupSearchFilter('y-search', 'y-column-container');

    document.getElementById('select-all-x').addEventListener('click', () => selectAllColumns(true));
    document.getElementById('deselect-all-x').addEventListener('click', () => selectAllColumns(false));

    columns.forEach(column => {
        // X列选项
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

        xColumnItem.addEventListener('click', function (e) {
            if (e.target !== xCheckbox) {
                xCheckbox.checked = !xCheckbox.checked;
                const event = new Event('change');
                xCheckbox.dispatchEvent(event);
            }
        });

        xCheckbox.addEventListener('change', function () {
            if (this.checked) {
                if (!xColumns.includes(column)) {
                    xColumns.push(column);
                    xColumnItem.classList.add('selected');
                }
                if (yColumn === column) {
                    document.getElementById(`y-${column}`).checked = false;
                    document.querySelector(`.column-item[data-column-name="${column}"]`).classList.remove('selected');
                    yColumn = '';
                    updateColumnsCounter('y', '');
                }

                updateColumnsCounter('x', xColumns.length);

                if (xColumns.length > 0 && yColumn) {
                    updateStepIndicator(3);
                }
            } else {
                xColumns = xColumns.filter(col => col !== column);
                xColumnItem.classList.remove('selected');

                updateColumnsCounter('x', xColumns.length);

                if (xColumns.length === 0) {
                    updateStepIndicator(2);
                }
            }
        });

        // Y列选项
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

        yColumnItem.addEventListener('click', function (e) {
            if (e.target !== yRadio) {
                yRadio.checked = true;
                const event = new Event('change');
                yRadio.dispatchEvent(event);
            }
        });

        yRadio.addEventListener('change', function () {
            if (this.checked) {
                if (yColumn) {
                    const prevItem = document.querySelector(`.column-item[data-column-name="${yColumn}"]`);
                    if (prevItem) prevItem.classList.remove('selected');
                }

                yColumn = column;
                yColumnItem.classList.add('selected');
                updateColumnsCounter('y', yColumn);

                const xCheckbox = document.getElementById(`x-${column}`);
                if (xCheckbox && xCheckbox.checked) {
                    xCheckbox.checked = false;
                    document.querySelector(`#x-columns-container .column-item[data-column-name="${column}"]`).classList.remove('selected');
                    xColumns = xColumns.filter(col => col !== column);
                    updateColumnsCounter('x', xColumns.length);
                }

                if (xColumns.length > 0 && yColumn) {
                    updateStepIndicator(3);
                }
            }
        });
    });
}

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

function selectAllColumns(select) {
    const xColumnsContainer = document.getElementById('x-columns-container');
    const visibleItems = xColumnsContainer.querySelectorAll('.column-item:not(.hidden-column)');

    visibleItems.forEach(item => {
        const checkbox = item.querySelector('.column-checkbox');
        if (checkbox && checkbox.checked !== select) {
            checkbox.checked = select;
            const event = new Event('change');
            checkbox.dispatchEvent(event);
        }
    });
}

function updateColumnsCounter(type, count) {
    if (type === 'x') {
        document.querySelector('#x-columns-counter span').textContent = count;
    } else if (type === 'y') {
        document.querySelector('#y-columns-counter span').textContent = count || '无';
    }
}

async function handleTrainButtonClick() {
    clearErrors();
    hideImportSuccessMessage();

    if (!validateInputs()) {
        return;
    }

    showLoading(true);
    const modelValue = document.getElementById('model').value;
    if (modelValue.includes('nn_') || modelValue.includes('emax_')) {
        const useEarlyStopping = document.getElementById('early-stopping').checked;
        const epochs = document.getElementById('epochs').value;
        document.querySelector('#loading p').textContent = 
            `正在训练模型... (${epochs}轮${useEarlyStopping ? '，启用早停' : '，不使用早停'})`;
    } else {
        document.querySelector('#loading p').textContent = "正在处理数据并训练模型，请稍候...";
    }

    document.getElementById('export-model-button').style.display = 'none';
    document.getElementById('predict-imported-button').style.display = 'none';
    isModelImported = false;

    const formData = new FormData();
    formData.append('file', document.getElementById('file').files[0]);
    formData.append('x_columns', JSON.stringify(xColumns));
    formData.append('y_column', yColumn);
    formData.append('model_name', document.getElementById('model').value);
    
    if (modelValue.includes('nn_') || modelValue.includes('emax_')) {
        const epochs = parseInt(document.getElementById('epochs').value);
        const useEarlyStopping = document.getElementById('early-stopping').checked;
        
        formData.append('epochs', isNaN(epochs) || epochs < 1 ? 100 : epochs);
        formData.append('use_early_stopping', String(useEarlyStopping));
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/train`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP错误 ${response.status}`);
        }

        const result = await response.json();

        displayResults(result);
        updateStepIndicator(4);
        document.getElementById('export-model-button').style.display = 'inline-block';
        updateUIAfterModelStatusChange();

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

function validateInputs() {
    let isValid = true;

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

    const modelSelect = document.getElementById('model');
    if (!modelSelect.value) {
        showError('model-error', '请选择一个模型');
        isValid = false;
    }

    if (xColumns.length === 0) {
        showError('x-columns-error', '请至少选择一个特征列');
        isValid = false;
    }

    if (!yColumn) {
        showError('y-column-error', '请选择一个目标列');
        isValid = false;
    }

    return isValid;
}

function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = message;
        element.style.display = 'block';
        element.classList.add('animate-error');
        setTimeout(() => {
            element.classList.remove('animate-error');
        }, 500);
    }
}

function clearErrors() {
    const errorElements = document.querySelectorAll('.error');
    errorElements.forEach(element => {
        element.textContent = '';
        element.style.display = 'none';
    });
}

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

function displayResults(result) {
    const resultsElement = document.getElementById('results');
    resultsElement.style.display = 'block';

    document.getElementById('metrics-table').getElementsByTagName('tbody')[0].innerHTML = '';

    if (result.metrics) {
        displayMetrics(result.metrics);
    }

    if (result.plots) {
        displayPlots(result.plots);
    }
}

function displayMetrics(metrics) {
    const metricsTable = document.getElementById('metrics-table').getElementsByTagName('tbody')[0];
    metricsTable.innerHTML = '';

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

    const displayOrder = [
        'accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity',
        'r2', 'rmse', 'mse', 'mae'
    ];

    displayOrder.forEach(key => {
        if (metrics.hasOwnProperty(key)) {
            const value = metrics[key];
            const row = metricsTable.insertRow();
            const nameCell = row.insertCell(0);
            const valueCell = row.insertCell(1);

            nameCell.textContent = metricNames[key] || key;
            valueCell.textContent = typeof value === 'number' ? value.toFixed(4) : value;

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

    for (const [key, value] of Object.entries(metrics)) {
        if (key === 'confusion_matrix') continue;
        if (displayOrder.includes(key)) continue;

        const row = metricsTable.insertRow();
        const nameCell = row.insertCell(0);
        const valueCell = row.insertCell(1);

        nameCell.textContent = metricNames[key] || key;
        valueCell.textContent = typeof value === 'number' ? value.toFixed(4) : value;
    }
}

function displayPlots(plots) {
    // 特征重要性图
    const featureImportancePlot = document.getElementById('feature-importance-plot');
    const featureImportanceCard = featureImportancePlot.closest('.card');
    const featureImportanceTitle = featureImportanceCard.querySelector('h2');

    if (plots.feature_importance) {
        featureImportancePlot.src = `data:image/png;base64,${plots.feature_importance}`;
        featureImportanceTitle.textContent = '特征重要性';
        featureImportanceCard.style.display = 'block';
        
        // 处理其他可视化变体
        const extraPlotsContainer = document.getElementById('extra-plots-container');
        
        if (plots.feature_importance_horizontal_bar) {
            addExtraPlot(extraPlotsContainer, plots.feature_importance_horizontal_bar, '特征重要性排序', 'bar-chart-steps');
        }
        
        if (plots.feature_importance_vertical_bar) {
            addExtraPlot(extraPlotsContainer, plots.feature_importance_vertical_bar, '特征重要性条形图', 'bar-chart');
        }
        
        if (plots.feature_importance_pie) {
            addExtraPlot(extraPlotsContainer, plots.feature_importance_pie, '特征重要性分布', 'pie-chart');
        }
        
        if (plots.feature_importance_correlation) {
            addExtraPlot(extraPlotsContainer, plots.feature_importance_correlation, '重要特征相关性', 'grid-3x3');
        }
        
        if (plots.feature_importance_cumulative) {
            addExtraPlot(extraPlotsContainer, plots.feature_importance_cumulative, '特征重要性累积图', 'graph-up');
        }
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

    // 预测结果图
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

    // 额外图表
    const extraPlotsContainer = document.getElementById('extra-plots-container');
    if (extraPlotsContainer) {
        extraPlotsContainer.innerHTML = '';
        
        if (plots.feature_importance_horizontal_bar) {
            addExtraPlot(extraPlotsContainer, plots.feature_importance_horizontal_bar, '特征重要性排序', 'bar-chart-steps');
        }
        
        if (plots.feature_importance_vertical_bar) {
            addExtraPlot(extraPlotsContainer, plots.feature_importance_vertical_bar, '特征重要性条形图', 'bar-chart');
        }
        
        if (plots.feature_importance_pie) {
            addExtraPlot(extraPlotsContainer, plots.feature_importance_pie, '特征重要性分布', 'pie-chart');
        }
        
        if (plots.feature_importance_correlation) {
            addExtraPlot(extraPlotsContainer, plots.feature_importance_correlation, '重要特征相关性', 'grid-3x3');
        }
        
        if (plots.feature_importance_cumulative) {
            addExtraPlot(extraPlotsContainer, plots.feature_importance_cumulative, '特征重要性累积图', 'graph-up');
        }
        
        if (plots.pr_curve) {
            addExtraPlot(extraPlotsContainer, plots.pr_curve, '精确率-召回率曲线', 'graph-up');
        }
        
        if (plots.residuals_plot) {
            addExtraPlot(extraPlotsContainer, plots.residuals_plot, '残差分布图', 'graph-down');
        }
        
        if (plots.residuals_hist) {
            addExtraPlot(extraPlotsContainer, plots.residuals_hist, '残差直方图', 'bar-chart');
        }
        
        if (extraPlotsContainer.children.length > 0) {
            extraPlotsContainer.style.display = 'block';
        } else {
            extraPlotsContainer.style.display = 'none';
        }
    }
}

function addExtraPlot(container, plotData, title, iconName) {
    const cardDiv = document.createElement('div');
    cardDiv.className = 'card';
    
    let iconClass = 'graph-up';
    
    if (title.includes('相关性') || iconName === 'grid-3x3') {
        iconClass = 'grid-3x3';
    } else if (title.includes('饼图') || title.includes('分布') || iconName === 'pie-chart') {
        iconClass = 'pie-chart';
    } else if (title.includes('条形图') || title.includes('直方图') || iconName === 'bar-chart') {
        iconClass = 'bar-chart';
    } else if (title.includes('排序') || iconName === 'bar-chart-steps') {
        iconClass = 'bar-chart-steps';
    } else if (title.includes('残差分布') || iconName === 'graph-down') {
        iconClass = 'graph-down';
    }
    
    cardDiv.innerHTML = `
        <div class="section-title">
            <i class="bi bi-${iconClass}"></i>
            <h2>${title}</h2>
        </div>
        <div class="plot-container">
            <img class="plot-image" src="data:image/png;base64,${plotData}" alt="${title}">
        </div>
    `;
    container.appendChild(cardDiv);
}

async function handleExportModelClick() {
    showLoading(true);
    clearErrors();

    try {
        const response = await fetch(`${API_BASE_URL}/api/export-model`, {
            method: 'POST'
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP错误 ${response.status}`);
        }

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

        showImportSuccessMessage('模型导出成功！');

    } catch (error) {
        console.error('导出模型失败:', error);
        showError('import-model-error', `导出模型失败: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

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
        const response = await fetch(`${API_BASE_URL}/api/import-model`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || `HTTP错误 ${response.status}`);
        }

        showImportSuccessMessage(result.message);
        isModelImported = true;
        updateUIAfterModelStatusChange();
        updateStepIndicator(3);

    } catch (error) {
        console.error('导入模型失败:', error);
        showError('import-model-error', `导入模型失败: ${error.message}`);
        isModelImported = false;
        updateUIAfterModelStatusChange();
    } finally {
        showLoading(false);
        fileInput.value = '';
    }
}

async function handlePredictWithImportedModelClick() {
    clearErrors();
    hideImportSuccessMessage();

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
        showError('y-column-error', '请选择一个目标列用于评估');
        return;
    }

    showLoading(true);

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('x_columns', JSON.stringify(xColumns));
    formData.append('y_column', yColumn);

    try {
        const response = await fetch(`${API_BASE_URL}/api/predict-with-imported-model`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || `HTTP错误 ${response.status}`);
        }

        displayResults(result);
        updateStepIndicator(4);

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

function showImportSuccessMessage(message) {
    const successElement = document.getElementById('import-model-success');
    successElement.textContent = message;
    successElement.style.display = 'block';
}

function hideImportSuccessMessage() {
    document.getElementById('import-model-success').style.display = 'none';
}

function updateUIAfterModelStatusChange() {
    const trainButton = document.getElementById('train-button');
    const exportButton = document.getElementById('export-model-button');
    const predictImportedButton = document.getElementById('predict-imported-button');

    if (isModelImported) {
        trainButton.innerHTML = '<i class="bi bi-play-fill"></i> 开始新训练';
        exportButton.style.display = 'inline-block';
        predictImportedButton.style.display = 'inline-block';
    } else {
        trainButton.innerHTML = '<i class="bi bi-play-fill"></i> 开始训练';
        predictImportedButton.style.display = 'none';
    }
}