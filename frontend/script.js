const API_BASE_URL = 'http://localhost:19123';

// 保存文件中的列名
let fileColumns = [];
// 保存已选择的特征列
let xColumns = [];
// 保存选择的目标列
let yColumn = '';

document.addEventListener('DOMContentLoaded', function () {
    fetchAvailableModels();
    document.getElementById('file').addEventListener('change', handleFileChange);
    document.getElementById('train-button').addEventListener('click', handleTrainButtonClick);

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

        if (stepNum < currentStep) {
            step.classList.remove('active');
            step.classList.add('completed');
        } else if (stepNum === currentStep) {
            step.classList.add('active');
            step.classList.remove('completed');
        } else {
            step.classList.remove('active', 'completed');
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
        xColumnItem.addEventListener('click', function(e) {
            // 如果点击的是复选框本身，不需要额外处理
            if (e.target !== xCheckbox) {
                xCheckbox.checked = !xCheckbox.checked;
                // 手动触发change事件
                const event = new Event('change');
                xCheckbox.dispatchEvent(event);
            }
        });
        
        xCheckbox.addEventListener('change', function() {
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
        yColumnItem.addEventListener('click', function(e) {
            // 如果点击的是单选框本身，不需要额外处理
            if (e.target !== yRadio) {
                yRadio.checked = true;
                // 手动触发change事件
                const event = new Event('change');
                yRadio.dispatchEvent(event);
            }
        });
        
        yRadio.addEventListener('change', function() {
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
    
    searchInput.addEventListener('input', function() {
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

    // 验证输入
    if (!validateInputs()) {
        return;
    }

    // 显示加载状态
    showLoading(true);

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
            throw new Error(`HTTP错误 ${response.status}`);
        }

        const result = await response.json();

        // 处理结果
        displayResults(result);

        // 更新步骤指示器
        updateStepIndicator(4);

        // 平滑滚动到结果区域
        document.getElementById('results').scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    } catch (error) {
        console.error('训练请求失败:', error);
        showError('server-error', '服务请求失败，请检查服务是否启动或稍后重试。');
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

    // 显示指标
    displayMetrics(result.metrics);

    // 显示图表
    displayPlots(result.plots);
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
        'r2': 'R²决定系数',
        'mse': '均方误差(MSE)',
        'mae': '平均绝对误差(MAE)'
    };

    // 添加每个指标行
    for (const [key, value] of Object.entries(metrics)) {
        if (key === 'confusion_matrix') continue; // 混淆矩阵在图表中展示

        const row = metricsTable.insertRow();
        const nameCell = row.insertCell(0);
        const valueCell = row.insertCell(1);

        nameCell.textContent = metricNames[key] || key;

        // 格式化数值（保留4位小数）
        valueCell.textContent = typeof value === 'number' ? value.toFixed(4) : value;

        // 为良好的指标值添加颜色
        if ((key === 'accuracy' || key === 'precision' || key === 'recall' || key === 'f1' || key === 'r2') && value > 0.7) {
            valueCell.style.color = 'var(--success-color)';
            valueCell.style.fontWeight = 'bold';
        }
    }
}

/**
 * 显示图表
 */
function displayPlots(plots) {
    // 特征重要性图
    if (plots.feature_importance) {
        document.getElementById('feature-importance-plot').src = `data:image/png;base64,${plots.feature_importance}`;
    }

    // 学习曲线图
    if (plots.learning_curve) {
        document.getElementById('learning-curve-plot').src = `data:image/png;base64,${plots.learning_curve}`;
    } else if (plots.accuracy_curve) {
        document.getElementById('learning-curve-plot').src = `data:image/png;base64,${plots.accuracy_curve}`;
    }

    // 预测结果图
    if (plots.confusion_matrix) {
        document.getElementById('prediction-plot').src = `data:image/png;base64,${plots.confusion_matrix}`;
    } else if (plots.roc_curve) {
        document.getElementById('prediction-plot').src = `data:image/png;base64,${plots.roc_curve}`;
    } else if (plots.prediction_vs_actual) {
        document.getElementById('prediction-plot').src = `data:image/png;base64,${plots.prediction_vs_actual}`;
    }
}