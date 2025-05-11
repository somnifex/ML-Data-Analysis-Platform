# ML Data Analysis Platform

*Read this in [中文 (Chinese)](README_zh_CN.md)*

A demo for a machine learning data analysis platform – not overly powerful, but designed to allow users to train and evaluate common ML models without any coding. Simply upload your data, select a model, and get professional visualizations and metrics analysis in seconds. This demo isn't exhaustive, and we encourage users to modify it to suit their specific needs.

## Features

- 🚀 **No-code Interface**: User-friendly interface for model selection and configuration
- 📊 **Data Visualization**: Automatically generates model performance visualizations
- 📈 **Feature Analysis**: Provides feature importance analysis
- 🔄 **Multiple ML Models**: Supports various classification and regression models
- 📁 **Data Flexibility**: Works with Excel (.xlsx, .xls) and CSV files
- 📱 **Responsive Design**: Works on both desktop and mobile devices

## Getting Started

### Prerequisites

- Developed with Python 3.12 and pip (other environments not tested)
- Modern web browser (Chrome, Firefox, Edge, etc.)

### Installation

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/ml-data-analysis.git
   cd ml-data-analysis
   ```
2. Install the required Python packages

   ```bash
   pip install -r requirements.txt
   ```
3. Start the backend server

   ```bash
   cd backend
   python main.py
   ```
4. Open the frontend in your browser

   - Navigate to `frontend/index.html` in your file explorer and open it with your browser
   - Or run a simple HTTP server:
     ```bash
     cd frontend
     python -m http.server 8000
     ```

     Then visit `http://localhost:8000` in your browser

## Usage

1. **Upload Data**: Use the file upload button to upload your Excel or CSV data file
2. **Select Model**: Choose from various ML models depending on your task type
3. **Configure Features**: Select the feature columns and target variable
4. **Train Model**: Click the "Start Training" button and wait for results
5. **Review Results**: Explore the generated metrics and visualizations

## Supported Models

### Classification Models

- Neural Network Classifier
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- Gradient Boosting Classifier
- Support Vector Machine Classifier
- K-Nearest Neighbors Classifier
- XGBoost Classifier (if installed)
- LightGBM Classifier (if installed)

### Regression Models

- Neural Network Regressor
- Linear Regression
- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Support Vector Machine Regressor
- K-Nearest Neighbors Regressor
- Elastic Net Regression
- XGBoost Regressor (if installed)
- LightGBM Regressor (if installed)

## Project Structure

```
ml-data-analysis/
├── backend/               # Backend API server
│   ├── main.py            # FastAPI server entry point
│   ├── models.py          # ML model implementations
│   └── processing.py      # Data processing utilities
├── frontend/              # Frontend web interface
│   ├── index.html         # Main HTML page
│   ├── script.js          # JavaScript logic
│   └── assets/            # Images and resources
└── gendata.py             # Data generation utility script
```

## Optional: Generate Test Data

You can generate test data for various ML scenarios using the included script:

```bash
python gendata.py
```

This will create Excel files with synthetic data for classification, regression, and clustering tasks in a `ml_test_data` directory.

## License

This project is licensed under the GNU General Public License v3.0 - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Uses [scikit-learn](https://scikit-learn.org/) and [PyTorch](https://pytorch.org/) for machine learning
- Visualization powered by [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
