# **Credit Risk Intelligence ML Pipeline**

A comprehensive enterprise-level credit risk intelligence system that predicts default probability with 87% accuracy (ROC-AUC) using advanced machine learning techniques and interactive web visualization.

___

## 🎯 Project Overview

This project provides a complete solution for credit risk assessment, featuring automated data cleaning, multiple ML models comparison, and an interactive Streamlit dashboard. The system is designed for financial institutions to evaluate loan default risk with high accuracy and reliability.

### Key Features

- **🧠 Advanced ML Pipeline**: Multiple algorithms (Logistic Regression, Random Forest, XGBoost) with automated model selection
- **🔧 Automated Data Cleaning**: Intelligent preprocessing removes irrelevant marketing variables and fixes data quality issues
- **📊 Interactive Dashboard**: Streamlit web application with real-time predictions and comprehensive visualizations
- **📈 Enterprise Performance**: 87% ROC-AUC accuracy comparable to banking industry standards
- **⚖️ Balanced Learning**: SMOTE implementation handles class imbalance for robust predictions
- **🎨 Rich Visualizations**: ROC curves, confusion matrices, feature importance, and business impact metrics

___

## 🏗️ Project Structure

```
Credit-Risk-Intelligence/
├── app.py                            # Streamlit web application
├── credit_risk_intelligence.py       # Main ML pipeline script
├── credit_risk_intelligence.ipynb    # Jupyter notebook version
├── requirements.txt                  # Python dependencies
├── dataset_banco.csv                 # Banking dataset
├── utils/                            # Utility modules
│   ├── __init__.py
│   ├── data_processor.py             # Data cleaning and processing
│   ├── model_loader.py               # Model training and management
│   └── visualizations.py             # Plotting and visualization
├── LICENSE                           # MIT license
├── README.md                         # Documentation
└── CONTRIBUTING.md                   # Contributing guidelines
```

___

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Sol-so-special/Credit-Risk-Intelligence
cd Credit-Risk-Intelligence
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Interactive Web Dashboard (Recommended)
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser.

#### Option 2: Command Line Pipeline
```bash
python credit_risk_intelligence.py
```

#### Option 3: Jupyter Notebook
```bash
jupyter notebook credit_risk_intelligence.ipynb
```

___

## 📊 Dataset Information

The system uses a cleaned banking dataset with the following features:

### Credit-Relevant Features (8)
- **age**: Customer age (18-100 years)
- **job**: Occupation type (admin, blue-collar, management, etc.)
- **marital**: Marital status (married, single, divorced)
- **education**: Education level (primary, secondary, tertiary, unknown)
- **default**: Credit default history (target variable)
- **balance**: Average yearly account balance (euros)
- **housing**: Housing loan status (yes/no)
- **loan**: Personal loan status (yes/no)

### Automated Data Cleaning
The system automatically removes 9 irrelevant marketing columns:
- contact, day, month, duration, campaign, pdays, previous, poutcome, y

Additional cleaning operations:
- Age values > 100 replaced with mean age
- Marital status standardization ('div.' → 'divorced')
- Education standardization ('sec.' → 'secondary', 'UNK' → 'unknown')
- Removal of null/empty values

___

## 🔬 Technical Architecture

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Smart categorical encoding (OneHot for low cardinality, Target encoding for high cardinality)
   - Numerical scaling with StandardScaler
   - Missing value imputation

2. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Oversampling Technique)
   - Applied only to training data to prevent data leakage

3. **Model Training & Selection**
   - 3-fold stratified cross-validation
   - Multiple algorithms comparison
   - Automated best model selection based on ROC-AUC

4. **Evaluation Metrics**
   - ROC-AUC (primary metric)
   - Precision, Recall, F1-Score
   - Confusion Matrix
   - Model stability analysis

___

## 🎮 Using the Dashboard

### 1. Overview Page
- Model performance summary
- Dataset statistics
- Data quality metrics
- Business impact overview

### 2. Model Performance
- Interactive model comparison charts
- ROC curves and confusion matrices
- Feature importance analysis
- Probability distribution visualization

### 3. Risk Predictor
- Individual customer risk assessment
- Real-time probability calculation
- Risk level interpretation
- Business recommendations

### 4. Dataset Analysis
- Comprehensive data exploration
- Feature distribution plots
- Data quality assessment
- Statistical summaries

___

## 💼 Business Applications

### Risk Assessment Thresholds
- **🟢 Low Risk (0-20%)**: Automatic approval recommended
- **🟡 Medium Risk (20-40%)**: Manual review required
- **🔴 High Risk (40%+)**: Likely rejection

### Use Cases
- **Loan Origination**: Automated initial screening
- **Portfolio Management**: Risk monitoring and analysis
- **Regulatory Compliance**: Documentation and audit trails
- **Business Intelligence**: Integration with existing systems

___

## 🔧 Configuration

### Environment Variables
Create a `.env` file for custom configurations:
```bash
# Dataset path
DATASET_PATH=dataset_banco.csv

# Model parameters
TEST_SIZE=0.25
RANDOM_STATE=42
CV_FOLDS=3

# Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Model Hyperparameters
Modify model parameters in `utils/model_loader.py`:
```python
models = {
    'XGBoost': XGBClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
}
```

___

## 📈 API Usage

### Programmatic Access
```python
from utils.model_loader import train_and_cache_models, predict_single_case
from utils.data_processor import load_and_process_dataset

# Load and clean data
df, info = load_and_process_dataset("dataset_banco.csv")

# Train models
model_data = train_and_cache_models(df)

# Make prediction
input_data = {
    'age': 35,
    'job': 'management',
    'balance': 1500.0,
    # ... other features
}

probability, prediction = predict_single_case(
    model_data['best_model'],
    model_data['preprocessor'],
    input_data
)
```

___

## 🧪 Testing

Run the test suite:
```bash
# Unit tests
python -m pytest tests/

# Integration tests
python tests/test_integration.py

# Performance validation
python tests/test_performance.py
```

___

## 📦 Dependencies

### Core Libraries
- **pandas** (≥2.0.0): Data manipulation and analysis
- **scikit-learn** (≥1.3.0): Machine learning algorithms
- **xgboost** (≥1.7.0): Gradient boosting framework
- **streamlit** (≥1.28.0): Web application framework

### Visualization
- **plotly** (≥5.15.0): Interactive plotting
- **matplotlib** (≥3.7.0): Static plotting
- **seaborn** (≥0.12.0): Statistical visualization

### Advanced ML
- **imbalanced-learn** (≥0.11.0): Class imbalance handling
- **category-encoders** (≥2.5.0): Categorical encoding

See `requirements.txt` for complete dependency list.

___

## 🔍 Troubleshooting

### Common Issues

1. **Import Error**: Ensure all dependencies are installed
```bash
pip install --upgrade -r requirements.txt
```

2. **Dataset Not Found**: Verify `dataset_banco.csv` is in the project root
```bash
ls -la dataset_banco.csv
```

3. **Memory Issues**: Reduce dataset size for testing
```python
df_sample = df.sample(n=10000, random_state=42)
```

4. **Port Already in Use**: Change Streamlit port
```bash
streamlit run app.py --server.port 8502
```

### Performance Optimization

- Use `@st.cache_data` and `@st.cache_resource` for caching
- Implement data sampling for large datasets
- Consider model serialization for production deployment

___

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

___

## 🙏 Acknowledgments

- Banking dataset providers
- Scikit-learn community
- Streamlit team
- Open source ML community

___

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section
- Review the contributing guidelines

---

**Built with ❤️ for the financial technology community**