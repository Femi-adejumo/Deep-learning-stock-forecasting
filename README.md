# 📈 Deep Learning for Stock Market Forecasting

This repository contains the full implementation of my final-year project:  
**"Design and Implementation of a Deep Learning Model for Multi-Stock Price Forecasting and Real-Time Analysis."**

The project explores deep learning approaches (BiLSTM and Transformer) for predicting stock prices, with deployment via Streamlit for interactive forecasting.

---

## 📁 Project Structure

```
stock-forecasting-project/
├── data/            # Stock data (CSV)
├── notebooks/       # Jupyter notebooks for EDA & model development
├── models/          # Saved models (.h5, .pkl)
├── app.py           # Main Streamlit app
├── reports/         # Project report, screenshots
├── requirements.txt # Python dependencies
└── README.md        # Project overview
```

---

## 🚀 Models Implemented
- **BiLSTM (Baseline, deployed model)** – Captures sequential dependencies in stock time series.  
- **Transformer (Advanced, experimental)** – Leverages attention mechanism for long-range dependencies.  

---

## 🛠️ Tools and Libraries
- **Python**: Data analysis & modeling  
- **Pandas, NumPy**: Data wrangling  
- **Scikit-learn**: Preprocessing & evaluation metrics  
- **TensorFlow / PyTorch**: Deep learning frameworks  
- **yfinance, TA-Lib**: Financial data and technical indicators  
- **Matplotlib, Seaborn**: Visualization  
- **Streamlit**: Interactive web app deployment  

---

## ⚙️ How to Run Locally

```bash
# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

## 🌐 Live Demo  

Check out the deployed Streamlit app here:  

👉 [Stock Forecasting App](https://deep-learning-stock-forecasting-hqhpe37os3krcvzjrm9tsa.streamlit.app/)  

---

## 📌 Author
Developed by **Femi Adejumo** as a final-year Computer Science project.  

---
