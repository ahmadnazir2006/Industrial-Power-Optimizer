# ⚡ Industrial Power Optimizer (IPO)
### AI-Based Peak Demand Forecasting & Load Scheduling for Manufacturing

[![ClickUp](https://img.shields.io/badge/ClickUp-Project_Management-crimson?style=for-the-badge&logo=clickup)](https://sharing.clickup.com/90181392008/t/86exfd4t3/simulatereal-timeenergysensorstreamingin-streamlit)
[![WandB](https://img.shields.io/badge/Weights_%26_Biases-Live_Training-orange?style=for-the-badge&logo=weightsandbiases)](https://wandb.ai/ahmad823-fast-nuces/Industrial%20Power%20Optimizer/runs/i6hkfu1l?nw=nwuserahmad823)
![DVC](https://img.shields.io/badge/DVC-Data_Versioning-grey?style=for-the-badge&logo=dataversioncontrol)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-blue?style=for-the-badge&logo=streamlit)

---

## 📌 Overview
Industrial manufacturing facilities are subject to **"Peak Demand Charges,"** where utility companies charge high premiums for maximum power consumed during short windows. 

The **Industrial Power Optimizer (IPO)** is a full-stack AI solution that uses Deep Learning to forecast energy surges and provides **Prescriptive Recommendations** to shift loads, directly reducing operational costs.

## 🚀 Professional Infrastructure (Agile & MLOps)
This project follows an industry-standard lifecycle to ensure transparency and reproducibility:
*   **Agile Project Management:** Tasks and sprints were tracked using **ClickUp**. [View Project Roadmap here.](https://sharing.clickup.com/90181392008/t/86exfd4t3/simulatereal-timeenergysensorstreamingin-streamlit)
*   **Experiment Tracking:** Training metrics, loss curves, and hyperparameters were logged in real-time via **Weights & Biases**. [View Live Training Logs here.](https://wandb.ai/ahmad823-fast-nuces/Industrial%20Power%20Optimizer/runs/i6hkfu1l?nw=nwuserahmad823)
*   **Data Versioning:** Utilized **DVC** to track dataset lineage, ensuring every model version is linked to a specific state of processed data.

---

## 🛠️ Tech Stack
| Category | Tools |
| :--- | :--- |
| **Deep Learning** | TensorFlow, Keras (Stacked LSTM) |
| **MLOps & Management** | ClickUp, DVC, Weights & Biases |
| **Interface** | Streamlit, Plotly (Interactive Charts) |
| **Data Engineering** | Pandas, NumPy, Scikit-Learn |

---

## 📊 The ML Pipeline
1.  **Data Ingestion:** Automated pipeline fetching 35,040 records from the UCI Steel Industry dataset.
2.  **Feature Engineering:** Implemented **Cyclical Temporal Encoding** (Sine/Cosine) for 15-minute intervals.
3.  **Windowing:** Transformed 2D data into **3D Tensors** using a 24-hour (96-step) sliding history window.
4.  **Model Architecture:** 2-Layer Stacked LSTM (64/32 units) with Dropout (0.2) for regularization.
5.  **Optimization:** Trained with `EarlyStopping` (restoring best weights from Epoch 36) for maximum generalization.

---

## 📉 Results & Business Impact
*   **Forecasting Accuracy:** ~81% (0.026 MAE) on unseen "Future" test data (Nov-Dec).
*   **Operational Savings:** Identified **44 critical peaks** in a 30-day window.
*   **ROI:** Calculated a potential **$2,756.28 monthly saving** through automated load-shifting recommendations.

---

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/ahmadnazir2006/Industrial-Power-Optimizer.git
cd Industrial-Power-Optimizer
2. Setup Environment
code
Bash
pip install -r requirements.txt
3. Run the Dashboard
code
Bash
streamlit run app/main.py
👤 Author
Ahmad Nazir
BS Computer Science | FAST-NUCES, Lahore
LinkedIn | GitHub
