⚡ Industrial Power Optimizer (IPO)
AI-Based Peak Demand Forecasting & Load Scheduling for Manufacturing
![alt text](https://img.shields.io/badge/ClickUp-Project_Management-crimson)

![alt text](https://img.shields.io/badge/Weights_%26_Biases-Tracking-orange)

![alt text](https://img.shields.io/badge/Streamlit-Dashboard-blue)

![alt text](https://img.shields.io/badge/DVC-Data_Versioning-grey)
📌 Overview
Industrial manufacturing facilities face massive financial penalties due to "Peak Demand Charges." The Industrial Power Optimizer (IPO) is a full-stack AI solution that uses Deep Learning to forecast energy surges and provides Prescriptive Recommendations to shift loads, directly reducing operational costs.
🚀 Professional Infrastructure (Agile & MLOps)
This project was developed using industry-standard tools to ensure reproducibility and organized delivery:
Project Management (ClickUp): Utilized ClickUp for Sprint planning, task breakdown, and tracking the 45-day development roadmap.
Data Version Control (DVC): Managed dataset lineage, ensuring that every model version is linked to a specific state of the processed data.
Experiment Tracking (Weights & Biases): Logged 40+ training epochs, monitoring real-time loss curves and hyperparameter performance to prevent overfitting.
🛠️ Tech Stack
Deep Learning: TensorFlow, Keras (Stacked LSTM)
MLOps & Management: ClickUp, DVC, WandB
Interface: Streamlit, Plotly (Interactive Charts)
Data Engineering: Pandas, NumPy, Scikit-Learn
Environment: VS Code, Git, Python 3.11
📊 The ML Pipeline
Data Ingestion: Automated pipeline fetching 35,040 records from the UCI Steel Industry dataset.
Feature Engineering: Implemented Cyclical Temporal Encoding (Sine/Cosine) for 15-minute intervals and day-of-week patterns.
Windowing: Transformed 2D data into 3D Tensors using a 24-hour (96-step) sliding history window.
Model Architecture:
2-Layer Stacked LSTM (64/32 units) with Dropout (0.2) for regularization.
Dense layer with ReLU activation for non-linear pattern recognition.
Optimization: Trained with EarlyStopping (restoring best weights from Epoch 36) to achieve maximum generalization.
📉 Results & Business Impact
Forecasting Accuracy: ~81% (0.026 MAE) on unseen "Future" test data (Nov-Dec).
Operational Savings: Identified 44 critical peaks in a 30-day window.
ROI: Calculated a potential $2,756.28 monthly saving for the facility through the implemented load-shifting strategy.
💻 Installation & Usage
Clone & Setup:
code
Bash
git clone https://github.com/ahmadnazir2006/Industrial-Power-Optimizer.git
cd Industrial-Power-Optimizer
pip install -r requirements.txt
Run Dashboard:
code
Bash
streamlit run app/main.py
👤 Author
Ahmad Nazir
BS Computer Science | FAST-NUCES, Lahore
LinkedIn | GitHub
This project demonstrates a complete AI lifecycle from data versioning to prescriptive analytics.
