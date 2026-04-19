**Energy Consumption Prediction & Analytics Dashboard**

This project is an end-to-end machine learning and data analytics system designed to predict and analyze building energy consumption using the ASHRAE dataset. It combines big data processing, machine learning, and interactive visualization to help facility managers detect inefficiencies and reduce energy waste.
As highlighted in the project presentation, commercial buildings consume 40% of global energy, with 30% wasted due to inefficiencies

**This project solves that by:**
Predicting expected energy usage
Comparing it with actual usage
Identifying waste and anomalies

**Key Components:**
⚡ Apache Spark – Large-scale data preprocessing
📦 Parquet Storage – Efficient columnar data format
🤖 Random Forest Model – Energy prediction
🌐 Flask API – Backend logic & inference
📊 Chart.js Dashboard – Interactive visualization

**🤖 Machine Learning Model**
Model Used: Random Forest Regressor

**📊 Dashboard Features**
🔹 KPI Cards
Total records
Average & max energy
Waste rate
Peak waste hour
**🔹 Visualizations**
📈 Actual vs Predicted Energy
📊 Waste by Hour
📅 Energy by Month
📉 Waste by Day of Week
🔥 Heatmap (Hour × Day)
🌡️ Energy vs Temperature
**🔹 Predict & Compare**
User inputs conditions
Model predicts energy
Compares with historical values
Shows deviation (%)

**📈 Results & Insights**
Successfully identified peak waste hours
Detected seasonal energy patterns
Found strong correlation between temperature and energy usage
Enabled data-driven decision-making


**How to Run**
Clone Repo
git clone https://github.com/your-username/energy-dashboard.git
cd energy-dashboard
Install Dependencies
pip install -r requirements.txt
Run Application
python app.py
Open Dashboard
http://localhost:5000

<img width="1416" height="646" alt="Screenshot 2026-04-08 at 5 25 05 AM" src="https://github.com/user-attachments/assets/bfd3b25c-1199-473f-bf36-9c439de570c4" />
