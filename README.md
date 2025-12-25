# 🚕🌱 EquiRide Surge AI  
### Fair & Sustainable Ride Repositioning System

**EquiRide Surge AI** is an intelligent ride-hailing optimization platform that predicts **hyperlocal demand surges** and proactively reallocates drivers to balance **profitability, fairness, and environmental sustainability**. The system addresses supply–demand mismatches that lead to long wait times, unfair driver earnings, and increased carbon emissions from inefficient cruising.

---

## 🚀 Key Features
- **Hyperlocal Demand Forecasting:** Micro-zone (500m–1km) demand prediction for fine-grained spatial accuracy.
- **Hybrid Model Architecture:** Combines **LightGBM** for tabular learning with **Graph WaveNet** for spatial-temporal modeling.
- **Fairness-Aware Optimization:** Multi-objective optimization using the **Hungarian Algorithm** to balance profit, emissions, and driver equity.
- **Real-Time Driver Alerts:** Automated surge alerts triggered by a **2-sigma threshold**, delivered via **Twilio SMS**.
- **Interactive Dashboards:** Actionable insights for drivers and operators to enable smarter urban mobility decisions.

---

## 🛠️ Tech Stack & Methodology
- **Machine Learning:** LightGBM, Graph WaveNet, ST-GCN  
- **Operational Research:** Hungarian Algorithm  
- **Communication:** Twilio API  
- **Development:** Modular Python codebase with scalable APIs  

---

## 📊 Performance Metrics
Experimental evaluation on simulated spatio-temporal datasets shows significant improvements:

- **Graph WaveNet (Final):** Test MAE = **34.28**
- **ST-GCN (Baseline):** Test MAE = **47.59**
- **LightGBM (Standalone):** Test MAE = **47.51**

---

## 👥 Contributors

- **Dhatri P Sriram**
- **Harshini H** 
- **Namratha A** 
- **Bhumika L** 

---

## 📜 License
This project is released for academic and research purposes.
