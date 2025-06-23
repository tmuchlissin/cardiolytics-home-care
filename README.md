# 💓 Cardiolytics

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/backend-Flask-orange.svg)
![IoT](https://img.shields.io/badge/IoT-ESP32-green.svg)
![AI](https://img.shields.io/badge/AI-Ensemble--Learning-purple.svg)

**Integrated Cardiovascular Monitoring and Prediction System**

Cardiolytics is a web-based health monitoring system that integrates **IoT devices**, **ensemble machine learning models**, and a **document-grounded chatbot** to support cardiovascular disease prediction and homecare services.

---

## 🖼️ Overview

<img src="backend/app/static/ui.png" alt="Web Screenshot" width="100%">

> Example UI: Landing Page Cardiolytics

---

## ⚙️ Features

- ✅ Real-time blood pressure monitoring (ESP32 + EMQX + Kafka)
- 📊 Cardiovascular disease risk prediction using hybrid ensemble ML/DL
- 🤖 AI chatbot with document-based Q\&A (Gemini + Pinecone)
- 🔐 Role-based access: Patient & Admin
- 📍 Flask backend + Jinja UI + MySQL + MQTT

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/tmuchlissin/cardiolytics_home_care.git
cd cardiolytics_home_care
```

### 2. Set Up Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # Core Flask environment
pip install -r requirements_el.txt       # Optional: Ensemble model dependencies
```

### 3. Configure Database

```bash
flask db init
flask db migrate
flask db upgrade
```

### 4. Upload Firmware to ESP32

- Open `src/esp_simulator.ino` in Arduino IDE
- Flash to your ESP32 board
- Ensure MQTT credentials match your `.env` file

### 5. Set Up Node-RED Flow

- Access Node-RED at: `http://localhost:1880`
- Import flow from: `src/flows.json`

---

## 🖥️ Run the Application

```bash
export FLASK_APP=backend/app.py
export FLASK_ENV=development
flask run
```

Open your browser and visit: [http://localhost:5000](http://localhost:5000)

---

## 🧽 System Architecture

<img src="backend/app/static/workflow.png" alt="System Architecture" width="100%">

---

## 📝 License

This project is licensed under the [MIT License](LICENSE)
© 2025 [T. Muchlissin](https://github.com/tmuchlissin)

---

## 🙌 Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [Pinecone](https://www.pinecone.io/)
- [Google Gemini](https://deepmind.google/technologies/gemini/)
- [Node-RED](https://nodered.org/)
- [ESP32](https://www.espressif.com/)

---

> For research collaboration or demo requests, feel free to open an issue or contact me via GitHub.
