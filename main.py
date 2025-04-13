import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sqlite3
import datetime
import base64

# Load models
leak_detection_model = joblib.load('leak_detection_model.pkl')
leak_rate_model = joblib.load('leak_rate_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize database
def init_db():
    """Create the SQLite database and table if it doesn't exist."""
    conn = sqlite3.connect("leak_alerts.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    CH4L REAL,
                    CH4R REAL,
                    P REAL,
                    RsL REAL,
                    RsR REAL,
                    leak_rate REAL
                )''')
    conn.commit()
    conn.close()

init_db()

# Function to set background image
def set_bg(image_file):
    """Set background image using base64 encoding."""
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# Call function to set background image
set_bg("image.jpeg")  # Ensure image.jpeg is in the same directory

# Streamlit UI

st.title("Gas Leak Prediction System")
st.markdown("Enter sensor values to predict gas leak and estimated leak rate.")

# User inputs
CH4L = st.number_input("CH4L (Methane Level Left)", min_value=0.0, format="%.2f")
CH4R = st.number_input("CH4R (Methane Level Right)", min_value=0.0, format="%.2f")
P = st.number_input("P (Pressure)", min_value=0.0, format="%.2f")
RsL = st.number_input("RsL (Resistance Left)", min_value=0.0, format="%.2f")
RsR = st.number_input("RsR (Resistance Right)", min_value=0.0, format="%.2f")

def save_alert(CH4L, CH4R, P, RsL, RsR, leak_rate):
    """Save a leak detection alert in SQLite."""
    conn = sqlite3.connect("leak_alerts.db")
    c = conn.cursor()
    c.execute("INSERT INTO alerts (timestamp, CH4L, CH4R, P, RsL, RsR, leak_rate) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), CH4L, CH4R, P, RsL, RsR, leak_rate))
    conn.commit()
    conn.close()

def get_alerts():
    """Fetch stored alerts from the database."""
    conn = sqlite3.connect("leak_alerts.db")
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
    alerts = c.fetchall()
    conn.close()
    return alerts

if st.button("Predict"):
    test_input = pd.DataFrame([[CH4L, CH4R, P, RsL, RsR]], columns=['CH4L', 'CH4R', 'P', 'RsL', 'RsR'])
    test_input = scaler.transform(test_input)

    leak_prediction = leak_detection_model.predict(test_input)[0]

    if leak_prediction == 1:
        leak_rate = leak_rate_model.predict(test_input)[0]
        st.error(f"🚨 Leak Detected! Estimated Leak Rate: {leak_rate:.2f}")
        save_alert(CH4L, CH4R, P, RsL, RsR, leak_rate)
    else:
        st.success("✅ No Leak Detected.")

# Display alerts
st.subheader("📜 Leak Alert History")
alerts = get_alerts()

if alerts:
    df_alerts = pd.DataFrame(alerts, columns=["ID", "Timestamp", "CH4L", "CH4R", "P", "RsL", "RsR", "Leak Rate"])
    st.dataframe(df_alerts)
else:
    st.info("No alerts recorded yet.")

def clear_alerts():
    """Delete all records from the alerts table."""
    conn = sqlite3.connect("leak_alerts.db")
    c = conn.cursor()
    c.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()

if st.button("🗑️ Clear Alert History"):
    clear_alerts()
    st.warning("All alerts have been cleared!")

st.markdown("</div>", unsafe_allow_html=True)  # Close the container div

# Add FAQ Section
st.markdown("---")  # Adds a horizontal divider
st.markdown(
    """
    <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 1px solid #ddd;">
    <h2 style="text-align: center; color: #333;">❓ Frequently Asked Questions (FAQ)</h2>
    """,
    unsafe_allow_html=True,
)

faq_data = {
    "What is the purpose of this project?": 
    "This project aims to detect leaks using sensor data and predict the leak rate if a leak is detected. It utilizes machine learning models trained on features such as Concentration, Pressure, and Resistance.",
    
    "Which machine learning models are used in this project?": 
    "The project uses a **RandomForestClassifier** for binary leak detection and a **RandomForestRegressor** for leak rate prediction.",
    
    "How does the model handle missing values and outliers?": 
    "The dataset is preprocessed by dropping null values. Outliers are detected using the **Interquartile Range (IQR) method** and removed to improve model performance.",
    
    "How does a user provide input for real-time predictions?": 
    "The program prompts the user to enter sensor values (CH4L, CH4R, P, RsL, RsR). These values are scaled and fed into the trained models to determine if a leak is present and estimate the leak rate.",
    
    "How are the results visualized?": 
    "The project plots a graph comparing actual vs. predicted leak rates using **Matplotlib**, helping to evaluate the model’s performance visually."
}

# Display each FAQ in an expandable section
for question, answer in faq_data.items():
    with st.expander(question):
        st.write(answer)

st.markdown("</div>", unsafe_allow_html=True)  # Close the FAQ container div
