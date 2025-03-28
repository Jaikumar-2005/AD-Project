#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained clustering model
model = joblib.load("model.pkl")

# Load dataset to get feature names
csv_path = "World_Development_Dataset.csv"
df = pd.read_csv(csv_path)
features = df.columns.tolist()  # Extract feature names

@app.route("/")
def home():
    return render_template("home.html", features=features)

@app.route("/result", methods=["POST"])
def result():
    try:
        # Get user input for all features
        user_data = [float(request.form.get(feature, 0)) for feature in features]
        user_data = np.array(user_data).reshape(1, -1)
        
        # Predict cluster
        cluster = model.predict(user_data)[0]
        
        return render_template("result.html", user_input=user_data.tolist(), cluster=cluster)
    except Exception as e:
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


# In[ ]:




