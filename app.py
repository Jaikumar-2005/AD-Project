#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

app = Flask(__name__)

# Load dataset
csv_path = "World_Development_Dataset.csv"
df = pd.read_csv(csv_path)

# Select only numeric features for clustering
features = df.select_dtypes(include=[np.number]).columns.tolist()

# Train Agglomerative Clustering model
n_clusters = 3  # Change based on your dataset
model = AgglomerativeClustering(n_clusters=n_clusters)
df["Cluster"] = model.fit_predict(df[features])  # Assign clusters

# Compute approximate cluster centers by averaging feature values
cluster_centers = df.groupby("Cluster").mean().values

@app.route("/")
def home():
    return render_template("home.html", features=features)

@app.route("/result", methods=["POST"])
def result():
    try:
        # Get user input for all features
        user_data = [float(request.form.get(feature, 0)) for feature in features]
        user_data = np.array(user_data).reshape(1, -1)

        # Find nearest cluster center
        distances = cdist(user_data, cluster_centers)
        cluster = np.argmin(distances)  # Assign to the closest cluster

        return render_template("result.html", user_input=user_data.tolist(), cluster=cluster)
    except Exception as e:
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)



# In[ ]:




