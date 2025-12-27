# ============================================================
# TASK 3: Multimodal House Price Prediction
# Using HousesInfo.txt + Dummy Images (Colab-ready)
# ============================================================

# -----------------------------
# 1️⃣ INSTALL LIBRARIES
# -----------------------------
!pip install -q pandas numpy scikit-learn tensorflow matplotlib pillow gitpython

# -----------------------------
# 2️⃣ IMPORT LIBRARIES
# -----------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from git import Repo

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

# -----------------------------
# 3️⃣ CLONE THE REPOSITORY (Metadata only)
# -----------------------------
REPO_URL = "https://github.com/emanhamed/Houses-dataset"
if not os.path.exists("Houses-dataset"):
    Repo.clone_from(REPO_URL, "Houses-dataset")

# -----------------------------
# 4️⃣ LOAD TABULAR METADATA
# -----------------------------
meta_file = "Houses-dataset/Houses Dataset/HousesInfo.txt"
columns = ["bedrooms","bathrooms","area","zipcode","price"]
df = pd.read_csv(meta_file, sep="\s+", header=None)
df.columns = columns
print("Metadata shape:", df.shape)
print(df.head())

# -----------------------------
# 5️⃣ GENERATE DUMMY IMAGES
# -----------------------------
num_samples = len(df)
img_height, img_width = 128, 128
np.random.seed(42)

# Random RGB images for each house
images = np.random.randint(0, 256, size=(num_samples, img_height, img_width, 3), dtype=np.uint8)
print("Dummy images shape:", images.shape)

# -----------------------------
# 6️⃣ TABULAR FEATURES
# -----------------------------
X_tab = df[["bedrooms","bathrooms","area","zipcode"]].values
y = df["price"].values

# Standard scaling
scaler = StandardScaler()
X_tab_scaled = scaler.fit_transform(X_tab)

# -----------------------------
# 7️⃣ TRAIN-TEST SPLIT
# -----------------------------
X_img_train, X_img_test, X_tab_train, X_tab_test, y_train, y_test = train_test_split(
    images, X_tab_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# 8️⃣ CNN BRANCH FOR IMAGE FEATURES
# -----------------------------
image_input = Input(shape=(img_height, img_width, 3))
base_cnn = ResNet50(weights=None, include_top=False, input_tensor=image_input)
x = Flatten()(base_cnn.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
img_features = Dense(64, activation="relu")(x)

# -----------------------------
# 9️⃣ TABULAR BRANCH
# -----------------------------
tab_input = Input(shape=(X_tab_train.shape[1],))
t = Dense(64, activation="relu")(tab_input)
t = Dense(32, activation="relu")(t)

# -----------------------------
# 🔟 FUSION + OUTPUT
# -----------------------------
combined = Concatenate()([img_features, t])
z = Dense(64, activation="relu")(combined)
z = Dense(32, activation="relu")(z)
output = Dense(1)(z)

# -----------------------------
# 1️⃣1️⃣ COMPILE MODEL
# -----------------------------
model = Model(inputs=[image_input, tab_input], outputs=output)
model.compile(optimizer=Adam(1e-4), loss="mse", metrics=["mae"])

# -----------------------------
# 1️⃣2️⃣ TRAIN MODEL
# -----------------------------
history = model.fit(
    [X_img_train, X_tab_train],
    y_train,
    validation_data=([X_img_test, X_tab_test], y_test),
    epochs=10,
    batch_size=16
)

# -----------------------------
# 1️⃣3️⃣ EVALUATION
# -----------------------------
# -----------------------------
y_pred = model.predict([X_img_test, X_tab_test])
mae = mean_absolute_error(y_test, y_pred)

# RMSE manually
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n✅ Evaluation Metrics:")
print("MAE:", mae)
print("RMSE:", rmse)


# -----------------------------
# 1️⃣4️⃣ PLOT METRICS
# -----------------------------
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Val MAE")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.show()
