# 🏠 Multimodal House Price Prediction (Tabular + Image Data)

## 📌 Objective
The main goal of this task is to **predict house prices** using a **multimodal machine learning approach** that combines:
- **Tabular data** (bedrooms, bathrooms, area, zipcode)
- **Image data** (house images – dummy images for demonstration)

By combining both numerical and visual information, this task demonstrates how machine learning can **leverage multiple sources of information** to improve prediction accuracy.


## 🧠 Purpose & Advantages
In real-world real estate applications:
- **Numerical/tabular data alone** may not fully capture the factors affecting house prices.
- **Visual appearance of a house** can play a major role in valuation.
- **Multimodal learning** allows us to combine these two data types to create a more accurate predictive model.

**Benefits:**
1. Better predictions by fusing multiple data types.
2. Demonstrates modern deep learning techniques using CNNs for images.
3. Provides a reproducible and extendable pipeline for real-world datasets.
4. Helps understand feature interactions between tabular and image data.



## 📂 Dataset
### 🗂 Tabular Data
- Source: `HousesInfo.txt`  
- Features:
  - `bedrooms`, `bathrooms`, `area`, `zipcode`  
- Target:
  - `price`  

### 🖼 Image Data
- Colab-friendly **dummy RGB images** (128×128)  
- One image per house generated randomly  
- Demonstrates model workflow without requiring real images



## ⚙️ Methodology / Approach
1. **Data Loading & Preprocessing**
   - Clone the dataset repository
   - Load tabular data using Pandas
   - Generate dummy images
   - Scale tabular features
   - Split into training and testing sets

```python
df = pd.read_csv(meta_file, sep="\s+", header=None)
df.columns = ["bedrooms","bathrooms","area","zipcode","price"]

# Standard scaling for tabular data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_tab_scaled = scaler.fit_transform(df[["bedrooms","bathrooms","area","zipcode"]])
```
## Model Architecture

Image Branch: CNN (ResNet50) → Flatten → Dense → Dropout → Image Features

Tabular Branch: Dense layers → Tabular Features

Fusion: Concatenate image and tabular features → Dense → Output (house price)

```
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model

# Fusion
combined = Concatenate()([img_features, tab_features])
z = Dense(64, activation="relu")(combined)
output = Dense(1)(z)
model = Model(inputs=[image_input, tab_input], outputs=output)
```
## Model Training

Optimizer: Adam (1e-4)

Loss: Mean Squared Error (MSE)

Metrics: Mean Absolute Error (MAE)

Epochs: 10, Batch size: 16

## 📊 Evaluation Metrics

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)
```
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("MAE:", mae)
print("RMSE:", rmse)
```
## 📈 Visualization

Plot training vs validation MAE to monitor model performance and detect overfitting
```
import matplotlib.pyplot as plt
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Val MAE")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.show()
```
## 🚀 How to Use

1.Clone Repository
```
git clone https://github.com/your-username/multimodal-house-price-prediction.git
cd multimodal-house-price-prediction
```
2.Install Dependencies
```
pip install -r requirements.txt
```
3.Run Notebook
```
jupyter notebook notebooks/multimodal.py
```
Or run the Python script directly:
```
python src/multimodal.py
```
## 💡 Key Insights

Fusion of image and tabular data improves prediction accuracy

CNN captures visual patterns, Dense layers handle numeric features

Pipeline can be extended to real datasets with actual house images

Serves as a template for multimodal regression tasks
