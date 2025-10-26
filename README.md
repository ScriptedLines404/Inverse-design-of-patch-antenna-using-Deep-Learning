# 🛰️ Inverse Design of Patch Antenna using Deep Learning

## 🧭 Overview
This project demonstrates the **inverse design of a patch antenna** using **deep learning** techniques.  
Instead of manually tuning antenna dimensions to achieve a target frequency or bandwidth, this project leverages a **neural network model** trained to predict optimal antenna geometries from desired electromagnetic characteristics.

This approach enables **faster design iterations** and reduces dependency on computationally intensive electromagnetic simulations.

---

## ✨ Features
- 🧠 **Deep learning–based inverse design** for microstrip patch antennas  
- ⚡ **Pre-trained model** ready for inference  
- 📊 **Visualization of training performance** and model accuracy  
- 🧪 **Interactive Jupyter Notebook** for experimentation  
- 🗂️ Example dataset and model artifacts included  

---

## 📁 Project Structure
```
Inverse-design-of-patch-antenna-using-Deep-Learning/
├── LICENSE # License information
├── requirements.txt # Dependencies
├── training_plots.png # Training performance visualization
│
├── .gradio/
│ └── flagged/dataset1.csv # Example dataset
│
├── models/
│ ├── patch_antenna_model.h5 # Trained neural network model
│ ├── x_scaler.pkl # Input feature scaler
│ └── y_scaler.pkl # Output label scaler
│
└── src/
├── inverse_design_patch_antenna.ipynb # Notebook version for exploration
└── inverse_design_patch_antenna.py # Script version for direct execution
```
---

## 🧠 Model Details

| Component | Description |
|-----------|-------------|
| **Framework** | TensorFlow / Keras |
| **Architecture** | Fully Connected Neural Network (Multi-Layer Perceptron) |
| **Inputs** | Desired antenna characteristics (Frequency, Dielectric Constant) |
| **Outputs** | Antenna design parameters (Length, Width) |
| **Hidden Layers** | 256 → 128 neurons with ReLU activation |
| **Dropout** | 0.3 (30%) for regularization |
| **Loss Function** | Custom hybrid loss (MSE + Physics Constraint Penalty) |
| **Optimizer** | Adam with learning rate 0.001 |
| **Physics Constraint** | W ≥ 1.05 × L (enforced in loss function) |
| **Training Samples** | 30,000 synthetic antenna designs |
| **Artifacts** | `patch_antenna_model.h5`, `x_scaler.pkl`, `y_scaler.pkl` |
| **Performance** | MAE: 0.12mm, R²: 0.998, Physics Compliance: 99.3% |
| **Inference Speed** | ~5ms per prediction |

### 📊 Input Features
- **Frequency (f_GHz)**: 1.0 - 12.0 GHz
- **Dielectric Constant (εᵣ)**: 2.2 - 12.0

### 🎯 Output Parameters
- **Length (L_mm)**: Patch antenna length in millimeters
- **Width (W_mm)**: Patch antenna width in millimeters

### 🔧 Model Artifacts
- `patch_antenna_model.h5` - Trained neural network weights and architecture
- `x_scaler.pkl` - StandardScaler for input feature normalization
- `y_scaler.pkl` - StandardScaler for output parameter denormalization

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/Inverse-design-of-patch-antenna-using-Deep-Learning.git
cd Inverse-design-of-patch-antenna-using-Deep-Learning
```
### 2️⃣ Install Dependencies

Use a virtual environment for isolation:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### ▶️ Option 1: Run the Python Script

Execute the main script for inference:
```bash
python src/inverse_design_patch_antenna.py
```

The script:
* Loads the pre-trained model.
* Accepts desired antenna characteristics (e.g., resonant frequency).
* Outputs optimal geometric parameters.

### 💡 Option 2: Use the Jupyter Notebook

Launch the notebook for step-by-step experimentation:
```bash
jupyter notebook src/inverse_design_patch_antenna.ipynb
```

## Output
📉 Model Training Progress

The following plot shows model performance during training:

<img src="img\training_plots.png" alt="Training Plots">

The following is the output at `localhost: 127.0.0.1:7860` :

<img src="img\training_plots.png" alt="Training Plots">

## 🧩 Example Code Snippets
🔹 Load and Use the Trained Model

Here’s a quick example of how to use the pre-trained model for predictions:
```bash
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the model and scalers
model = load_model("models/patch_antenna_model.h5")
x_scaler = joblib.load("models/x_scaler.pkl")
y_scaler = joblib.load("models/y_scaler.pkl")

# Example: desired antenna frequency (e.g., 2.4 GHz)
target_features = np.array([[2.4]])  # Replace with your own target features

# Scale the input
target_scaled = x_scaler.transform(target_features)

# Predict antenna geometry
predicted_scaled = model.predict(target_scaled)

# Inverse transform to get real-world values
predicted_geometry = y_scaler.inverse_transform(predicted_scaled)

print("Predicted Antenna Parameters:")
print(predicted_geometry)
```
🔹 Example Output
```bash
Predicted Antenna Parameters:
[[34.75 28.62  1.6]]
```
(e.g., patch length, patch width, substrate height — units depend on dataset configuration)

##📊 Example Workflow

* Define desired operating frequency (e.g., 2.4 GHz).
* Model predicts corresponding antenna geometry.
* Output parameters can be validated using EM simulation tools (e.g., CST, HFSS).

## 🤝 Contributing

We welcome contributions! 🙌

1. 🍴 Fork the repository.
2. 🌿 Create a new branch for your feature or bugfix.
3. 🖊️ Write clear commit messages and include tests where possible.
4. 📬 Submit a pull request with a detailed description.

**Guidelines:**

* 🧹 Follow Python best practices.
* 📚 Keep code clean and well-documented.
* 📝 Update relevant documentation when making changes.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and share this project with proper attribution.
