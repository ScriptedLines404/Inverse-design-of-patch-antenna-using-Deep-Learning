"""
Inverse Design of Patch Antennas - Production Ready Version
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt
import gradio as gr
import json
import os

class PatchAntennaDesigner:
    def __init__(self):
        self.model = None
        self.X_scaler = None
        self.y_scaler = None
        
    def generate_dataset(self, num_samples=30000):
        """Generate synthetic dataset for patch antenna design"""
        c = 3e8  # Speed of light (m/s)
        
        # Design parameters
        f = np.random.uniform(1e9, 12e9, size=num_samples)
        er = np.random.uniform(2.2, 12.0, size=num_samples)
        h = np.random.uniform(0.5e-3, 3e-3, size=num_samples)
        
        # Analytical formulas for patch antenna
        W = np.clip(c / (2*f) * np.sqrt(2/(er+1)), 1e-3, 300e-3)
        eps_eff = (er+1)/2 + (er-1)/(2*np.sqrt(1+12*h/W))
        delta_L = 0.412*h*((eps_eff+0.3)*(W/h+0.264))/((eps_eff-0.258)*(W/h+0.8))
        L = np.clip(c / (2*f*np.sqrt(eps_eff)) - 2*delta_L, 1e-3, 300e-3)
        
        # Add realistic fabrication noise
        W_noisy = W * np.random.normal(1.0, 0.015, size=num_samples)
        L_noisy = L * np.random.normal(1.0, 0.015, size=num_samples)
        
        # Build dataset with physical enforcement
        data = pd.DataFrame({
            'f_GHz': f/1e9,
            'W_mm': np.maximum(W_noisy*1e3, L_noisy*1e3 * 1.05),
            'L_mm': L_noisy*1e3,
            'er': er,
            'h_mm': h*1e3,
        })
        
        return data
    
    def custom_loss(self, y_true, y_pred):
        """Physics-informed custom loss function"""
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        constraint_penalty = tf.reduce_mean(tf.square(
            tf.maximum(1.05*y_pred[:,0] - y_pred[:,1], 0)))
        return mse + 0.001 * constraint_penalty
    
    def constraint_satisfaction_metric(self, y_true, y_pred):
        """Metric to track physics constraint compliance"""
        return tf.reduce_mean(
            tf.cast(y_pred[:,1] >= 1.05 * y_pred[:,0], tf.float32)) * 100
    
    def build_model(self, hp):
        """Build model with hyperparameter tuning"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units=hp.Int('units1', 128, 512, step=64), 
                activation='relu', 
                input_shape=(2,)),
            tf.keras.layers.Dropout(hp.Float('dropout', 0.1, 0.5)),
            tf.keras.layers.Dense(
                units=hp.Int('units2', 64, 256, step=64), 
                activation='relu'),
            tf.keras.layers.Dense(2)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])),
            loss=self.custom_loss,
            metrics=['mae', self.constraint_satisfaction_metric]
        )
        return model
    
    def train_model(self, data, save_path="models/"):
        """Train the inverse design model"""
        # Prepare data
        X = data[['f_GHz', 'er']].values
        y = data[['L_mm', 'W_mm']].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        self.X_scaler = StandardScaler().fit(X_train)
        self.y_scaler = StandardScaler().fit(y_train)
        
        X_train_scaled = self.X_scaler.transform(X_train)
        X_test_scaled = self.X_scaler.transform(X_test)
        
        # Hyperparameter tuning
        tuner = kt.RandomSearch(
            self.build_model,
            objective=kt.Objective("val_constraint_satisfaction_metric", direction="max"),
            max_trials=15,
            directory='tuner_results',
            project_name='patch_antenna_inverse_design'
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_constraint_satisfaction_metric', 
                patience=5, 
                mode='max', 
                restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]
        
        print("Starting hyperparameter tuning...")
        tuner.search(X_train_scaled, y_train,
                    epochs=60,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=2)
        
        # Train best model
        best_hp = tuner.get_best_hyperparameters()[0]
        self.model = tuner.hypermodel.build(best_hp)
        
        print("Training best model...")
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=150,
            batch_size=128,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=2
        )
        
        # Save model and scalers
        os.makedirs(save_path, exist_ok=True)
        self.model.save(os.path.join(save_path, "final_patch_antenna_model.keras"))
        
        with open(os.path.join(save_path, "best_hyperparameters.json"), "w") as f:
            json.dump({
                'units1': best_hp.get('units1'),
                'units2': best_hp.get('units2'),
                'dropout': best_hp.get('dropout'),
                'learning_rate': best_hp.get('learning_rate')
            }, f, indent=4)
        
        # Save scalers
        import joblib
        joblib.dump(self.X_scaler, os.path.join(save_path, "x_scaler.pkl"))
        joblib.dump(self.y_scaler, os.path.join(save_path, "y_scaler.pkl"))
        
        return history
    
    def predict(self, f_GHz, er):
        """Predict patch dimensions for given frequency and dielectric constant"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        X_input = self.X_scaler.transform([[f_GHz, er]])
        pred = self.model.predict(X_input, verbose=0)
        L, W = self.y_scaler.inverse_transform(pred)[0]
        W = max(W, 1.05 * L)  # Enforce physics constraint
        
        return L, W

def create_gradio_app(designer):
    """Create Gradio web interface"""
    
    def plot_antenna(L, W):
        """Plot patch antenna geometry"""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.add_patch(plt.Rectangle((0, 0), L, W, fill=False, 
                                 edgecolor='blue', linewidth=2))
        ax.set_xlim(0, max(L, W) + 10)
        ax.set_ylim(0, max(L, W) + 10)
        ax.set_aspect('equal')
        ax.set_title(f"Patch Antenna\\nL: {L:.1f} mm | W: {W:.1f} mm")
        ax.grid(True)
        return fig
    
    def predict_patch(f_GHz, er):
        """Predict and visualize patch antenna"""
        L, W = designer.predict(f_GHz, er)
        text_output = f"Predicted L = {L:.2f} mm, W = {W:.2f} mm"
        return text_output, plot_antenna(L, W)
    
    iface = gr.Interface(
        fn=predict_patch,
        inputs=[
            gr.Slider(1.0, 12.0, value=2.4, step=0.1, label="Frequency (GHz)"),
            gr.Slider(2.2, 12.0, value=4.4, step=0.1, label="Dielectric Constant (εᵣ)")
        ],
        outputs=[
            gr.Textbox(label="Predicted Dimensions"),
            gr.Plot(label="Patch Antenna Geometry")
        ],
        title="Patch Antenna Designer",
        description="Deep Learning-based Inverse Design | Physics-Constrained"
    )
    
    return iface

if __name__ == "__main__":
    # Example usage
    designer = PatchAntennaDesigner()
    
    # Generate and train on new data
    data = designer.generate_dataset(30000)
    history = designer.train_model(data)
    
    # Create and launch Gradio app
    app = create_gradio_app(designer)
    app.launch(share=True)
