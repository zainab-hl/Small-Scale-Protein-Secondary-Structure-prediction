import torch
import tkinter as tk
from tkinter import messagebox
from visualize import predict_single_sequence
from cnn import CNNSSPredictor  # your CNN class

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = r'C:\Users\hp\OneDrive\Desktop\SS_prediction\src\cnn_ss_predictor2.pth'
model = CNNSSPredictor().to(DEVICE)
model.load_state_dict(torch.load(path, map_location=DEVICE))

# Function to handle prediction
def run_prediction():
    seq = seq_entry.get().strip()
    if not seq:
        messagebox.showerror("Error", "Please enter an amino acid sequence.")
        return
    
    try:
        pred_ss = predict_single_sequence(model, seq)
        output_label.config(text=f"Predicted SS: {pred_ss}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI setup
root = tk.Tk()
root.title("Secondary Structure Prediction")
root.configure(bg="#f0f4f7")  # Light gray-blue background

# Input label
tk.Label(
    root,
    text="Enter Amino Acid Sequence:",
    bg="#f0f4f7",
    fg="#333",
    font=("Arial", 12, "bold")
).pack(pady=5)

# Entry box
seq_entry = tk.Entry(root, width=50, font=("Arial", 11))
seq_entry.pack(pady=5, ipady=3)

# Predict button
tk.Button(
    root,
    text="Predict",
    command=run_prediction,
    bg="#4CAF50",
    fg="white",
    font=("Arial", 11, "bold"),
    relief="raised",
    activebackground="#45a049"
).pack(pady=10)

# Output label
output_label = tk.Label(
    root,
    text="",
    font=("Courier", 13, "bold"),
    fg="#0066cc",
    bg="#f0f4f7"
)
output_label.pack(pady=15)

root.mainloop()
