import torch
import numpy as np
import matplotlib.pyplot as plt
from cnn import CNNSSPredictor
from dataset import ss_encoder, one_hot_encode

MODEL_PATH = r'C:\Users\hp\OneDrive\Desktop\SS_prediction\src\cnn_ss_predictor2.pth'
SEQUENCE = "AETVESCLAKSHTENSFTNVXKDDKTLDRYANYEGCLWNATGVVVCTGDETQCYGTWVPI"  # replace with any amino acid seq
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Color mapping for secondary structure classes
SS_COLOR_MAP = {
    'H': 'red',      # Helix
    'E': 'blue',     # Beta-strand
    'C': 'green',    # Coil
}

def predict_single_sequence(model, seq):
    model.eval()
    with torch.no_grad():
        x = one_hot_encode(seq).unsqueeze(0).to(DEVICE)
        outputs = model(x)
        preds = outputs.argmax(dim=-1).squeeze(0).cpu().numpy()
        preds = preds[:len(seq)]
        pred_ss = ss_encoder.inverse_transform(preds)
        return "".join(pred_ss)

def visualize_prediction(sequence, prediction):
    fig, ax = plt.subplots(figsize=(len(sequence) * 0.4, 2))

    for i, aa in enumerate(sequence):
        ax.text(i, 1, aa, ha='center', va='center', fontsize=10, fontweight='bold', color='black')

    for i, ss in enumerate(prediction):
        ax.plot(i, 0, 'o', color=SS_COLOR_MAP.get(ss, 'gray'), markersize=10)

    ax.set_xlim(-1, len(sequence))
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')

    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Helix (H)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Beta-strand (E)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Coil (C)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    plt.tight_layout()
    plt.show()

def main():
    model = CNNSSPredictor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    predicted_ss = predict_single_sequence(model, SEQUENCE)

    print("Sequence:     ", SEQUENCE)
    print("Predicted SS: ", predicted_ss)

    visualize_prediction(SEQUENCE, predicted_ss)

if __name__ == "__main__":
    main()
