import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from dataset import ProteinSSDataset, ss_encoder, MAX_LEN  # import dataset and encoder

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys
#1d convolutional nn
class CNNSSPredictor(nn.Module):
    def __init__(self, input_dim=20, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.classifier = nn.Conv1d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.classifier(x)  
        x = x.permute(0, 2, 1)  
        return x

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)  
        outputs = outputs.contiguous().view(-1, outputs.shape[-1])
        y_batch = y_batch.contiguous().view(-1)

        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            preds = outputs.argmax(dim=-1).cpu().numpy()
            labels = y_batch.cpu().numpy()
            for p_seq, l_seq in zip(preds, labels):
                valid_indices = l_seq != -1
                all_preds.extend(ss_encoder.inverse_transform(p_seq[valid_indices]))
                all_labels.extend(ss_encoder.inverse_transform(l_seq[valid_indices]))
    print(classification_report(all_labels, all_preds, digits=4))

def main():
    # since im working with limited units of gpu on google colab
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets / dataloaders
    train_dataset = ProteinSSDataset('train.csv')
    valid_dataset = ProteinSSDataset('valid.csv')
    test_dataset = ProteinSSDataset('test.csv')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    model = CNNSSPredictor().to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # ignore padding labels
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 20
    #since we're working with very limited ressource, we follow the following techniques: 
    #1- we run the training for 10 epochs, 
    #2- save the cnn weights 
    #re load the cnn with the saved weights and re launch the training 
    #until getting good results
    for epoch in range(1, EPOCHS+1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        print("Validation report:")
        evaluate(model, valid_loader, device)

    torch.save(model.state_dict(), './cnn_ss_predictor.pth')
    print("Model weights saved.")

    print("Final test set evaluation:")
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()
