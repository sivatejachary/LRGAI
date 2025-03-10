import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file
from lrgaimodel import load_model

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        text = self.lines[idx].strip()
        tokenized_text = self.tokenizer.encode(text)
        if len(tokenized_text) > self.max_length:
            tokenized_text = tokenized_text[:self.max_length]
        return torch.tensor(tokenized_text, dtype=torch.long)

def train_model(data_path, epochs=5, batch_size=16, learning_rate=0.001, save_interval=4):
    model, tokenizer, retriever, generator = load_model()
    model.train()
    dataset = TextDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    os.makedirs("Ai-thalli", exist_ok=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(torch.long)
            optimizer.zero_grad()
            outputs = model(batch, mask=None)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (batch_idx + 1) % save_interval == 0:
                model_path = f"Ai-thalli/model{batch_idx + 1}.safetensors"
                save_file(model.state_dict(), model_path)
                print(f"Model checkpoint saved at {model_path}")
                
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
    
    final_model_path = "Ai-thalli/model_final.safetensors"
    save_file(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved as {final_model_path}")

if __name__ == "__main__":
    train_model("train.txt")

# run_lrgaimodel.py
from lrgaimodel import load_model
model, tokenizer, retriever, generator = load_model()
print("LRGAI Model Loaded Successfully")
