import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

def train_model():
    model, tokenizer, retriever, generator = load_model()
    dataset = load_dataset("wikipedia", "20220301.en")
    dataloader = DataLoader(dataset["train"], batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(3):
        for batch in dataloader:
            optimizer.zero_grad()
            text = batch["text"]
            tokens = [tokenizer.encode(t) for t in text]
            input_tensor = torch.tensor(tokens)
            mask = torch.ones_like(input_tensor)
            output = model(input_tensor, mask)
            loss = output.mean()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed")
    print("Training finished")
