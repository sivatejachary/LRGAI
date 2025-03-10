import torch

def top_k_top_p_filtering(logits, top_k=50, top_p=0.95):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    if top_k > 0:
        sorted_indices_to_remove[..., top_k:] = True
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits

class Generator:
    def __init__(self, model, retriever):
        self.model = model
        self.retriever = retriever
    
    def generate(self, input_tensor, mask):
        with torch.no_grad():
            logits = self.model(input_tensor, mask)
            filtered_logits = top_k_top_p_filtering(logits)
            probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
            return torch.multinomial(probabilities, num_samples=1)
