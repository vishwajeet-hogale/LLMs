from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm
import os

def load_data(link):
    return pd.read_json(link, lines=True)
def pre_process_data(df):
    data = []
    for _,row in df.iterrows():
        data.append({
            "question": row["instruction"],
            "context": row["context"],
            "answer": row["response"]
        })
    return data
        
class CustomQnADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_input_length=256, max_output_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        sample = self.data[idx]
        q = sample["question"]
        c = sample["context"]
        a = sample["answer"]
        
        prompt = f"Answer the question in good deatil. question: {q} context: {c}"
        input_embeddings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_input_length,
            padding="max_length",  
            return_tensors="pt"
        )
        output_embeddings = self.tokenizer(
            a,
            truncation=True,
            max_length=self.max_output_length,
            padding="max_length",  
            return_tensors="pt"
        )
        return {
            "input_ids": input_embeddings["input_ids"].squeeze(0),       # (seq_len)
            "attention_mask": input_embeddings["attention_mask"].squeeze(0),  # (seq_len)
            "labels": output_embeddings["input_ids"].squeeze(0)          # (seq_len)
        }
        
def train(dataloader, model, device, epochs = 1):
    
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr= 1e-5)
    for i in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels)
            loss = outputs.loss
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch [{i+1}/{epochs}], Step [{i+1}], Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{i+1}/{epochs}] completed. Avg Loss: {avg_loss:.4f}")
    model.save_pretrained("my_custom_t5_qa_model_large")
    tokenizer.save_pretrained("my_custom_t5_qa_model_large")
def test(model, tokenizer, device, question, context=""):
    """
    Generates an answer from a question and optional context using the fine-tuned model.
    """

    model.eval()  # Switch to eval mode
    # Build the prompt
    prompt = f"question: {question} context: {context}"
    # Tokenize the prompt
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        max_length=256, 
        truncation=True
    )
    # Move tensors to device
    input_ids = inputs["input_ids"].to(device)
    # Generate output (greedy search, for example)
    # You can adjust max_length, num_beams, temperature, etc. for better results
    output_ids = model.generate(
        input_ids,
        max_length=64,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )
    # Decode to string
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Q:", question)
    print("A:", answer)

if __name__ == "__main__":
    device = torch.device("mps")
    link = "hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl"
    df = load_data(link)
    df = df[df["category"] == "closed_qa"]

    data = pre_process_data(df)
    model_name = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    datatset = CustomQnADataset(data, tokenizer, 256, 64)
    dataloader = DataLoader(datatset)
    MODEL_DIR = "./my_custom_t5_qa_model_large"
    if not os.path.exists(MODEL_DIR):
        train(dataloader, model, device,1)
    else: 
        print(f"Model directory '{MODEL_DIR}' already exists. Loading model from disk...")
        model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
        model = model.to(device)
        tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
        
    test(model, tokenizer, device, "What is a dispersive prism?", context="In optics, a dispersive prism is an optical prism that is used to disperse light, that is, to separate light into its spectral components (the colors of the rainbow). Different wavelengths (colors) of light will be deflected by the prism at different angles. This is a result of the prism material's index of refraction varying with wavelength (dispersion). Generally, longer wavelengths (red) undergo a smaller deviation than shorter wavelengths (blue). The dispersion of white light into colors by a prism led Sir Isaac Newton to conclude that white light consisted of a mixture of different colors.")
    
    
    
    
    