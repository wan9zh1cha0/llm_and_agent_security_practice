import urllib.request
import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# we can then build a vocabulary from the preprocessed list and complete the encode function definition
# decode function can map token IDs back to words and join them to get LLM output

tokenizer = tiktoken.get_encoding("gpt2") 
tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"}) 

# data sampling
class GPTDatasetV1(Dataset): # inherit from pytorch's Dataset class, extract input-target pairs from text, need to define __len__ and __getitem__ functions
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = [] # input-target pair - input tensor
        self.target_ids = [] # input-target pair - target tensor
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True, 
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            drop_last=drop_last, num_workers=num_workers) # drop_last means drop the last batch if its length is less than batch_size, num_workers means how many CPUs to use
    return dataloader
    
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=3, shuffle=False) 
# batch_size (small) will affect training (negatively) and memory usage (positively), is a hyperparameter
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print(inputs)
print(targets)
vocab_size = 50257
output_dim = 256
torch_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
token_embeddings = torch_embedding_layer(inputs)
context_size = max_length
pos_embedding_layer = torch.nn.Embedding(context_size, output_dim) # position here means the position of each word in the sliding window
pos_embeddings = pos_embedding_layer(torch.arange(context_size))
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings)


enable = False
if enable == True:
    tokenizer = tiktoken.get_encoding("gpt2") # use GPT-2's byte pair encoding tokenizer
    text = ("hello, I am pythonahahaha")
    print(tokenizer.encode(text, allowed_special={"<|endoftext|>"}))

    # embedding test
    vocab_size = 6
    output_dim = 3
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)
    input_ids = torch.tensor([2, 3, 5, 1])
    print(embedding_layer(input_ids))
