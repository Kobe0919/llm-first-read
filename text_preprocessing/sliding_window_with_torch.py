import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

# Inherit Dataset so it can be used by PyTorch DataLoader
class GPTDataset(Dataset):  
    def __init__(self, text, tokenizer, context_length, stride):
        self.token_ids = tokenizer.encode(text)
        self.context_length = context_length
        self.stride = stride

        """ 
        Eagerly programming

        for i in range(0, len(self.token_ids) - self.context_length, stride):  # too eager
            input_chunk = self.token_ids[i: i + self.context_length]
            target_chunk = self.token_ids[i + 1: i + self.context_length + 1]
            
            self.input_ids.append(torch.tensor(
                input_chunk, 
                dtype = torch.long
            ))
            self.target_ids.append(torch.tensor(
                target_chunk, 
                dtype = torch.long
            ))
        """
    
    # ============================
    # Two essential methods for Dataset
    # ============================    
    def __len__(self):
        return (len(self.token_ids) - self.context_length) // self.stride
    
    def __getitem__(self, idx):
        i = idx * self.stride

        input_chunk = self.token_ids[i: i + self.context_length]
        target_chunk = self.token_ids[i + 1: i + self.context_length + 1]

        return (
            torch.tensor(input_chunk, dtype= torch.long),
            torch.tensor(target_chunk, dtype = torch.long),
        )
    
def Build_GPTDataLoader(txt, context_length= 256, stride= 128, 
                  batch_size= 4, shuffle= True, drop_last= True):
    """
    Build DataLoader for GPT-style models using sliding window approach.

    Args:
        txt (str): The input text data.
        context_length (int): The length of each input sequence.
        stride (int): The step size for the sliding window.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        drop_last (bool): Whether to drop the last incomplete batch.
    
    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, context_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
    )

    return dataloader
