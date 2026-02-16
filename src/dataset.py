from data.processed.train_data import train_data
from data.processed.test_data import test_data
from data.processed.val_data import validation_data
from data.raw.data import data
from src.collate_fn import custom_collate_fn
from src.device import device

#ready collate function for dataloader
from functools import partial
customize_collate_fn = partial(custom_collate_fn,device=device,allowed_max_length=1024)


#Creating DataLoaders and Dataset
from src.data_formating import format_input
from src.tokenizer import tokenizer
import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
  def __init__(self,data,tokenizer):
    self.data = data
    self.encoded_text = []

    for entry in self.data:
      instruction_input_text = format_input(entry)
      response_text = f"\n\n###Response:\n{entry['output']}"
      full_text = instruction_input_text + response_text
      self.encoded_text.append(
          tokenizer.encode(full_text)
      )

  def __getitem__(self,index):
    return self.encoded_text[index]

  def __len__(self):
    return len(self.data)


#Creating Dataloaders for LLM
 
from torch.utils.data import DataLoader
batch_size = 8
num_workers = 0

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data,tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customize_collate_fn,
    shuffle = False,
    drop_last = False,
    num_workers = num_workers
)

validation_dataset = InstructionDataset(validation_data,tokenizer)
validation_loader = DataLoader(
    validation_dataset,
    batch_size = batch_size,
    collate_fn = customize_collate_fn,
    shuffle = False,
    drop_last = False,
    num_workers = num_workers
)

test_dataset = InstructionDataset(test_data,tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    collate_fn = customize_collate_fn,
    shuffle = False,
    drop_last = False,
    num_workers = num_workers
)


#dataloader ley ekpalta 8 ota matra data lauxa dataset bata ani input and target token stack garxa and yeuta batch sakkesi feri next batch i.e 8 ota rowise data lauxa and calc input and target so on repeatly kaam garxa at the end of dataset.



#Check if i want to see data loaded correctly or not
print("Train Loaders:")

for inputs,targets in train_loader:
  print(inputs.shape,targets.shape)
