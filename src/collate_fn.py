
from src.device import device
import torch

#Define custom collate func where each batch is converted to input and target for model
def custom_collate_fn(batch,device,allowed_max_length=None,padding_token=50256,ignore_index=-100):
  #find max length in each batch
  max_length = max(len(item)+1 for item in batch)  #here +1 for the future target input.
  input_list = []
  target_list = []

  #pad the padding token
  for item in batch:
    new_items = item.copy()
    new_items += [padding_token]
    padded_items = new_items + [padding_token] * (max_length-len(new_items))
    input_ids = torch.tensor(padded_items[:-1]) #truncate the last 50256 for input
    target_ids = torch.tensor(padded_items[1:]) #shift +1 to the right for targets

    #Replace all  but the first padding token by ignore_index
    mask = target_ids == padding_token  #check garxa target_ids vitra kaha kaha padding token xa and return boolean value in mask like suppose mask = tensor(false,false,true,true,true).
    indices = torch.nonzero(mask).squeeze() # torch.nonzero ley mask vitra ko true vayeko position lai nikalxa and dimension hatauxa .squeeze le and  yesto hunxa indices = [2,3,4] 0 and 1 th position ma false xa so true matra ligyo.
    if indices.numel()>1:
      target_ids[indices[1:]] = ignore_index #aaba yesley indices [3,4] 1th position xodera 3 and 4 ma matra 50256 rakhxa and target yesto hunxa target =[1,2,3,4,50256,-100-100]

    if allowed_max_length is not None:
      input_ids = input_ids[:allowed_max_length]
      target_ids = target_ids[:allowed_max_length]


    input_list.append(input_ids)
    target_list.append(target_ids)

  #converting the input list to tensor and transfer to target device
  input_tensor = torch.stack(input_list).to(device)
  target_tensor = torch.stack(target_list).to(device)

  return input_tensor,target_tensor


##test if it works correctly

# input_1 = [1,2,3]
# input_2 = [4,5,6,7]
# input_3 = [8,9]
# batch = [
#     input_1,
#     input_2,
#     input_3
# ]
# inputs,targets = custom_collate_fn(batch,device)
# print(inputs)
# print(targets)