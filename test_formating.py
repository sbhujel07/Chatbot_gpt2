from Src.data_formating import format_input
import json
with open("data/raw/instruction-data.json","r") as f:
    data = json.load(f)

model_input = format_input(data[0])
desired_response = f"\n\n###Response:\n{data[0]['output']}"
print(model_input + desired_response)
print("Total length of Data:",len(data))

#test if we need to see the total length of data ,train data ,test data and train data.
with open("data/processed/test.json","r")as f:
    test_data = json.load(f)

with open("data/processed/train.json","r")as f:
    train_data = json.load(f)

with open("data/processed/val.json","r")as f:
    validation_data = json.load(f)

print("Train data:",len(train_data))
print("Test_data:",len(test_data))
print("Validation_data:",len(validation_data))