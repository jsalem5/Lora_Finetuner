from model import LoraModel
from trainer import LoraTrainer
from dataset import SFTDataset
import ollama
import os
import subprocess
name="testmodel"
save_dir="../model"

lora_model = LoraModel("unsloth/llama-3-8b-bnb-4bit", ollama_model="lora_llama_model")
dataset = SFTDataset("emotion")
dataset.format(tokenizer=lora_model.get_tokenizer(), input_column="text", output_column="label")
trainer = LoraTrainer(lora_model, dataset)
trainer.train(50)
lora_model.save()
lora_model.save_ollama()
# lora_model.inference("What is 2+2=")
# lora_trainer = LoraTrainer(lora_model, dataset)
# lora_trainer.train(steps=20,fresh=True, checkpoint="./outputs/checkpoint-95")
# lora_trainer.train(steps=50)
# lora_trainer.train(steps=25)

print(lora_model.inference("What is 2 + 2")) 
