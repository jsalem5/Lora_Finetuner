from unsloth import FastLanguageModel
import torch
import os
import ollama
import subprocess
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from transformers import TextStreamer

template = """{% set instruction = '' %}
{% for message in messages %}
    {% if message['role'] == 'system' %}
        {% set instruction = message['content'] %}
    {% elif message['role'] == 'user' %}
        ### Instruction:
        {{ instruction or 'Complete the sequence' }}

        ### Input:
        {{ message['content'] }}
    {% elif message['role'] == 'assistant' %}
        ### Response:
        {{ message['content'] }}
    {% endif %}
{% endfor %}"""




def get_modelname_list():
        return ["unsloth/mistral-7b-v0.3-bnb-4bit", 
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", 
        "unsloth/llama-3-8b-bnb-4bit", 
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "unsloth/llama-3-70b-bnb-4bit", 
        "unsloth/Phi-3-mini-4k-instruct", 
        "unsloth/Phi-3-medium-4k-instruct", 
        "unsloth/mistral-7b-bnb-4bit", 
        "unsloth/gemma-7b-bnb-4bit"
        ]
 

def valid_repo(repo_id, token=None):
        if os.path.isdir(repo_id):
                return True
        if token:
                model_info(repo_id, token=token)
        else:
                model_info(repo_id)
        return True

class LoraModel:
        def __init__(self, base_model, ollama_model = None, max_seq_length = 1024, dtype=None, weight_4bit = True, lora_r = 8, lora_alpha = 16, huggingface_token = None):
                valid_repo(base_model, token=huggingface_token)

                self.model, self.tokenizer = FastLanguageModel.from_pretrained(model_name = base_model, max_seq_length = max_seq_length, dtype = dtype, load_in_4bit = weight_4bit)
                self.model = FastLanguageModel.get_peft_model(self.model,
                        r = lora_r,
                        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj",],
                        lora_alpha = lora_alpha,
                        lora_dropout = 0,
                        bias = "none",
                        use_gradient_checkpointing = "unsloth",
                        random_state = 3407,
                        use_rslora = False,
                        loftq_config = None
                        )
                self.messages = []
                self.ollama_model = ollama_model

        def save(self, save_dir="lora_model"):
                self.model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                print(f"Model saved to {save_dir}")

        def save_ollama(self, name, save_dir="model"):
                self.tokenizer.chat_template = """{% set instruction = '' %}
                {% for message in messages %}
                {% if message['role'] == 'system' %}
                        {% set instruction = message['content'] %}
                {% elif message['role'] == 'user' %}
                        ### Instruction:
                        {{ instruction or 'Complete the sequence' }}

                        ### Input:
                        {{ message['content'] }}
                {% elif message['role'] == 'assistant' %}
                        ### Response:
                        {{ message['content'] }}
                {% endif %}
                {% endfor %}"""
                self.model.save_pretrained_gguf(save_dir, self.tokenizer, quantization_method = "q4_k_m", maximum_memory_usage = 0.75)
                result = subprocess.run(["ollama", "create", f"{name}", "-f", os.path.join(save_dir, "Modelfile")], capture_output=True, text=True, check=True)
                self.ollama_model = name
                print(f"Ollama model created: {self.ollama_model}")

        def get_tokenizer(self):
                return self.tokenizer
        
        def get_model(self):
                return self.model

        def inference(self, message, interactive=True, force_save=False, force_save_name = None):
                if force_save:
                        if not force_save_name:
                                raise ValueError("force_save_name must be set if force_save is True")
                        self.save_ollama(name=force_save_name, save_dir=force_save_name)
                if not self.ollama_model:
                        print("No ollama model provided, call save_ollama() to save the LoraModel as an ollama model")
                        return
                if interactive:
                        subprocess.run(["ollama", "run", f"{self.ollama_model}"])
                        return
                self.messages.append({'role': 'user', 'content': message})
                response = ollama.chat(model=self.ollama_model, messages=self.messages)
                self.messages.append(response['message'])
                return response['message']['content']
