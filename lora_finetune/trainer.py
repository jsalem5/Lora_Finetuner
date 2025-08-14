from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from transformers.trainer_utils import get_last_checkpoint
from dataset import SFTDataset
import os
import json

class LoraTrainer:
    def __init__(self, lora_model, dataset, batch_size = 16, max_seq_length=1024, output_dir="outputs"):
        self.lora_model = lora_model
        if not dataset.is_formatted():
            raise ValueError("The SFTDataset must be formatted. Call the format method before passing into LoraTrainer.")
        self.training_set = dataset.training
        self.testing_set = dataset.testing
        self.logs = []
        self.output_dir = output_dir
        self.last_checkpoint = None
        per_device_batch = batch_size // 2
        eval_strategy = "no"
        if self.testing_set:
                eval_strategy = "steps"
        self.__trainer = SFTTrainer(
            model = self.lora_model.get_model(),
            tokenizer = self.lora_model.get_tokenizer(),
            train_dataset = self.training_set,
            eval_dataset = self.testing_set,
            max_seq_length = max_seq_length,
            packing = True, 
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 2,
                per_device_eval_batch_size = 2,
                gradient_accumulation_steps = per_device_batch,
                warmup_steps = 100,
                max_steps = 0,
                eval_steps = 0,
                learning_rate = 2e-4,
                logging_steps = 1,
                optim = "paged_adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = output_dir,
                report_to = "none",
                dataloader_num_workers = 2,
                eval_strategy=eval_strategy 
            ),
        )

    def train(self, steps = 100,  batch_size = 16,  logging_rate = 1, test_frequency = 10, fresh=False, checkpoint=None):
        resume_from = checkpoint if checkpoint is not None else self.last_checkpoint
        if fresh:
            self.last_checkpoint = None
            resume_from = None
        elif not self.last_checkpoint:
            self.last_checkpoint = get_last_checkpoint(self.output_dir)
        if not fresh and resume_from and os.path.isdir(resume_from):
            trainer_state_path = os.path.join(resume_from, "trainer_state.json")
            if(os.path.exists(trainer_state_path)):
                with open(trainer_state_path, 'r') as trainer_state_file:
                    trainer_state = json.load(trainer_state_file)
                    steps += trainer_state["global_step"]
        self.__trainer.args.max_steps = steps
        self.__trainer.args.logging_steps = logging_rate
        self.__trainer.args.eval_steps = test_frequency
        output = self.__trainer.train()
        self.last_checkpoint = f"{self.output_dir}/checkpoint-{steps}"
        return output

    def test(self, testing_set = None, test_samples = None):
        copy = self.__trainer.args.max_eval_samples
        if test_samples:
            self.__trainer.args.max_eval_samples = test_samples
        output = None
        if testing_set:
            output = self.__trainer.evaluate(testing_set)
        else:
            output = self.__trainer.evaluate()
        self.__trainer.args.max_eval_samples = copy
        return output

    def logs(self):
        if self.__trainer:
            self.logs.append(self.__trainer.state.log_history)
        return self.logs