from datasets import load_dataset
from huggingface_hub import dataset_info
from unsloth import to_sharegpt, standardize_sharegpt, apply_chat_template

def valid_dataset(dataset_id, token=None):
    if token:
        dataset_info(dataset_id, token=token)
    elif not token:
        dataset_info(dataset_id)
    return True

def standardize_dataset(dataset, tokenizer, input_column, output_column, instruction_column=None, instruction=None):
    ins = instruction if instruction else "Complete the sequence"
    dataset = to_sharegpt(
        dataset,
        merged_prompt=f"{("{" + instruction_column + "}" if (instruction_column is not None) else (ins))}[[\nYour input is:\n{"{" + input_column + "}"}]]",
        output_column_name=output_column,
        conversation_extension=3,  # Select more to handle longer conversations
    )
    dataset = standardize_sharegpt(dataset)
    chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.
    ### Instruction:
    {INPUT}
    ### Response:
    {OUTPUT}"""
    dataset = apply_chat_template(
        dataset,
        tokenizer=tokenizer,
        chat_template=chat_template
    )
    return dataset

class SFTDataset:
    def __init__(self, dataset_id, token=None):
        valid_dataset(dataset_id, token)
        results = load_dataset(dataset_id)
        self.training = results.get('train')
        self.testing = results.get('test')
        self.__formatted = False


    def get_column_names(self):
        if self.training:
            return self.training.features
        elif self.testing:
            return self.testing.features
        else:
            return []
    
    def get_training_sample(self, row):
        return self.training[row] if self.training else None

    def get_testing_sample(self, row):
        return self.testing[row] if self.testing else None

    def get_num_training_samples(self):
        return self.training.num_rows if self.training else 0
    
    def get_num_testing_samples(self):
        return self.testing.num_rows if self.testing else 0

    def format(self, tokenizer, input_column, output_column, instruction_column=None, instruction=None):
        if not self.__formatted:
            self.training = standardize_dataset(self.training, tokenizer, input_column, output_column, instruction_column, instruction)
            if self.testing:
                self.testing = standardize_dataset(self.testing, tokenizer, input_column, output_column, instruction_column, instruction)
            self.__formatted = True
        return
    
    def is_formatted(self):
        return self.__formatted