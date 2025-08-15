import argparse, json

# function used to extrat a value for the key from a json_file  
def __get_json_file_key(json_path, key):
    with open(json_path, 'r') as json_file:
        json_copy = json.load(json_file)
        return json_copy.get(key)

# function checks that certain arguments are set within proper range and also that certain arguements are set given a certain mode
def check_args(args):
    if args.steps <= 0:
        raise argparse.ArgumentTypeError("Steps must be greater 0")
    if args.batch_size <= 0:
        raise argparse.ArgumentTypeError("Batch size must be greater than 0")
    if args.mode == 'train':
        if not args.dataset:
            raise argparse.ArgumentTypeError("In \"train\" mode, the name of a huggingface dataset must be provided using \"--dataset\"")
        if not args.input_col and not args.ins_col:
            raise argparse.ArgumentTypeError("In \"train\" mode, either the name of the input column or instruction column of the database must be provided using \"--input_col\"")
    if args.mode == 'inference':
        if not args.prompt:
            raise argparse.ArgumentTypeError("In \"inference\" mode, the a prompt must be provided \"--prompt\"")

# Used to represent the type of checkpoint the training will start from
RESUME_CHECKPOINT = -1
FRESH_TRAIN = 0
# This the template that the input and output data from the provided database will be molded into
CHAT_TEMPLATE = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.
                
                ### Instruction:
                {INPUT}
                        
                ### Response:
                {OUTPUT}"""

def main(args):
    # it is important that these imports are placed here, because unsloth must be imported first for proper function. 
    # unsloth takes some time to load and displays messages so its important that the imports happen after the user has passed the args
    from unsloth import FastLanguageModel, to_sharegpt, standardize_sharegpt, apply_chat_template
    import torch, os
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset, load_from_disk, Dataset
    from transformers.trainer_utils import get_last_checkpoint
    import pandas as pd

    # applys the set chat template to the given tokenizer. This only for use cases where you need to add a template but don't need or have a dataset.
    def add_chat_template(tokenizer):
        # we create an empty dataset since the apply_chat_template needs a dataset arg. However we won't need this dataset so leaving it empty is efficient.
        columns = ["INPUT", "OUTPUT"]
        empty_df = pd.DataFrame({col: [] for col in columns})
        empty_dataset = Dataset.from_pandas(empty_df)
        apply_chat_template(empty_dataset,tokenizer=tokenizer,chat_template=CHAT_TEMPLATE)

    # takes a huggingface dataset and standardize the dataset to a chat format that the passed in tokenizer understands
    # input_column is the name of the column in the dataset containing input data, common names are "input, "text", and "prompt"
    # instruction_column is the name of the column in the dataset containing instruction data
    # output_column is the name of the column in the dataset containing the output data, common names are "output", "response", and "label"
    # Since the input_column and insturction_column data will be combined, the dataset does not need to have both these types of columns, but it must have one of them
    def get_formated_dataset(dataset, tokenizer, output_column, input_column=None, instruction_column=None, instruction=None):
        if not input_column and not instruction_column:
            raise ValueError("input_column and instruction_column cannot both be None")  
        ins = "Complete the sequence"
        # if no insturction column is provided put generic "Complete the sequence" prompt
        instruction_prompt = f"{"[[{" + instruction_column + "}]]" if instruction_column else ins}"
        # if input column is provided leave it blank
        input_prompt = f"{"[[\nYour input is:\n{" + input_column + "}]]" if input_column else ""}"
        # merges the instruciton and input columns in the dataset
        dataset = to_sharegpt(
            dataset,
            merged_prompt=f"{instruction_prompt}{input_prompt}",
            output_column_name=output_column,
            conversation_extension=1,
        )
        dataset = standardize_sharegpt(dataset)
        # take each row standardized dataset and insert input and output data into the chat template
        dataset = apply_chat_template(
            dataset,
            tokenizer=tokenizer,
            chat_template=CHAT_TEMPLATE
        )
        return dataset
    
    check_args(args)

    # check whether the model passed in is a peft adapter
    is_peft = False
    if os.path.isdir(args.model) and os.path.exists(os.path.join(args.model, "adapter_config.json")):
        # adapter directories contain a config json
        adapter_config = os.path.join(args.model, "adapter_config.json")
        peft_type = __get_json_file_key(adapter_config, "peft_type")
        # see if the peft type is lora
        if peft_type and peft_type == "LORA":
            is_peft = True
    
    # loads a huggingface model and an appropriate tokenizer for the model
    model, tokenizer = FastLanguageModel.from_pretrained(model_name = args.model, max_seq_length = args.context_window, dtype = None, load_in_4bit = True)
    if not is_peft:
        # this must be a base model so we need to load a peft lora model
        # goto https://huggingface.co/docs/peft/main/en/conceptual_guides/lora to learn about what lora is
        model = FastLanguageModel.get_peft_model(
            model,
            r = 8,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            use_rslora = False,
            loftq_config = None,
        )
    if args.mode == 'train':
        # see if the user set the checkpoint flag 
        if args.checkpoint is RESUME_CHECKPOINT or args.checkpoint > 0:
            checkpoint_path = get_last_checkpoint(args.checkpoint_dir) if (args.checkpoint is RESUME_CHECKPOINT) else (os.path.join(args.checkpoint_dir, f"checkpoint-{args.checkpoint}"))
            if os.path.isdir(checkpoint_path):
                # the step count is stored in a trainer state json in the checkpoint dir
                trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
                if(os.path.exists(trainer_state_path)):
                    # the trainer step count is absolute so to determine the new absolute 
                    # step count we need add the previous step count from the checkpoint to the amount of steps the user wants to train
                    args.steps += __get_json_file_key(trainer_state_path, "global_step")
        
        # load the dataset and format it
        dataset = load_dataset(args.dataset, split = 'train') if not args.from_disk else load_from_disk(args.dataset)
        training_set = get_formated_dataset(dataset, tokenizer, input_column=args.input_col, output_column=args.output_col)
        
        # create a SFTTrainer, goto https://huggingface.co/docs/trl/en/sft_trainer#trl.SFTTrainer 
        # and https://huggingface.co/docs/transformers/en/main_classes/trainer#api-reference%20][%20transformers.Trainer to 
        # learn about what each parameter means
        trainer = SFTTrainer(model = model,
                            tokenizer = tokenizer,
                            train_dataset = training_set,
                            max_seq_length = args.context_window,
                            packing = False, 
                            dataset_text_field = "text",
                            args = SFTConfig(
                                per_device_train_batch_size = args.batch_size,
                                gradient_accumulation_steps = (args.batch_size * 4),
                                warmup_steps = 100,
                                max_steps = args.steps,
                                learning_rate = 2e-4,
                                logging_steps = 5,
                                optim = "paged_adamw_8bit",
                                weight_decay = 0.01,
                                lr_scheduler_type = "linear",
                                seed = 3407,
                                output_dir = args.checkpoint_dir,
                                report_to = "none",
                                dataloader_num_workers = 10,
                                save_strategy="steps",
                                save_total_limit=3,
                                )
                            )

        if args.checkpoint is FRESH_TRAIN:
            # no checkpoint to start from
            trainer.train()
            # save the model and tokenzier
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            print(f"Saving model and tokenizer to {args.save_dir}")
        else:
            checkpoint_path = get_last_checkpoint(args.checkpoint_dir) if (args.checkpoint is RESUME_CHECKPOINT) else (os.path.join(args.checkpoint_dir, f"checkpoint-{args.checkpoint}"))
            if os.path.isdir(checkpoint_path):
                # check that checkpoint is valid, then start training from checkpoint. 
                # When you train from a checkpoint the training optimizer state and dataset progress is perserved
                trainer.train(resume_from_checkpoint=checkpoint_path)
                print(f"Saving model and tokenizer to {args.save_dir}")
                model.save_pretrained(args.save_dir)
                tokenizer.save_pretrained(args.save_dir)
            else:
                print("invalid checkpoint, skipping training")        
    elif args.mode == 'test':
        testing_set = load_dataset(args.dataset) if not args.from_disk else load_from_disk(args.dataset)
        # attempt to load a training split from the dataset
        if 'test' in testing_set:
            testing_set = testing_set['test']
        elif 'validation' in testing_set:
            testing_set = testing_set['validation']
        elif 'train' in testing_set:
            print("Only training split avaiable, will pick a subset of the training split to test against")
            testing_set = testing_set.shuffle(seed=100)
        # select the number of samples to test against
        testing_set = testing_set.select(list(range(min(args.steps, testing_set.num_rows))))
        testing_set = get_formated_dataset(testing_set, tokenizer, input_column=args.input_col, output_column=args.output_col)
        # create lightweight trainer to test with
        trainer = SFTTrainer(model = model,
                            tokenizer = tokenizer,
                            train_dataset = testing_set, 
                            eval_dataset = testing_set,
                            max_seq_length = args.context_window,
                            packing = True, 
                            args = SFTConfig(
                                dataset_text_field = "text",
                                per_device_eval_batch_size = 1,
                                warmup_steps = 100,
                                seed = 3407,
                                output_dir = args.checkpoint_dir,
                                dataloader_num_workers = 10,
                                )
                            )
        print(trainer.evaluate())
        return
    elif args.mode == "ollama":
        import subprocess, ollama
        # must add chat template for proper ollama model generation
        add_chat_template(tokenizer)

        # function to check if the ollama model already exsists
        def ollama_model_exists(model_name):
            models = ollama.list()
            for model in models.get("models", []):
                if model.get("model") == model_name:
                    return True
        assert tokenizer._ollama_modelfile
        # check to see whether a gguf folder already exsists for the model, if not create proper gguf folders for model
        if args.force_ollama or not os.path.isdir(f"{args.model}_gguf_model"):
            model.save_pretrained_gguf(f"{args.model}_gguf_model", tokenizer, quantization_method = "q4_k_m", maximum_memory_usage = 0.75)
        else:
            print("gguf folder already exsists for this model name. If you want to rewrite the folder and recreate the Ollama model, set the \"--force_ollama\" flag")
        
        # check to see if ollama model already exists, if it doesn't, make it. The user can force the creation of a new ollama model with the force_ollama flag
        if args.force_ollama or not ollama_model_exists(f"{args.model}-ollama-model:latest"):
            print("Creating ollama model using gguf")
            subprocess.run(["ollama", "create", f"{args.model}-ollama-model", "-f", os.path.join(f"{args.model}_gguf_model", "Modelfile")], capture_output=True, text=True, check=True)
            print("Success")
        else:
            print(f"{args.model}-ollama-model already exists. Ollama will use this model.")
        print("Staring Ollama terminal")
        subprocess.run(["ollama", "run", f"{args.model}-ollama-model:latest"])
        return
    elif args.mode == "inference":
        # must add chat template to tokenzier for proper inference generation
        add_chat_template(tokenizer)
        FastLanguageModel.for_inference(model)
        messages = [{"role": "user", "content": args.prompt}]
        # tokenize user input
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt = True, return_tensors = "pt").to("cuda")
        attention_mask = torch.ones_like(input_ids)
        from transformers import TextStreamer
        print("\nResponse:")
        # steam response to user
        text_streamer = TextStreamer(tokenizer, skip_prompt = True)
        _ = model.generate(input_ids, attention_mask=attention_mask, streamer = text_streamer, max_new_tokens = 250, pad_token_id = tokenizer.eos_token_id)

def entry():
    # get all the parameters
    parser = argparse.ArgumentParser(description='CLI tool to fine tune and interact with huggingface models. The script uses unsloth to perform the finetuning.')
    parser.add_argument('mode', choices=['train', 'test', 'inference', 'ollama'], 
                        help="Mode to run the model in. \"Train\" will train the given model on the choosen dataset.\n\"Test\" \
                        will test the the model againt the choose dataset. NOTE: if the provided huggingface database does not have \
                        a testing split, a subset of the training set will be used. \n\"Inference\" can be used to generate new output data on input data.\n")
    parser.add_argument('model', type=str, help='Either a huggingface model id, a path to a huggingface compatible local pre-trained model, \
                        or a path to a local directory containing a lora adapter\n')
    parser.add_argument('--checkpoint_dir', type=str, default='lf_checkpoints', help='Path to directory where checkpoints are located and where new training checkpoints will be stored.')
    parser.add_argument( '--checkpoint', nargs="?", type=int, default=FRESH_TRAIN, const=RESUME_CHECKPOINT, help='Whether to start training from a saved checkpoint or to start training from scratch. \
                        If the flag is present but no specifc number is set, will automatically resume from the most recent checkpoint in checkpoint directory\n')
    parser.add_argument('--steps', type=int, default=50, help='How many steps to perform in training or testing\n')
    parser.add_argument('--dataset', type=str, help="Id of huggingface database to use for training or testing\n")
    parser.add_argument('--from_disk', action="store_true", help="Set this flag if the dataset that is being loaded is from local disk\n")
    parser.add_argument('--input_col', type=str, help="Name of the column of the input data in the huggingface database. NOTE: the input column and instruction column will be combined, so you need at least one of these columns but it is optional to have both.\n")
    parser.add_argument('--output_col', type=str, help="Name of the column of the output data in the huggingface database\n")
    parser.add_argument('--ins_col', type=str, help="Name of the column of the instruction data in the huggingface database NOTE: the input column and instruction column will be combined, so you need at least one of these columns but it is optional to have both.\n")
    parser.add_argument('--save_dir', type=str, default="lf_model", help="Directory where model will be saved after training\n")
    parser.add_argument('--batch_size', type=int, default=16, help="Training and testing batch size. Decreasing this can lower vram usage.\n")
    parser.add_argument('--prompt', type=str, help="Message that model will generate output from in inference mode. Enclose your prompt in \"\"")
    parser.add_argument('--force_ollama', action="store_true", help="If this flag is set when in \"Ollama\" mode it will rewrite the exsisting gguf_folder for the model")
    parser.add_argument('--context_window', type=int, default=512, help="Sets the context window. Decreasing this can lower vram usage.")
    main(parser.parse_args())

if __name__ == "__main__":
    entry()