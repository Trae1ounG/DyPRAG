# Stage 2: DyPRAG training
import os
import gc
import json
import argparse
import torch
import random
from tqdm import tqdm
import time
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete, get_attributes, delta_inject, delta_remove
from encode import get_train_data
from projector import ParameterTranslator
from transformers import DefaultDataCollator
from typing import List, Dict
import torch.nn.functional as F
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from collections import defaultdict


def create_lora_passage_dataset(lora_passage_pairs):
    """
    Create a PyTorch dataset from LoRA-passage pairs
    """
    from torch.utils.data import Dataset
    
    class LoRAPassageDataset(Dataset):
        def __init__(self, pairs):
            self.pairs = pairs
            
        def __len__(self):
            return len(self.pairs)
            
        def __getitem__(self, idx):
            return self.pairs[idx]
    
    return LoRAPassageDataset(lora_passage_pairs)

def prepare_training_data_multi_datasets(args, tokenizer, datasets):
    """
    Prepare training data from multiple datasets
    """
    all_training_samples = []
    ignored_id = -100
    max_length = 3000
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        # Update args for current dataset
        args.dataset = dataset_name
        if dataset_name in ("2wikimultihopqa", "hotpotqa"):
            prompt_template.get_fewshot(dataset_name)
            args.with_cot = True
        else:
            args.with_cot = False
        if dataset_name == "popqa":
            num_train_epochs = 2
        else:
            num_train_epochs = 1
        projector = True

        data_list = load_data(dataset_name, None, args.augment_model, projector, data_dir="./data_aug_projector")
        for filename, fulldata in data_list:
            filename = filename.split(".")[0]
            print(f"Collecting data from {filename}")
            data_size = len(fulldata) 
            sample_data = random.sample(range(data_size),int(data_size * args.sample_rate))
            for test_id in tqdm(sample_data):
                data = fulldata[test_id]
                augment = data["augment"] 
                for pid, aug in enumerate(augment):
                    adapter_path = os.path.join(
                        ROOT_DIR, 
                        "offline", 
                        args.model_name, 
                        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
                        dataset_name,
                        f"lr={args.learning_rate}_epoch={num_train_epochs}_{'cot' if args.with_cot else 'direct'}",
                        f"aug_model={args.augment_model}_projector",
                        filename,
                        f"data_{test_id}",
                        f"passage_{pid}"
                    )
                    if os.path.exists(adapter_path):
                        # Get raw prompt_ids using the augmented data
                        raw_prompt_ids = get_train_data(args.augment_model, [aug], tokenizer, args)[0]
                        # Process prompt_ids following encode.py's TrainingData class
                        labels = raw_prompt_ids.copy()
                        if len(raw_prompt_ids) > max_length:
                            raw_prompt_ids = raw_prompt_ids[:max_length]
                            labels = labels[:max_length]
                        attention_mask = [1] * len(raw_prompt_ids) + [0] * (max_length - len(raw_prompt_ids))
                        raw_prompt_ids += [pad_token_id] * (max_length - len(raw_prompt_ids))
                        labels += [ignored_id] * (max_length - len(labels))
                        
                        # Tokenize passage for embedding
                        passage_tokens = tokenizer(
                            aug["passage"], 
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=max_length
                        )
                        all_training_samples.append({
                            'adapter_path': adapter_path,
                            'passage': aug["passage"],  
                            'passage_tokens': passage_tokens,
                            'input_ids': raw_prompt_ids,
                            'labels': labels,
                            'attention_mask': attention_mask,
                            'file_name': filename,
                            'data_id': test_id,
                            'passage_id': pid,
                            'dataset': dataset_name
                        })
    print(f"\nTotal prepared {len(all_training_samples)} training samples from {len(datasets)} datasets")
    return all_training_samples

class TrainingDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.model = model
    
    def __call__(self, examples: List[Dict[str, dict]]) -> Dict[str, torch.Tensor]:
        input_embeds = []
        adapter_paths = []
        model_inputs = []
        
        for example in examples:
            # Use pre-tokenized passage
            tokenized = example['passage_tokens']
            
            with torch.no_grad():
                output = self.model(tokenized.input_ids, output_hidden_states=True)
                # Take the last hidden state as document embedding
                input_embeds.append(output.hidden_states[-1][:,-1,:])
                adapter_paths.append(example['adapter_path'])
                model_inputs.append({
                    'input_ids': torch.tensor(example['input_ids']).unsqueeze(0).to(self.device),
                    'labels': torch.tensor(example['labels']).unsqueeze(0).to(self.device),
                    'attention_mask': torch.tensor(example['attention_mask']).unsqueeze(0).to(self.device)
                })
                    
        return {
            "input_embeds": input_embeds,
            "adapter_paths": adapter_paths,
            "model_inputs": model_inputs,
        }

def main(args):
    # Define datasets to use
    datasets = args.datasets
    # We use two GPUs for training (one A100 is not enough for LLaMA-8B)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_projector = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Model device: {device}")
    print(f"Projector device: {device_projector}")
    
    base_model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens = args.max_new_tokens,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create base LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=['down_proj', 'gate_proj', 'up_proj'],
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
    )
    # Convert to PeftModel
    model = get_peft_model(base_model, peft_config)
    # Prepare training data from all datasets
    training_samples = prepare_training_data_multi_datasets(args, tokenizer, datasets)
    dataset = create_lora_passage_dataset(training_samples)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=TrainingDataCollator(tokenizer, device, model)
    )
    print(f"initialize projector with {args.projector_p} hidden layers")
    # Initialize projector
    projector = ParameterTranslator(
        ["down_proj", "up_proj", "gate_proj"],
        list(range(model.config.num_hidden_layers)),
        model.config.hidden_size,
        model.config.intermediate_size,
        args.lora_rank,
        args.projector_p,
    ).to(device_projector)
    projector.train()
    model.eval()
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.dyprag_learning_rate)
    
    # Initialize loss tracking
    losses = defaultdict(list)
    DYPRAG_TRAIN_EPOCH = args.dyprag_train_epochs
    total_time = 0
    
    for epoch in range(DYPRAG_TRAIN_EPOCH):
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            optimizer.zero_grad()
            input_embeds = batch['input_embeds'][0]
            adapter_path = batch['adapter_paths'][0]
            model_inputs = batch['model_inputs'][0]
            lora_state_dict = load_file(os.path.join(adapter_path, "adapter_model.safetensors"), device=device_projector.type)
            with torch.no_grad():
                model.load_adapter(adapter_path, adapter_name="default")
                model.set_adapter("default")
                lm_origin_outputs = model(**model_inputs)
                lm_origin_loss = lm_origin_outputs.loss
                for name, param in model.named_parameters():
                    if "lora_" in name: # Set LoRA weights to zero
                        param.data.zero_() 

            outputs = projector(input_embeds.to(device_projector).to(torch.float32))
            # Move outputs to model device before injection
            outputs = {k: v.to(device) for k, v in outputs.items()}
            delta_inject(model, outputs)
            # Language Modeling Loss
            with torch.set_grad_enabled(True):
                # Get language model loss
                lm_outputs = model(**model_inputs)
                lm_loss = lm_outputs.loss
                # KL Divergence Loss
                kl_loss = F.kl_div(F.log_softmax(lm_outputs.logits, dim=-1), F.softmax(lm_origin_outputs.logits, dim=-1), reduction='batchmean') * 0.01
            # MSE Loss
            mse_loss = torch.tensor(0.0, device=device_projector, dtype=torch.float32)  
            for key in outputs:
                if key in lora_state_dict:
                    mse_loss += F.mse_loss(outputs[key].to(device_projector).to(torch.float32), lora_state_dict[key].to(device_projector).to(torch.float32))
            mse_loss *= 100
            # Total Loss
            total_loss = lm_loss.to(device_projector) + kl_loss.to(device_projector) + mse_loss

            # Backward
            total_loss.backward()
            optimizer.step()
            delta_remove(model, outputs)
            del outputs
            torch.cuda.empty_cache()
            print(f"Stage 2: Epoch {epoch}, Step {step}, MSE Loss: {mse_loss.item():.4f}, LM Loss: {lm_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}, Target LM Loss: {lm_origin_loss.item():.4f}")
        save_dir = os.path.join(ROOT_DIR, "projector", f'{args.model_name}_hidden{args.projector_p}_sample{args.sample_rate}_lr{args.dyprag_learning_rate}')
        os.makedirs(save_dir, exist_ok=True)
        print(f"Save projector to {save_dir}")
        torch.save({'model_state_dict': projector.state_dict()}, os.path.join(save_dir, f'epoch_{epoch}.pt'))

    print(f"Finished DyPRAG Training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, )
    parser.add_argument("--max_new_tokens", type=int, )
    parser.add_argument("--datasets", type=list, default=["2wikimultihopqa", "hotpotqa", "popqa", "complexwebquestions"])
    # Previous parameterizing settings 
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    # DyPRAG training settings
    parser.add_argument("--dyprag_train_epochs", type=int, description="number of DyPRAG training epochs")
    parser.add_argument("--sample_rate", type=float, default=0.2)
    parser.add_argument("--dyprag_learning_rate", type=float, default=1e-5)
    parser.add_argument("--projector_p", type=int, default=32)
    parser.add_argument("--augment_model", type=str, default=None)
    args = parser.parse_args()
    
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)