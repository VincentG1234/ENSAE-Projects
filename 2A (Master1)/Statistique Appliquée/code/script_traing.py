from collections import defaultdict
from tqdm import tqdm
import requests
import torch

from transformers import pipeline
from transformers import GPT2Tokenizer, AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
from transformers import GPT2LMHeadModel, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import scripts.config_train_model_gpt as conf
import wandb
from datasets import load_metric

import shutil


def perplexity(model, encodings, context_length=256):            # Code copié du site https://huggingface.co/docs/transformers/perplexity
    max_length = model.config.n_positions
    stride = context_length
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # metrics from the datasets library have a `compute` method
    return accuracy_metric.compute(predictions=predictions, references=labels)


def train_phase(
    model_engine='gpt2',
    tokenizer_engine='gpt2-medium',
    LOGIN='vincentg',
    file='tokenized_256_top100.json',
    train_val_size = [0.95, 0.05],
    context_length=256,
    n_embd=256,
    n_layer=8,
    n_head=6,
    learning_rate=3e-3,
    epochs = 1,
    batch_size = 16,
    model_=None,
    reprise=None,
    warmup_steps= 1000
):
    assert n_embd%n_head == 0  # Le nombre de head doit diviser la dimension de l'embedding

    
    if DEVICE == "cpu":
        GPU=False
    else:
        GPU=True

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_engine)

    if reprise == None:
    
        name = f'{n_embd}_{n_layer}_{n_head}_{learning_rate}_{train_val_size[0]}_{file[10:-5]}_{epochs}'

        # start a new wandb run to track this script
        wandb.init(
        # set the wandb project where this run will be logged
        project="models 256 de context length",
        name= name,
    
        # track hyperparameters and run metadata
        config={
                  "architecture": "GPT2",
                  "dataset": "wikipedia",
                    "epochs": epochs,
                                       })

        config = AutoConfig.from_pretrained(
                                     model_engine,
                                     vocab_size=len(tokenizer),
                                      n_embd= n_embd,
                                      n_layer= n_layer,
                                      n_head= n_head,
                                      n_ctx=context_length,
                                       bos_token_id=tokenizer.bos_token_id,
                                       eos_token_id=tokenizer.eos_token_id
                                                                          )

        print('Config : check.')
    
        model = GPT2LMHeadModel(config)
    else:

        name = reprise
        wandb.init(
        # set the wandb project where this run will be logged
        project="models 256 de context length",
        name= name,
    
        # track hyperparameters and run metadata
        config={
                  "architecture": "GPT2",
                  "dataset": "wikipedia",
                    "epochs": epochs,
                                       })
        model = model_
        
        
        
    print('Nom de la sauvegarde : ', name)

    if file == 'tokenized_128_top10.json' or file == 'tokenized_256_top10.json':
        print(f"Taille en Go du train dataset: {train_val_size[0] * 1.2}")
        wandb.log({"taille Go": train_val_size[0] * 1.2})
    elif file == 'tokenized_128_top100.json' or file=='tokenized_256_top100.json':
        print(f"Taille en Go du train dataset: {train_val_size[0] * 5.8}")
        wandb.log({"taille Go": train_val_size[0] * 5.8})

    with fs.open(LOGIN+"/StatApp/"+file, 'r') as f:
        data_tokenized = Dataset.from_dict({"input_ids": json.load(f)})
        
    size = len(data_tokenized)
    size_train = int(size*train_val_size[0])
    size_val = int(size*train_val_size[1])

    
    all_tokenized = DatasetDict(
    {
        "train": data_tokenized.select(range(size_train)),  
        "valid": data_tokenized.select(range(size-1,size-size_val, -1)),      # On part de la fin pour sélectionner le dataset d'évaluation, 
                                                                            # Cela permet de garder le même tout en variant le train
    })
    
  
    
    print('Nombre de poids dans le modèle : ', model.num_parameters())

    wandb.log({"Nombre de params en millions": round( model.num_parameters(), -6)/1000**2})
    print(f"pourcentage des poids sur l'embedding : {n_embd * len(tokenizer) /model.num_parameters() * 100}")
    # Juste pour être sûr, avec une dépendance non linéaire en les arguments.
    
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir= 'test',    
        per_device_train_batch_size= batch_size,          
        per_device_eval_batch_size=batch_size,          
        evaluation_strategy="steps",
        logging_steps=10,
        gradient_accumulation_steps=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        learning_rate=learning_rate,
        save_steps=40_000,
        fp16= GPU,  # Si GPU == False, c'est horriblement lent.
        report_to = "wandb",
        run_name = "my_run_test",
        do_eval=True,
        eval_steps = 400,
        prediction_loss_only = True,
        load_best_model_at_end=True,
        metric_for_best_model = "eval_loss",
        
    )

    print('Args : check.')

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=all_tokenized["train"],
        eval_dataset=all_tokenized["valid"],
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
        
    )

    print('Trainer : check.')

    trainer.train()

    print('Train : tchou-tchou.')

    perplexity_result = perplexity(model, encoded_test)

    wandb.log({"perplexity": perplexity_result})

    return tokenizer, model, trainer, name

    

def local_save(tokenizer, model, trainer, name):
    model.save_pretrained(f'./model_{name}')
    tokenizer.save_pretrained(f'./model_{name}')
    train_result = pd.DataFrame(trainer.state.log_history)
    train_result.to_csv(f'./train_results/{name}.csv')
    print('Sauvegardes tempos du modèle : check.')

def minIO_save(LOGIN, name):
    fs.put(f'./model_{name}', f'{LOGIN}/Modèles/{name}', recursive=True)
    print('Sauvegarde MinIO : check.')



def generate_text_gpt2(model=None, tokenizer=None, seed_text=None, length=100, temperature=1.0, sample= True):
    model.eval()

    # Tokenizer le texte initial
    input_ids = tokenizer.encode(seed_text, return_tensors="pt")

    # Générer du texte avec l'attention mask spécifiée
    with torch.no_grad():
        input_ids = input_ids.to(DEVICE)

        if sample==True:
            output = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            num_return_sequences=3,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id)
        else:
            output = model.generate(input_ids,
            max_new_tokens = 100,
            num_beams = 5,
            no_repeat_ngram_size = 2,
            num_return_sequences = 5,
            early_stopping = True,
            temperature = temperature,
            pad_token_id=tokenizer.eos_token_id)
            

    # Convertir les IDs de tokens en texte
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

def generate_multiple(
    model=None, 
    tokenizer=None, 
    prompt_list='prompts_simples.json', 
    length=100, 
    temperature=1.0,
    sample=True):
    
    generated_texts = {}
    
    with open(prompt_list, 'r') as file:
        prompts = json.load(file)
        
    for prompt in prompts:
        gen_text = generate_text_gpt2(
            model=model,
            tokenizer=tokenizer,
            seed_text=prompt,
            length=length,
            temperature=temperature,
            sample=sample
        )
        generated_texts[prompt] = gen_text
        
    return generated_texts        

    

def main(
    LOGIN='vincentg',
    file='tokenized_256_top100.json',
    train_val_size = [0.95, 0.05],
    context_length=256,
    n_embd=256,
    n_layer=8,
    n_head=8,
    learning_rate=3e-3,
    epochs = 1,
    batch_size = 16,
    model_=None,
    reprise= None,
    warmup_steps= 1000):

    # Les paramètres en n_ par défaut donnent un petit modèle, de l'ordre de 7M paramètres.
    
    tokenizer, model, trainer, name = train_phase(
        LOGIN=LOGIN,
        file=file,
        train_val_size = train_val_size,
        context_length=context_length,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        learning_rate=learning_rate,
        model_ = model_,
        reprise = reprise,
        warmup_steps=warmup_steps,
        epochs=epochs
    )
    
    local_save(tokenizer, model, trainer, name)
    minIO_save(LOGIN, name)

    device = torch.device(DEVICE) 
    pipe = pipeline(
        "text-generation", model=f'./model_{name}', device=device
    )

    generated_simples = generate_multiple(sample=True,
        model=model, 
        tokenizer=tokenizer, 
        prompt_list='prompts_simples.json', 
        length=100, 
        temperature=1.0)

    with open(f'./generated_texts/simple_{name}.json', 'w') as file:
        json.dump(generated_simples, file, indent=2)

    print('Retours sur prompts simples : check.') 

    generated_complexes = generate_multiple(
        model=model, 
        tokenizer=tokenizer, 
        prompt_list='prompts_complexes.json', 
        length=100, 
        temperature=1.0)

    # Ici il faudrait sauvegarder les deux retours en tenant compte du name

    with open(f'./generated_texts/complexes_{name}.json', 'w') as file:
        json.dump(generated_complexes, file, indent=2)

    print('Retours sur prompts complexes : check.')

    wandb.finish()

    return tokenizer, model, trainer, name, generated_simples, generated_complexes


    
# model_path = "./saved_model"  # Assurez-vous que le chemin est correct

# # Charger le modèle pré-entraîné et le tokenizer
# model = GPT2LMHeadModel.from_pretrained(model_path)
# tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# # Définir le jeton de fin de séquence comme jeton de remplissage
# model.config.pad_token_id = tokenizer.eos_token_id