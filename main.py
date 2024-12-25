import os

from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline,TrainingArguments,BitsAndBytesConfig
import torch
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
import bitsandbytes as bnb

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
TORCH_DTYPE = torch.float16
DEVICE_MAP = "auto"
TRUST_REMOTE_CODE = True
CACHE_DIR = "/home/ravich/models"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
TEST_SIZE = 0.3
EVAL_TEST_RATIO = 0.5
RANDOM_STATE = 42


def load_pretrained_model(model_id):
    """Load and return the tokenizer and model for a specific model_id."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP,
        trust_remote_code=TRUST_REMOTE_CODE,
        quantization_config=bnb_config,
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


def initialize_model_pipeline(tokenizer, model):
    """Initialize and return the text-generation pipeline and tokenizer."""
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP,
    )
    return text_generation_pipeline, tokenizer

def llm_hello_word(tokenizer, pipe):
    messages = [{"role": "user", "content": "What is the tallest building in the world?"}]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=120, do_sample=True)
    print(outputs[0]["generated_text"])


# Define the prompt generation functions
def generate_prompt(text, label):
    return f"""
            Classify the text into spam, ham and return the answer as the corresponding label. 
            text: {text} 
            label: {label}""".strip()

def generate_test_prompt(data_point):
    return f"""
            Classify the text into spam, ham and return the answer as the corresponding label. 
            text: {data_point}
            label: """.strip()


def predict(X_test, model, tokenizer):
    test = X_test.apply(generate_test_prompt)
    y_pred = []
    categories = ["spam", "ham"]

    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]
        pipe = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=2,
                        temperature=0.1)

        result = pipe(prompt)
        answer = result[0]['generated_text'].split("label:")[-1].strip()

        # Determine the predicted category
        for category in categories:
            if category.lower() in answer.lower():
                y_pred.append(category)
                break
        else:
            y_pred.append("none")

    return y_pred


def evaluate(y_true, y_pred):
    labels = ["ham", "spam"]
    mapping = {label: idx for idx, label in enumerate(labels)}

    def map_func(x):
        return mapping.get(x, -1)  # Map to -1 if not found, but should not occur with correct data

    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    print(f'Accuracy: {accuracy:.3f}')

    # Generate accuracy report
    unique_labels = set(y_true_mapped)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true_mapped)) if y_true_mapped[i] == label]
        label_y_true = [y_true_mapped[i] for i in label_indices]
        label_y_pred = [y_pred_mapped[i] for i in label_indices]
        label_accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {labels[label]}: {label_accuracy:.3f}')

    # Generate classification report
    class_report = classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped, target_names=labels,
                                         labels=list(range(len(labels))))
    print('\nClassification Report:')
    print(class_report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true_mapped, y_pred=y_pred_mapped, labels=list(range(len(labels))))
    print('\nConfusion Matrix:')
    print(conf_matrix)


tokenizer, model = load_pretrained_model(MODEL_ID)
pipe, tokenizer = initialize_model_pipeline(tokenizer, model)
#llm_hello_word(tokenizer, pipe)


DATASET_PATH = "/home/ravich/spam_datasets/emails.csv"
def preprocess_dataset(file_path):
    dataframe = pd.read_csv(file_path)
    dataframe.columns = ["text", "spam"]
    dataframe = dataframe.drop_duplicates()
    dataframe['label'] = dataframe['spam'].apply(lambda x: 'spam' if x == 1 else 'ham')
    return dataframe

def split_data(dataframe):
    X = dataframe['text']
    y = dataframe['label']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=EVAL_TEST_RATIO, random_state=RANDOM_STATE, stratify=y_temp)
    return X_train, X_eval, X_test, y_train, y_eval, y_test

emails_df = preprocess_dataset(DATASET_PATH)
X_train, X_eval, X_test, y_train, y_eval, y_test = split_data(emails_df)



y_pred = predict(X_test, model, tokenizer)
evaluate(y_test, y_pred)

output_dir=f"{CACHE_DIR}/{MODEL_ID}-fine-tuned-model"



def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
modules = find_all_linear_names(model)
print(modules)

from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from datasets import Dataset


X_train_prompts = [{"text": generate_prompt(x, y)} for x, y in zip(X_train, y_train)]
X_eval_prompt = [{"text": generate_prompt(x, y)} for x, y in zip(X_eval, y_eval)]

train_data = Dataset.from_list(X_train_prompts)
eval_data = Dataset.from_list(X_eval_prompt)


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules,
)

training_arguments = TrainingArguments(
    output_dir=output_dir,                    # directory to save and repository id
    num_train_epochs=1,                      # number of training epochs
    per_device_train_batch_size=1,            # batch size per device during training
    gradient_accumulation_steps=8,            # number of steps before performing a backward/update pass
    gradient_checkpointing=False,              # use gradient checkpointing to save memory
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=2e-4,                       # learning rate, based on QLoRA paper
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
    max_steps=-1,
    warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
    group_by_length=False,
    lr_scheduler_type="cosine",               # use cosine learning rate scheduler
    report_to="none",                  # report metrics to w&b
    eval_strategy="steps",              # save checkpoint every epoch
    eval_steps = 0.2
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    #dataset_text_field="text",
    tokenizer=tokenizer,
    #max_seq_length=512,
    #packing=False,
    #dataset_kwargs={
    #"add_special_tokens": False,
    #"append_concat_token": False,
    #}
)

trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

y_pred = predict(X_test, model, tokenizer)
evaluate(y_test, y_pred)