import os

import bitsandbytes as bnb
import wandb
from dotenv import load_dotenv
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from datasets import Dataset


def generate_prompt(text, label):
    return f"""
    Classify the text into spam, ham and return the answer as the corresponding label. 
    text: {text} 
    label: {label}""".strip()


class SpamLLamaClassificationModel:
    def __init__(self, model_id, cache_dir, device_map="auto", torch_dtype="float16", trust_remote_code=True):
        load_dotenv()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code

        # Load tokenizer and model
        self.tokenizer, self.model = self.load_pretrained_model()
        self.pipeline = self.initialize_pipeline()
        wandb.login()

    def load_pretrained_model(self):
        """Load and return the tokenizer and model for a specific model_id."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
            quantization_config=bnb_config,
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer, model

    def initialize_pipeline(self):
        """Initialize and return the text-generation pipeline and tokenizer."""
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )

    @staticmethod
    def generate_test_prompt(data_point):
        return f"""
        Classify the text into spam, ham and return the answer as the corresponding label. 
        text: {data_point}
        label: """.strip()

    def predict(self, X_test):
        test = X_test.apply(self.generate_test_prompt)
        y_pred = []

        for i in tqdm(range(len(test))):
            prompt = test.iloc[i]
            result = self.pipeline(prompt, max_new_tokens=2)
            answer = result[0]['generated_text'].split("label:")[-1].strip()

            # Determine the predicted category
            if "spam" in answer:
                y_pred.append("spam")
            elif "ham" in answer:
                y_pred.append("ham")
            else:
                y_pred.append("ham")  # Default to ham if unclear

        return y_pred

    def fit(self, X_train, y_train, X_eval, y_eval, output_dir, num_train_epochs=8, per_device_train_batch_size=4,
            gradient_accumulation_steps=2, name="spam_classifier", lora_rank=64):
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

        modules = find_all_linear_names(self.model)

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0,
            r=lora_rank,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules,
        )

        X_train_prompts = [{"text": generate_prompt(x, y)} for x, y in zip(X_train, y_train)]
        X_eval_prompts = [{"text": generate_prompt(x, y)} for x, y in zip(X_eval, y_eval)]

        train_data = Dataset.from_list(X_train_prompts)
        eval_data = Dataset.from_list(X_eval_prompts)

        training_arguments = TrainingArguments(
            output_dir=output_dir,  # directory to save and repository id
            
            num_train_epochs=num_train_epochs,  # number of training epochs
            per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
            gradient_accumulation_steps=gradient_accumulation_steps,
            # number of steps before performing a backward/update pass
            gradient_checkpointing=False,  # use gradient checkpointing to save memory
            optim="paged_adamw_32bit",
            logging_steps=1,
            learning_rate=2e-4,  # learning rate, based on QLoRA paper
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
            max_steps=-1,
            warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
            group_by_length=False,
            lr_scheduler_type="cosine",  # use cosine learning rate scheduler
            report_to="wandb",  # report metrics to w&b
            run_name=name,
            eval_strategy="steps",
            eval_steps=100
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=peft_config,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

def test_spam_llama_classification_model():
    from toy_dataset import DatasetProcessor
    from spam_model_eval import evaluate

    # Initialize the processor
    processor = DatasetProcessor()

    # Preprocess the dataset
    processor.preprocess_dataset()

    # Split into train, eval, and test datasets
    X_train, X_eval, X_test, y_train, y_eval, y_test = processor.split_data()

    model = SpamLLamaClassificationModel(model_id="meta-llama/Llama-3.1-8B-Instruct", cache_dir="/home/ravich/models")
    y_pred = model.predict(X_test)
    evaluate(y_test, y_pred)

    model.fit(X_train, y_train, X_eval, y_eval, output_dir="/home/ravich/models/fine-tuned-model")

    y_pred = model.predict(X_test)
    evaluate(y_test, y_pred)

if __name__ == "__main__":
    test_spam_llama_classification_model()