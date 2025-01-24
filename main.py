from llama_spam import SpamLLamaClassificationModel
#from toy_dataset import DatasetProcessor
from enron_dataset import DatasetProcessor
from spam_model_eval import evaluate

EVAL_BATCH_SIZE=1
TRAIN_BATCH=1
TRAIN_EPOCHS=3
LORA_R=8

run_name=f"train_batch_{TRAIN_BATCH}_trainepochs_{TRAIN_EPOCHS}"

print("Loading dataset...")
processor = DatasetProcessor()

print("Preprocessing dataset...")
processor.preprocess_dataset()

X_train, X_eval, X_test, y_train, y_eval, y_test = processor.split_data()

print("Loading model...")
model = SpamLLamaClassificationModel(model_id="meta-llama/Llama-3.1-8B-Instruct", cache_dir="/home/ravich/models")

print("Evaluating model...")
y_pred = model.predict(X_test)
evaluate(y_test, y_pred, run_name=run_name+"_before_fine_tuning")

print("Finetune model...")
model.fit(X_train, y_train, X_eval, y_eval, output_dir="/home/ravich/models/fine-tuned-model",
          num_train_epochs=TRAIN_EPOCHS,
          per_device_train_batch_size=TRAIN_BATCH,
          gradient_accumulation_steps=8,
          lora_rank=LORA_R,
          name=run_name+"_fine_tuning")

print("Evaluating fine tuned model...")
y_pred = model.predict(X_test)
evaluate(y_test, y_pred, run_name=run_name+"_after_fine_tuning")