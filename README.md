# Project Name: Spam Detection with Fine-tuned LLaMA Model
This project demonstrates the use of a fine-tuned **LLaMA 3.1 Instruct** (8B model) for spam detection. The model leverages **transformers**, **peft/LoRA**, and **quantization** techniques to classify text into categories like "spam" and "ham." It also includes training, evaluation, and optimization workflows with **QLoRA** for efficient performance on limited hardware resources.
### Key Features
- **Pre-trained Model Loading:** Leverages `transformers` to initialize the LLaMA model and tokenizer.
- **Text Classification Pipeline:** Implements a pipeline for spam classification by adapting the model to the task.
- **Dataset Preprocessing and Splitting:** Handles dataset cleaning, deduplication, and stratified splitting for training, evaluation, and testing.
- **Model Fine-tuning with LoRA:** Adopts Low-Rank Adaptation (LoRA) to fine-tune a large language model with focus on memory efficiency.
- **Evaluation Metrics:** Provides accuracy, classification reports, and confusion matrices for performance evaluation.
- **Quantization Optimization:** Uses **bitsandbytes** library for efficient 4-bit model quantization.

### Dependencies
- `transformers`
- `torch`
- `bitsandbytes`
- `sklearn`
- `pandas`
- `tqdm`
- `datasets`
- `peft`
- `trl`

### Workflow
1. **Load Dataset:** The project uses a deduplicated dataset containing email texts labeled as spam or ham.
2. **Data Preprocessing:** Prepare the dataset by adding labels, stratified splitting, and applying custom prompts tailored for classification tasks.
3. **Fine-tuning:** Train the model using LoRA configuration for efficient adaptation and save the fine-tuned model.
4. **Prediction and Evaluation:** Generate classifications for test data, calculate metrics, and assess the model’s performance.

### Use Case
This project serves as a blueprint for:
- Implementing spam detection for emails and other text-based systems.
- Fine-tuning large language models for custom classification tasks.
- Leveraging quantization and LoRA for resource-efficient training.
