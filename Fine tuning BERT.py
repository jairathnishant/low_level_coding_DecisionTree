from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding
import pandas as pd

# dataset_dict = pd.read_csv('SPAM text message 20170820 - Data.csv')
dataset_dict = load_dataset("csv", data_files="SPAM text message 20170820 - Data.csv")

def add_label_column(example):
    example['label'] = 1 if example["Category"] == "spam" else 0
    return example

dataset_dict = {k: v.map(add_label_column) for k, v in dataset_dict.items()}

# First split
split = dataset_dict["train"].train_test_split(test_size=0.3, seed = 42)
train_dataset = split["train"]
temp_dataset = split["test"]

# Second split
val_test_split = temp_dataset.train_test_split(test_size=0.5, seed = 42)
validation_dataset = val_test_split["train"]
test_dataset = val_test_split["test"]

# combine into a datasetdict
dataset_dict = {
    "train":train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
}

# define pre-trained model path
model_path = "google-bert/bert-base-uncased"

# load model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# load model with binary classification head
id2label = {"ham": "Safe", "spam": "Not Safe"}
label2id = {"Safe": "ham", "Not Safe": "spam"}
model = AutoModelForSequenceClassification.from_pretrained(model_path, 
                                                           num_labels=2, 
                                                           id2label=id2label, 
                                                           label2id=label2id,)
for name,param in model.base_model.named_parameters():
    if not ("pooler in name"): 
        param.requires_grad = False

# define text preprocessing
def preprocess_function(examples):
    # return tokenized text with truncation
    return tokenizer(examples["Message"], truncation = True)

# tokenized_data = dataset_dict.map(preprocess_function, batched = True)

tokenized_data = {
    split: ds.map(preprocess_function, batched = True) for split, ds in dataset_dict.items()
}


# dynamically pad sequences in a batch during training to make the length same

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# load metrics
accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    # get predictions
    predictions, labels = eval_pred

    # apply softmax to get probabilities
    probabilities = np.exp(predictions)/np.exp(predictions).sum(-1, keepdims = True)

    # Use probabilities of the positive class for ROC AUC
    positive_class_probs = probabilities[:, 1]

    # compute AUC
    auc = np.round(auc_score.compute(prediction_scores = positive_class_probs, references = labels)['roc_auc'], 3)

    # predict most probable class
    predicted_classes = np.argmax(predictions, axis = 1)

    # compute accuracy
    acc = np.round(accuracy.compute(predictions = predicted_classes, references = labels)['accuracy'], 3)

    return {'Accuracy':acc, "AUC": auc}

# Fine tuning BERT model
lr = 2e-4
batch_size = 8
num_epcohs = 10

training_args = TrainingArguments(
    output_dir = "bert-phishing-classifier_teacher",
    learning_rate= lr,
    per_device_train_batch_size= batch_size,
    per_device_eval_batch_size = batch_size,
    num_train_epochs = num_epcohs,
    logging_strategy= "epoch",
    eval_strategy="epoch",
    save_strategy= "epoch",
    load_best_model_at_end= True
)

# pass the training arguments into trainer class for model training

trainer = Trainer(model = model,
                  args = training_args,
                  train_dataset= tokenized_data["train"],
                  eval_dataset= tokenized_data["test"],
                  tokenizer= tokenizer,
                  data_collator= data_collator,
                  compute_metrics= compute_metrics)

trainer.train()

# apply model to validation dataset
predictions = trainer.predict(tokenized_data["validation"])

# Extract the logits and labels from the predictions object
logits = predictions.predictions
labels = predictions.label_ids

# Use compute_metrics function
metrics = compute_metrics((logits, labels))
# print(metrics)