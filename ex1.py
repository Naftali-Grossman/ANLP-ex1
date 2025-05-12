from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, TrainingArguments,Trainer, DataCollatorWithPadding, TrainerCallback, default_data_collator
import numpy as np
from datasets import load_dataset
from evaluate import load
import numpy as np
import wandb
import random
import torch
from scipy.special import softmax
import argparse
import os


os.environ["WANDB_MODE"] = "disabled"

def train_predict_mrpc_data(max_train_samples :int = 0, 
                            max_eval_samples :int = 0, 
                            max_predict_samples:int = 0, 
                            num_train_epochs:int = 3,
                            lr:float = 2e-5,  
                            batch_size:int = 16, 
                            do_train:bool = True, 
                            do_predict:bool = True,
                            model_path:str = "bert-base-uncased",
                            wandb_project:str = "bert-mrpc-paraphrase"
                            ):
    #Here we set the seed for the random number generator
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #load dataset
    mrpc_dataset = load_dataset("nyu-mll/glue", "mrpc")
    #load_parameters
    config :dict = {'learning_rate': lr, 'num_train_epochs': num_train_epochs, 'per_device_train_batch_size': batch_size}
    #load_metric
    metric = load("glue", "mrpc")
    #compute_metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return metric.compute(predictions=preds, references=labels)
    #load_model
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, num_labels=2)
    #load_tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    #load_data

    
    
    def preprocess_mrpc(example):
        return tokenizer(
            example["sentence1"],
            example["sentence2"],
            truncation=True
        )
    #preprocess_mrpc
    preprocessed_mrpc_dataset = mrpc_dataset.map(preprocess_mrpc, batched=True)
    #data_collator for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #For training
    if do_train:
        #login to wandb
        wandb.login()  # You only need to do this once
        wandb.init(
        project=wandb_project,
        name=f"mrpc-lr{config['learning_rate']}-bs{config['per_device_train_batch_size']}",
        config=config,
    )

        # Ensure max_train_samples doesn't exceed dataset size
        actual_max_train_samples = min(max_train_samples, len(preprocessed_mrpc_dataset["train"])) if max_train_samples != 1 else len(preprocessed_mrpc_dataset["train"])
        train_dataset = preprocessed_mrpc_dataset["train"].select(range(actual_max_train_samples))
        #Ensure max_eval_samples doesn't exceed dataset size
        actual_max_eval_samples = min(max_eval_samples, len(preprocessed_mrpc_dataset["validation"])) if max_eval_samples != 1 else len(preprocessed_mrpc_dataset["validation"])
        eval_dataset = preprocessed_mrpc_dataset["validation"].select(range(actual_max_eval_samples))


        #Preparing Training Arguments
        training_args = TrainingArguments(
      output_dir='saved_model',
      eval_strategy="epoch",
      save_strategy="epoch",
      logging_strategy="steps",
      logging_steps=1,  # Log loss every step
      per_device_train_batch_size=config["per_device_train_batch_size"],
      per_device_eval_batch_size=64,
      num_train_epochs=config["num_train_epochs"],
      learning_rate=config["learning_rate"],
      load_best_model_at_end=True,
      metric_for_best_model="accuracy",
      report_to="wandb",
      run_name=f"run-mrpc",
  )
        #Preparing Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        #Training
        trainer.train()
        #Evaluating
        trainer.evaluate()
        #Finishing Wandb
        wandb.finish()  
        #Saving Model
        trainer.save_model('saved_model')

    #For prediction
    if do_predict:
        #If training was done, load the model
        if do_train:
            model = AutoModelForSequenceClassification.from_pretrained('saved_model')
        #Ensure max_predict_samples doesn't exceed dataset size
        actual_max_predict_samples = min(max_predict_samples, len(preprocessed_mrpc_dataset["test"])) if max_predict_samples != 1 else len(preprocessed_mrpc_dataset["test"])
        predict_dataset = preprocessed_mrpc_dataset["test"].select(range(actual_max_predict_samples))
        model.eval()
        #Predicting
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
       
        #Obtaining Predictions
        output = trainer.predict(predict_dataset)
        metrics = compute_metrics((output.predictions, output.label_ids))
        #Printing Metrics
        
        print(metrics)
            
        #Computing Probabilities In order to obtain the top K predictions
        probs = softmax(output.predictions, axis=1)
        preds = np.argmax(probs, axis=1)
        #Getting True Labels
        true_labels = output.label_ids

        # Confidence of the predicted class
        confidences = probs[np.arange(len(probs)), preds]
        #Getting Correct Predictions
        correct_mask = preds == true_labels
        correct_indices = np.where(correct_mask)[0]
        #Sorting Correct Predictions
        sorted_correct = correct_indices[np.argsort(-confidences[correct_indices])]
        #Getting Top K Predictions
        top_k = min(20, len(sorted_correct)-1)
        top_k_correct = sorted_correct[:top_k]
        #Writing Predictions to Predictions File
        for ind in top_k_correct:
            index_num = int(ind)
            with open('predictions.txt', 'a') as f:
                f.write(f"{mrpc_dataset['test'][index_num]['sentence1']}")
                f.write("###")
                f.write(f"{mrpc_dataset['test'][index_num]['sentence2']}")
                f.write("###")
                f.write(f"{int(predict_dataset[index_num]['label'] == 1)}\n")

if __name__ == "__main__":
    #Parsing Arguments
    parser = argparse.ArgumentParser(description="Train and evaluate BERT on MRPC")

    parser.add_argument("--max_train_samples", type=int, default=1)
    parser.add_argument("--max_eval_samples", type=int, default=1)
    parser.add_argument("--max_predict_samples", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default="bert-base-uncased")
    parser.add_argument("--wandb_project", type=str, default="bert-mrpc-paraphrase")

    args = parser.parse_args()

    train_predict_mrpc_data(
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_predict_samples=args.max_predict_samples,
        num_train_epochs=args.num_train_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        do_train=args.do_train,
        do_predict=args.do_predict,
        model_path=args.model_path,
        wandb_project=args.wandb_project
    )







