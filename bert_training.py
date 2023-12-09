from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertModel, EarlyStoppingCallback
from sklearn.preprocessing import LabelEncoder
from custom_dataset import CustomDataset
import joblib

class BertTraining:
    def __init__(self, data, input_col, output_col):
        self.data = data
        self.input_col = input_col
        self.output_col = output_col
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_encoder = LabelEncoder()
    
    def tokenize_and_encode(self):
        encoded_data = self.tokenizer.batch_encode_plus(
            self.data[self.input_col].tolist(),
            padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        return encoded_data
    
    def prepare_datasets(self, train_indices, val_indices, encoded_data):

        # Fit the label encoder on the full dataset
        self.label_encoder.fit(self.data[self.output_col])
        
        # train_labels = self.label_encoder.fit_transform(self.data.loc[train_indices, self.output_col])
        train_labels = self.label_encoder.transform(self.data.loc[train_indices, self.output_col])

        val_labels = self.label_encoder.transform(self.data.loc[val_indices, self.output_col])

        train_dataset = CustomDataset({key: encoded_data[key][train_indices].clone().detach() for key in encoded_data}, train_labels)
        val_dataset = CustomDataset({key: encoded_data[key][val_indices].clone().detach() for key in encoded_data}, val_labels)
        return train_dataset, val_dataset
    
    def train_model(self, train_dataset, val_dataset, model_save_path, encoder_save_path):
        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=20,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            evaluation_strategy='epoch',
            logging_dir='./logs',
            logging_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            save_strategy='epoch',
            save_total_limit=1,
        )
        # Initialize the Trainer
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=len(self.label_encoder.classes_)
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        # Train the model
        trainer.train()
        # Save the model and the label encoder
        model.save_pretrained(model_save_path)
        joblib.dump(self.label_encoder, encoder_save_path)
        return trainer
