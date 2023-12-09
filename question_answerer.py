import re
import torch
from transformers import (
    BertTokenizer, BertForQuestionAnswering,
    GPT2Tokenizer, GPT2LMHeadModel,
    RobertaTokenizer, RobertaForQuestionAnswering
)

class QuestionAnswerer():
    def __init__(self):
        # Instantiate Bert, GPT2, and RoBERTa question answerers
        self.bert_question_answerer = BertQuestionAnswerer()
        self.gpt2_question_answerer = GPT2QuestionAnswerer()
        self.roberta_question_answerer = RobertaQuestionAnswerer()
    
    def answer_question(self, question):
        bert_answer = self.bert_question_answerer.answer_question(question)
        gpt2_answer = self.gpt2_question_answerer.answer_question(question)
        roberta_answer = self.roberta_question_answerer.answer_question(question)
        return self.choose_best_answer(bert_answer, gpt2_answer, roberta_answer)
        
    def choose_best_answer(self, bert_answer, gpt2_answer, roberta_answer):
        # If there's a consensus, choose answer
        if bert_answer == roberta_answer or bert_answer == gpt2_answer:
            best_answer = bert_answer
        elif gpt2_answer == roberta_answer and gpt2_answer != 'No usable result':
            best_answer = gpt2_answer
        # Fallback: If no consensus, choose based on preference
        # For example, let's assume we trust BERT's answers the most based on qualitative testing
        else:
          best_answer = bert_answer

        return bert_answer, gpt2_answer, roberta_answer, best_answer


class BertQuestionAnswerer:
    def __init__(self):
        # Load pre-trained tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    def answer_question(self, question):
        # Encode the question
        inputs = self.tokenizer.encode_plus(question, return_tensors='pt', add_special_tokens=True)

        # Get the model's answer
        outputs = self.model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        # Find the position of the start and end of the answer
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        # Decode and return the answer
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))


class GPT2QuestionAnswerer:
    def __init__(self):
        # Load pre-trained tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def answer_question(self, question):
        # Suppress the specific warning
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        # Remove trailing period from question (if applicable)
        question = self.remove_trailing_periods(question)

        # Encode the question
        inputs = self.tokenizer.encode_plus(question, return_tensors="pt")
        attention_mask = inputs['attention_mask']

        # Get the model's answer
        outputs = self.model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=50, num_return_sequences=1)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract and return the specific answer using regex
        pattern = r"The answer is (.*?)[.\n]"
        match = re.search(pattern, result)
        if match:
            return match.group(1)
        else:
            return "No usable result"
    
    def remove_trailing_periods(self, question):
        # Use regular expression to remove trailing periods
        question = re.sub(r'\.+$', '', question)
        return question


class RobertaQuestionAnswerer:
    def __init__(self):
        # Load pre-trained tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2')
        self.model = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')

    def answer_question(self, question):
        # Encode the question
        inputs = self.tokenizer.encode_plus(question, return_tensors='pt', add_special_tokens=True)

        # Get the model's answer
        outputs = self.model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        # Find the position of the start and end of the answer
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        # Decode the answer
        raw_answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        
        # Clean and return the answer
        return self.clean_answer(raw_answer, question)

    def clean_answer(self, raw_answer, question):
        # Remove the RoBERTa special tokens and question part
        cleaned_answer = raw_answer.replace('<s>', '').replace('</s>', '').strip()
        cleaned_answer = cleaned_answer.split(question)[-1].strip()
        if cleaned_answer is None or cleaned_answer == '':
          return "No usable result"
        else:
          return cleaned_answer
