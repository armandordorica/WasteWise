import torch
from transformers import BertModel, BertTokenizer

class BertEmbeddings:
    def __init__(self, model_name='bert-base-uncased'):
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def get_embedding(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def closest_sentence(self, input_sentence, sentence_list):
        input_embedding = self.get_embedding(input_sentence)

        max_similarity = -float('Inf')
        closest_sentence = None
        for sentence in sentence_list:
            sentence_embedding = self.get_embedding(sentence)
            similarity = torch.cosine_similarity(input_embedding, sentence_embedding).item()
            if similarity > max_similarity:
                max_similarity = similarity
                closest_sentence = sentence

        return closest_sentence
