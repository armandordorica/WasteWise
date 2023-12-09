# bert_predictor.py
# from transformers import BertTokenizer
from finetunedbert import BERTClassifier
from bert_embeddings import BertEmbeddings

class BertCategoryPredictor:
    def __init__(self,model_identifier, df):
        self.bert_prediction = BERTClassifier(model_identifier)
        self.df = df
        self.embeddings = BertEmbeddings()
    
    def predict_category(self, description):
        return self.bert_prediction.predict_category(description)

    def find_closest_title(self, best_guess, predicted_category):
        input_sentence = best_guess + " " + predicted_category
        sentence_list = list(self.df[self.df['category'] == predicted_category]['title'].unique())
        closest_category = self.embeddings.closest_sentence(input_sentence, sentence_list)
        return closest_category
