# pipeline.py
import asyncio
from blip2chat_async import BLIP2Chat
# from bert_question_answering import BertQuestionAnswerer
# from gpt2_question_answering import GPT2QuestionAnswerer
# from roberta_question_answering import RobertaQuestionAnswerer
# from ensemble_analysis_async import choose_best_guess
from bert_predictor_new import BertCategoryPredictor
from question_answerer import QuestionAnswerer

from data_preparation import DataPreparation
# from bert_training import BertTraining
# from bert_prediction import BertPrediction
# from transformers import BertTokenizer
from custom_dataset import CustomDataset
# from bert_embeddings import BertEmbeddings

fine_tuned_model_path = 'DeclanBracken/BERT_uncased_for_binary_TO_recycling_classification_augmented'
dataset_path = 'expanded_recycling_dataset_mapping_filtered.csv'
# dataset_path = 'expanded_recycling_dataset_mapping_filtered.csv'
input_column = 'keywords'
output_column = 'category'

# data_preparation = DataPreparation(dataset_path, input_column, output_column)
# df = data_preparation.load_data()



class ImageRecyclingClassifier:
    def __init__(self):
        self.blip_chat = BLIP2Chat(width=300, height=200)
        BLIP2Chat.preload_model_and_processor()
        # self.qa_model_bert = BertQuestionAnswerer()
        # self.qa_model_gpt2 = GPT2QuestionAnswerer()
        # self.qa_model_roberta = RobertaQuestionAnswerer()
        self.question_answerer = QuestionAnswerer()
        self.df = DataPreparation(dataset_path, input_column, output_column).load_data()
        self.category_predictor = BertCategoryPredictor(fine_tuned_model_path, self.df)
        

    async def classify_image(self, image_path):
        await self.blip_chat.load_image_async(image_path)
        caption = self.blip_chat.caption_and_display_image()
        self.blip_chat.clear_memory()
    
        # Normalize and strip the caption
        normalized_caption = caption.strip().replace('\n', ' ').replace('\r', ' ')
        print(f"Generated caption: '{caption}'")  # Log the caption
        print(f"Normalized caption for QA models: '{normalized_caption}'")


        question = f"What is the main item in the image described as {normalized_caption}?"
        print(f"Question: '{question}'")  # Log the question

        # Call each model and log their outputs
        # answer_bert = self.qa_model_bert.answer_question(normalized_caption, question)
        # print(f"BERT answer: '{answer_bert}'")
        
        # answer_gpt2 = self.qa_model_gpt2.answer_question(normalized_caption)
        # print(f"GPT-2 answer: '{answer_gpt2}'")
        
        # answer_roberta = self.qa_model_roberta.answer_question(normalized_caption, question)
        # print(f"RoBERTa answer: '{answer_roberta}'")
        bert_answer, gpt2_answer, roberta_answer, best_answer = self.question_answerer.answer_question(question)
        print(f"BERT answer: '{bert_answer}'")
        print(f"GPT-2 answer: '{gpt2_answer}'")
        print(f"RoBERTa answer: '{roberta_answer}'")

        # Check if any of the answers is null or unexpected
        # if not answer_bert or not answer_gpt2 or not answer_roberta:
        #     print("One or more models returned null or unexpected results.")

        # best_guess = choose_best_guess([answer_bert, answer_gpt2, answer_roberta])
        
        predicted_category, confidence = self.category_predictor.predict_category(best_answer)
        print("Confidence: ", "{:.2f}".format(confidence*100), "%")

        closest_category = self.category_predictor.find_closest_title(best_answer, predicted_category)
        return best_answer, predicted_category, closest_category

    def run_classification(self, image_path):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.classify_image(image_path))
