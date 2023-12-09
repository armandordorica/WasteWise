import os
import requests
import torch
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor
import asyncio
import gc
import nest_asyncio
nest_asyncio.apply()

class BLIP2Chat:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", cache_dir=None, width=None, height=None):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), '.cache', 'huggingface', 'transformers')
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=self.cache_dir)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, cache_dir=self.cache_dir, torch_dtype=torch.float16)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        self.context = []
        self.image = None
        self.display_width = width
        self.display_height = height

    @classmethod
    def preload_model_and_processor(cls, model_name="Salesforce/blip2-opt-2.7b"):
        cache_dir = os.path.join(os.path.expanduser("~"), '.cache', 'huggingface', 'transformers')
        _ = Blip2Processor.from_pretrained(model_name, cache_dir=cache_dir)
        _ = Blip2ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)


    async def load_image_async(self, image_source):
        if image_source.startswith('http://') or image_source.startswith('https://'):
            response = await requests.get(image_source, stream=True)
            self.image = Image.open(response.raw).convert('RGB')
        else:
            self.image = Image.open(image_source).convert('RGB')
        self.display_image()

    def display_image(self):
        if self.image:
            if self.display_width is not None and self.display_height is not None:
                display_image = self.image.resize((self.display_width, self.display_height))
            else:
                display_image = self.image
            display(display_image)
        else:
            print("No image is loaded to display.")

    def ask(self, question, max_new_tokens=20, use_context=True):
        if not self.image:
            print("Please load an image before asking a question.")
            return None

        # Prepare the prompt with limited context
        if use_context:
            context_str = " ".join([f"Question: {c[0]} Answer: {c[1]}" for c in self.context[-1:]])  # Use only the most recent context
        else:
            context_str = ""
        prompt = context_str + f" Question: {question} Answer:"

        # Process the inputs
        inputs = self.processor(self.image, text=prompt, return_tensors="pt").to(self.device, torch.float16)

        # Generate an answer to the question
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Update the context with the new question and answer
        self.context.append((question, answer))

        return answer


    def caption_and_display_image(self):
        if not self.image:
            print("Please load an image before captioning.")
            return None

        # Display the image
        plt.imshow(self.image)
        plt.axis("off")  # Hide the axis
        plt.show()

        # Preprocess the image
        inputs = self.processor(text=None, images=self.image, return_tensors="pt", padding=True)

        # Move the preprocessed inputs to the device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate captions
        outputs = self.model.generate(**inputs)

        # Decode and return the caption
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        print("Generated Caption:", caption)
        return caption

    def clear_memory(self):
        if self.image:
            del self.image
        gc.collect()
