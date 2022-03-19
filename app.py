#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
unmasker("War is [MASK].")


# In[2]:


from transformers import pipeline
classifier = pipeline("sentiment-analysis")
classifier("I am impressed with their fast and friendly service.")


# In[3]:


from transformers import pipeline

q_a = pipeline("question-answering")

context = "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System, being larger than only Mercury. In English, Mars carries the name of the Roman god of war and is often referred to as the Red Planet. The latter refers to the effect of the iron oxide prevalent on Mars's surface, which gives it a reddish appearance distinctive among the astronomical bodies visible to the naked eye.[18] Mars is a terrestrial planet with a thin atmosphere, with surface features reminiscent of the impact craters of the Moon and the valleys, deserts and polar ice caps of Earth."
question = "Who is the Roman God of war?"
q_a({"question": question, "context": context})


# In[4]:


context


# In[5]:


#!pip3 install transformers==4.11.2 soundfile sentencepiece torchaudio pydub pyaudio

#!pip install transformers –U

import torch
import soundfile as sf
# import librosa
import os
import torchaudio

# model_name = "facebook/wav2vec2-base-960h" # 360MB
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
model_name = "facebook/wav2vec2-large-960h-lv60-self" # 1.18GB

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)


# In[6]:


# load our wav file
speech, sr = torchaudio.load("Welcome.wav")
speech = speech.squeeze()
# or using librosa
# speech, sr = librosa.load(audio_file, sr=16000)
sr, speech.shape

# resample from whatever the audio sampling rate to 16000
resampler = torchaudio.transforms.Resample(sr, 16000)
speech = resampler(speech)
speech.shape

# tokenize our wav
input_values = processor(speech, return_tensors="pt", sampling_rate=16000)["input_values"]
input_values.shape

# perform inference
logits = model(input_values)["logits"]
logits.shape

# use argmax to get the predicted IDs
predicted_ids = torch.argmax(logits, dim=-1)
predicted_ids.shape

# decode the IDs to text
transcription = processor.decode(predicted_ids[0])
transcription.lower()


# In[7]:


from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests


# In[8]:


#url = 'http://images.cocodataset.org/val2017/000000039769.jpg’
#image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("Pug.jpg")
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])


# In[9]:


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_tokenizer_and_model(model="microsoft/DialoGPT-large"):
  """
    Load tokenizer and model instance for some specific DialoGPT model.
  """
  # Initialize tokenizer and model
  print("Loading model...")
  tokenizer = AutoTokenizer.from_pretrained(model)
  model = AutoModelForCausalLM.from_pretrained(model)
  
  # Return tokenizer and model
  return tokenizer, model


def generate_response(tokenizer, model, chat_round, chat_history_ids):
  """
    Generate a response to some user input.
  """
  # Encode user input and End-of-String (EOS) token
  new_input_ids = tokenizer.encode(input(">> You:") + tokenizer.eos_token, return_tensors='pt')

  # Append tokens to chat history
  bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_round > 0 else new_input_ids

  # Generate response given maximum chat length history of 1250 tokens
  chat_history_ids = model.generate(bot_input_ids, max_length=1250, pad_token_id=tokenizer.eos_token_id)
  
  # Print response
  print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
  
  # Return the chat history ids
  return chat_history_ids


def chat_for_n_rounds(n=5):
  """
  Chat with chatbot for n rounds (n = 5 by default)
  """
  
  # Initialize tokenizer and model
  tokenizer, model = load_tokenizer_and_model()
  
  # Initialize history variable
  chat_history_ids = None
  
  # Chat for n rounds
  for chat_round in range(n):
    chat_history_ids = generate_response(tokenizer, model, chat_round, chat_history_ids)


if __name__ == '__main__':
  chat_for_n_rounds(5)


# In[ ]:





# In[ ]:





# In[ ]:




