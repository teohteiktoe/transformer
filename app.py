#!/usr/bin/env python
# coding: utf-8

# In[1]:


from textblob import TextBlob


# In[2]:


from flask import Flask


# In[3]:


app = Flask(__name__)


# In[4]:


from flask import render_template,request
from transformers import pipeline
classifier = pipeline('sentiment-analysis', "mrm8488/bert-small-finetuned-squadv2")


# In[5]:


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        print(text)
        r1 = TextBlob(text).sentiment
        r2 = classifier(text)
        return(render_template("index.html", result1=r1, result2=r2))
    else:
        return(render_template("index.html", result1="2", result2="2"))


# In[ ]:


if __name__=="__main__":
    app.run()


# In[ ]:





# In[ ]:




