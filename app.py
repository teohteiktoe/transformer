#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


from flask import render_template,request


# In[4]:


name = "mrm8488/bert-small-finetuned-squadv2"
classifier = pipeline('sentiment-analysis',
                      name)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        print(text)
        r = classifier(text)
        return(render_template("index.html", result=r))
    else:
        return(render_template("index.html", result="2"))


# In[ ]:


if __name__=="__main__":
    app.run()


# In[ ]:


get_ipython().system('pip freeze')


# In[ ]:




