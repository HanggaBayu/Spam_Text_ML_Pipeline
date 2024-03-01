#!/usr/bin/env python
# coding: utf-8

# In[47]:


import requests
import tensorflow as tf
import json
import base64
from pprint import PrettyPrinter
import pandas as pd


# In[48]:


pp = PrettyPrinter()
pp.pprint(requests.get("http://localhost:8080/v1/models/spam-model").json())


# In[49]:


dataset = pd.read_csv('./Data/Spam_Data.csv')


# In[50]:


dataset


# In[51]:


dataset['Message'][2]


# ## Predict

# In[52]:


def prepare_json(text):
    feature_spec = {
        "Message": tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(text, "utf-8")]))
    }
    
    example = tf.train.Example(
        features=tf.train.Features(feature=feature_spec)
    ).SerializeToString()
    
    result = [
        {
            "examples": {
                "b64": base64.b64encode(example).decode()
            }
        }
    ]
    
    return json.dumps({
        "signature_name": "serving_default",
        "instances": result
    })


# In[53]:


text = dataset['Message'][2]


# In[54]:


text


# In[55]:


json_data = prepare_json(text)
endpoint = "http://localhost:8080/v1/models/spam-model:predict"
response = requests.post(endpoint, data=json_data)

prediction = response.json().get("predictions")

if prediction :
    prediction_value = prediction[0][0]
    result = "Ham" if prediction_value > 0.5 else "Spam"
else:
    result = "Error: No predictions found."

print(result)
    


# In[56]:


pip freeze > requirements.txt

