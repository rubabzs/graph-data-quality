import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
from google.colab import drive
import networkx as nx
import json
import time
import random
import copy
import tiktoken
import random
import os

os.environ["DGLBACKEND"] = "pytorch"

import torch
import dgl
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from sklearn.model_selection import train_test_split

from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info


# Convert labeled node properties to text
def labeled_node_to_text(node):
    node_id = node['id']
    node_text = f"Node {node['id']}, Type: {node['type']}, "
    properties_text = ", ".join([f"{k}: {v}" for k, v in node['properties'].items()])
    relationships_text = ", ".join(f"Node {node_id} is linked with Node {node['target_id']}" for node in node["relationships"])
    return f"{node_text}{properties_text}{relationships_text}, Label: {node['label']}"


# Recheck if label is being fed as a property
# Convert unlabeled node properties to text
def unlabeled_node_to_text(node):
    node_id = node['id']
    node_text = f"Node {node['id']}, Type: {node['type']}, "
    properties_text = ", ".join([f"{k}: {v}" for k, v in node['properties'].items()])
    relationships_text = ", ".join(f"Node {node_id} is linked with Node {node['target_id']}" for node in node["relationships"])
    return f"{node_text}{properties_text}{relationships_text}"


# Set up OpenAI API key
client = OpenAI(
    api_key='<enter your key here>'
)

# Function to classify node using LLM with few-shot examples
def classify_node_with_examples(node_text, examples):
    prompt = f"""
    Task: You task is to evaluate the data quality of incoming graph node encoded as text here {node_text} into Valid or Invalid quality. Learn the patterns of valid and invalid data quality
    from the nodes here {examples}.

    Output: Only return the predicted label.
    """
    #prompt = f"Here are some examples of labeled nodes:\n\n{examples}\n\nClassify the following node:\n\n{node_text}\n\nClass:"
    response = client.chat.completions.create(
      model="gpt-4",
      messages=[{
          "role": "user",
          "content": prompt
          }
      ],
      #prompt=prompt,
      temperature=1,
      max_tokens=256,
      top_p=1,
      stop=["\n"],
      frequency_penalty=0,
      presence_penalty=0
    )

    return response.choices[0].message.content.strip()

# Function to classify node using LLM with few-shot examples
def batch_classification(prompt):
    response = client.chat.completions.create(
      model="gpt-4",
      messages=[{
          "role": "user",
          "content": prompt
          }
      ],
      #prompt=prompt,
      temperature=1,
      max_tokens=2048,
      top_p=1,
      #stop=["\n"],
      frequency_penalty=0,
      presence_penalty=0
    )

    return response.choices[0].message.content.strip()


prompts = {
    4: [],
    12: []
}

cnodes = {
    4: [],
    12: []
}

chunk_size = 5

shots_lookup = {
    4: labeled_data_amazon_fine_foods_4_shot,
    12: labeled_data_amazon_fine_foods_12_shot
}

start = time.time()

chunk_size = 2
dataset = bad_relationships


for shot in [4, 12]:
  for i in range(0, len(dataset), chunk_size):
    try:
      nodes_to_classify = dataset[i:i + chunk_size]
      prompt = create_prompt_with_examples(shots_lookup[shot], nodes_to_classify)
      res = batch_classification(prompt)

      json_array_match = re.search(r'\[.*\]', res, re.DOTALL)

      if json_array_match:
        json_array_str = json_array_match.group()
        try:
          json_array = json.loads(json_array_str)
        except json.JSONDecodeError as e:
          print("Failed to decode JSON:", e)
          print("res: ", res)
          break
      else:
        print(res)
        print("No JSON array found in the response.")
        break
      classified_nodes = json_array
      prompts[shot] += [prompt]
      cnodes[shot] += classified_nodes
    except:
      print(res)
      print("not working")
      with open(str(shot) + '_shot.json', 'w', encoding='utf-8') as f:
        json.dump(cnodes[shot], f, ensure_ascii=False, indent=4)
        end = time.time()
      print("Total time taken: {0} minutes".format(round((end-start)/60, 2)))

# Write the output to a file
#with open('gdrive/MyDrive/qdb/results/' + str(shot) + '_shot_e2.json', 'w', encoding='utf-8') as f:
#    json.dump(cnodes[shot], f, ensure_ascii=False, indent=4)

end = time.time()

print("Total time taken: {0} minutes".format(round((end-start)/60, 2)))