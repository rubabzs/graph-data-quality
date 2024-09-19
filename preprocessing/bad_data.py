import random
import pandas as pd


def missing(node):
  # Introduce missing values
  prop_r = random.choice(['text', 'summary', 'userId', 'time', 'profileName'])
  prop_p = random.choice(['name', 'price'])

  if node['type'] == 'Product':
    node['properties'][prop_p] = None
  if node['type'] == 'Review':
    node['properties'][prop_r] = None

  node['label'] = "Invalid"
  node['dq_issue'] = 'Missing'

  return node

def incorrect(node):
  # Introduce incorrect values
  if node['type'] == 'Product':
    node['properties']['price'] = random.randrange(-100000,100000)
  if node['type'] == 'Review':
    node['properties']['score'] = random.randrange(-100000,100000)

  node['label'] = "Invalid"
  node['dq_issue'] = 'Incorrect'

  return node

def inconsistent(node):
  if node['type'] == 'Review':
    inconsistent_prop = random.choice(list(node['properties'].keys()))
    if inconsistent_prop == 'score':
      node['properties'][inconsistent_prop] = 'Excellent'
    elif inconsistent_prop == 'time':
      node['properties'][inconsistent_prop] = 'Yesterday'
    elif inconsistent_prop == 'userId':
      node['properties'][inconsistent_prop] = random.randrange(-100000,100000)
    elif inconsistent_prop == 'profileName':
      node['properties'][inconsistent_prop] = random.randrange(-100000,100000)
    elif inconsistent_prop == 'summary' or inconsistent_prop == 'text':
      if node['properties']['summary']:
        if 'Great' in node['properties']['summary']:
          node['properties']['text'] = 'Very bad'
        else:
          node['properties'][inconsistent_prop] = random.randrange(-100000,100000)


  node['label'] = "Invalid"
  node['dq_issue'] = 'Inconsistent'

  return node

def introduce_data_quality_issues(data, issue_types, issue_ratio=0.1):
    """
    Introduce data quality issues into the dataset and label nodes accordingly.

    Args:
        data (list): List of product and review nodes.
        issue_types (list): List of issue types to introduce (e.g., 'missing', 'incorrect', 'inconsistent', 'duplicate').
        issue_ratio (float): Ratio of data to introduce issues in.

    Returns:
        list: Data with introduced quality issues and labels.
    """
    data_with_issues = copy.deepcopy(data)
    num_issues = int(len(data) * issue_ratio)
    unique_data = {node['id']: node for node in data_with_issues}.values() 
    unique_data = list(unique_data)
    ds = []

    for i in range(0, min(100, len(unique_data))):
      node = unique_data[i]
      ds.append(missing(node))

    for i in range(101, min(200, len(unique_data))):
      node = unique_data[i]
      ds.append(incorrect(node))

    for i in range(201, min(300, len(unique_data))):
      node = unique_data[i]
      ds.append(inconsistent(node))

    for i in range(301, len(unique_data)):
      node = unique_data[i]
      node = missing(node)
      node = incorrect(node)
      node = inconsistent(node)
      ds.append(node)


    return data_with_issues, ds


# Function to introduce data quality issues and keep track of them
def introduce_relationship_issues(sampled_data, misconfigured_relationships, issue_type='all'):
    """
    Introduce specified data quality issues into the relationships of the graph.

    Parameters:
    sampled_data (list): List of sampled nodes from the dataset.
    misconfigured_relationships (dict): Dictionary to keep track of misconfigured relationships.
    issue_type (str): Type of issue to introduce. Default is 'all'.
    """
    sampled_data = {node['id']: node for node in sampled_data}.values()  # Ensure unique nodes by ID
    sampled_data = list(sampled_data)
    ds = []

    if issue_type in ['all', 'missing']:
        # Missing Relationships: Remove some edges
        for node in sampled_data[:20]:
            if node['type'] == 'Review' and node['relationships']:
                #if random.random() < 0.2:  # 10% chance to remove the edge
              removed_relationship = node['relationships'].pop(random.randint(0, len(node['relationships']) - 1))
              misconfigured_relationships['missing'].append((node['id'], removed_relationship['target_id']))
              node['dq_issue'] = 'Missing Relationship'
              ds.append(node)

    if issue_type in ['all', 'incorrect']:
        for node in sampled_data[21:40]:
            if node['type'] == 'Review':
              node['relationships'].append({'type': 'HAS_REVIEW', 'target_id': node['id']})
              node['dq_issue'] = 'Wrong Self-loop'
              misconfigured_relationships['incorrect'].append((node['id'], node['id']))
              ds.append(node)

    if issue_type in ['all', 'inconsistent']:
        # Inconsistent Relationships: Change relationship types
        for node in sampled_data[41:len(sampled_data)]:
            for relationship in node['relationships']:
              original_type = relationship['type']
              relationship['type'] = 'REVIEW_OF' if relationship['type'] == 'HAS_REVIEW' else 'HAS_REVIEW'
              node['dq_issue'] = 'Wrong Relationship'
              misconfigured_relationships['inconsistent'].append((node['id'], relationship['target_id'], original_type, relationship['type']))
              ds.append(node)

    return ds



def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def main():
  issue_types = ['missing', 'incorrect', 'inconsistent']
  amazon_sample = product_nodes[:200] + review_nodes[:550]
  random.shuffle(amazon_sample)
  
  data_with_issues, only_issues = introduce_data_quality_issues(amazon_sample, issue_types, issue_ratio=1)

  
  ramazon_sample = product_nodes[201:321] + review_nodes[551:671] + product_nodes[322:342]
  random.shuffle(ramazon_sample)

  bad_relationships = introduce_relationship_issues(ramazon_sample, misconfigured_relationships)

  return only_issues, bad_relationships