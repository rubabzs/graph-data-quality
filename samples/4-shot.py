labeled_data_amazon_fine_foods_4_shot = [
  {
    "id": "B001E4KFG0_A5GH6H7AUHU8GW_1303862400",
    "type": "Review",
    "properties": {
      "userId": "A3SGXH7AUHU8GW",
      "profileName": "jkhhjknmnmn",
      "helpfulness": "1/1",
      "score": -11115.0,
      "time": "1afaadasafa",
      "summary": "fajhjahjh2q8fuhaj",
      "text": None
    },
    "relationships": [{
        'type': 'HAS_REVIEW',
        'target_id': 'B001E4KFG0_A5GH6H7AUHU8GW_1303862400'
    }],
    "label": "Invalid"
  },
  {
    'id': 'B001E4KFGX',
    'type': 'Product',
    'properties': {
        'type': 'Product',
        'name': 'Fridge',
        'price': None
    },
    'relationships': [{
        'type': 'HAS_PRODUCT',
        'target_id': 'B001E4KFGX'
    }],
    'label': 'Invalid',
    'dq_issue': 'Missing'
  },
  {
    "id": "B001E4KFG0_A2SGXH7AUHU8GW_1303862400",
    "type": "Review",
    "properties": {
      "userId": "A3SGXH7AUHU8GW",
      "profileName": "Rubab Zahra",
      "helpfulness": "1/1",
      "score": 4.0,
      "time": "1294272000",
      "summary": "Wonderful Experience",
      "text": "This is the best cornmeal. I made regular cornbread and hot water cornbread with this meal and both were outstanding. Also fried some oysters with this meal, it gave them a great texture and flovor."
    },
    "relationships": [],
    "label": "Valid"
  },
  {
    'id': 'B001E4KFG1',
    'type': 'Product',
    'properties': {
        'type': 'Product',
        'name': 'Fridge',
        'price': 230000
    },
    'relationships': [{
        'type': 'HAS_REVIEW',
        'target_id': 'B001E4KFG0_A2SGXH7AUHU8GW_1303862400'
    }],
    'label': 'Valid'
  }
]