labeled_data_amazon_fine_foods_12_shot = [
  ## Review Node -- Invalid Cases
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
    "id": "B001E4KFG0_B5GH6H7AUHU8GW_1303862400",
    "type": "Review",
    "properties": {
      "userId": "A3SGXH7AUHU8GW",
      "profileName": "jkhhjknmnmn",
      "helpfulness": "1/1",
      "score": 2000000.0,
      "time": "090078601",
      "summary": "Great Food",
      "text": "Worst experience of my life every"
    },
    "relationships": [{
        'type': 'HAS_BOOK',
        'target_id': 'B001E4KFG0'
    }],
    "label": "Invalid"
  },
  {
    "id": "B001E4KFG0_C5GH6H7AUHU8GW_1303862400",
    "type": "Review",
    "properties": {
      "userId": "A3SGXH7AUHU8GW",
      "profileName": "67878790",
      "helpfulness": "-PIOPIK",
      "score": 8908977,
      "time": "8:30",
      "summary": "fajhjahjh2q8fuhaj",
      "text": None
    },
    "relationships": [{
    }],
    "label": "Invalid"
  },
  ## Review Node -- Valid Cases
  {
    "id": "B001E4KFG0_D2SGXH7AUHU8GW_1303862400",
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
    "id": "B001E4KFG0_E2SGXH7AUHU8GW_1303862400",
    "type": "Review",
    "properties": {
      "userId": "A3SGXH7AUHU8GW",
      "profileName": "Claire",
      "helpfulness": "1/1",
      "score": 2.0,
      "time": "1294272000",
      "summary": "Bad Experience",
      "text": "The food was cold and we wating for so long"
    },
    "relationships": [],
    "label": "Valid"
  },
  {
    "id": "B001E4KFG0_F2SGXH7AUHU8GW_1303862400",
    "type": "Review",
    "properties": {
      "userId": "A3SGXH7AUHU8GW",
      "profileName": "Bryan",
      "helpfulness": "1/1",
      "score": 3.0,
      "time": "1294272000",
      "summary": "Average experience",
      "text": "try at your own risk."
    },
    "relationships": [],
    "label": "Valid"
  },
  ## Product Node -- Valid Cases
  {
    'id': 'B001E4KFG1',
    'type': 'Product',
    'properties': {
        'type': 'Product',
        'name': 'Mobile',
        'price': 230000
    },
    'relationships': [{
        'type': 'HAS_REVIEW',
        'target_id': 'B001E4KFG0_D2SGXH7AUHU8GW_1303862400'
    }],
    'label': 'Valid'
  },
  {
    'id': 'C001E4KFG1',
    'type': 'Product',
    'properties': {
        'type': 'Product',
        'name': 'Fridge',
        'price': 30000
    },
    'relationships': [{
        'type': 'HAS_REVIEW',
        'target_id': 'B001E4KFG0_E2SGXH7AUHU8GW_1303862400'
    }],
    'label': 'Valid'
  },
  {
    'id': 'A001E4KFG1',
    'type': 'Product',
    'properties': {
        'type': 'Product',
        'name': 'Catapult',
        'price': 3000
    },
    'relationships': [{
    }],
    'label': 'Valid'
  },
  ## Product Node -- Invalid Cases
  {
    'id': 'D001E4KFGX',
    'type': 'Product',
    'properties': {
        'type': 'Product',
        'name': 'Fridge',
        'price': None
    },
    'relationships': [{
        'type': 'HAS_PRODUCT',
        'target_id': 'D001E4KFGX'
    }],
    'label': 'Invalid',
    'dq_issue': 'Missing'
  },
    {
    'id': 'E001E4KFGX',
    'type': 'Product',
    'properties': {
        'type': 'Product',
        'name': '89898F99',
        'price': None
    },
    'relationships': [{
        'type': 'HAS_SOMETHING',
        'target_id': 'B001E4KFGX'
    }],
    'label': 'Invalid',
    'dq_issue': 'Missing'
  },
  {
    'id': 'F001E4KFGX',
    'type': 'Product',
    'properties': {
        'type': 'Product',
        'name': None,
        'price': None
    },
    'relationships': [{
    }],
    'label': 'Invalid',
    'dq_issue': 'Missing'
  },
]