# Function to load and parse the Amazon Fine Foods Reviews data from a text file
def load_finefoods_data(filepath, encoding='ISO-8859-1'):
    products = []
    reviews = []
    product = {}
    with open(filepath, 'r', encoding=encoding) as file:
        for line in file:
            line = line.strip()
            if line.startswith("product/productId:"):
                if product:
                    products.append(product)
                product_id = line.split(": ")[1]
                product = {
                    "id": product_id,
                    "type": "Product",
                    "properties": {"type":"Product", "name":random.choice(["Book", "Mobile", "TV", "Fridge"]), "price": random.randrange(0, 250000)},
                    "relationships": []
                }
            elif line.startswith("review/userId:"):
                review = {"userId": line.split(": ")[1]}
            elif line.startswith("review/profileName:"):
                review["profileName"] = line.split(": ")[1]
            elif line.startswith("review/helpfulness:"):
                review["helpfulness"] = line.split(": ")[1]
            elif line.startswith("review/score:"):
                review["score"] = float(line.split(": ")[1])
            elif line.startswith("review/time:"):
                review["time"] = line.split(": ")[1]
            elif line.startswith("review/summary:"):
                review["summary"] = line.split(": ")[1]
            elif line.startswith("review/text:"):
                review["text"] = line.split(": ")[1]
                review_id = f"{product_id}_{review['userId']}_{review['time']}"
                review["type"] = "Review"
                review_node = {
                    "id": review_id,
                    "type": "Review",
                    "properties": review
                }
                reviews.append(review_node)
                product["relationships"].append({"type": "HAS_REVIEW", "target_id": review_id})
        if product:
            products.append(product)
    return products, reviews

# Replace 'finefoods.txt' with the actual path to your dataset file
products_data, reviews_data = load_finefoods_data(amazon_fine_foods)

# Function to parse and format product data
def parse_finefoods_product(product):
    node = {
        "id": product.get('id', ''),
        "type": "Product",
        "properties": product.get("properties", {}),
        "relationships": product.get("relationships", [])
    }
    return node

# Function to parse and format review data
def parse_finefoods_review(review):
    node = {
        "id": review.get('id', ''),
        "type": "Review",
        "properties": review.get("properties", {}),
        "relationships": review.get("relationships", [])
    }
    return node

# Extract and format data
product_nodes = [parse_finefoods_product(product) for product in products_data]
review_nodes = [parse_finefoods_review(review) for review in reviews_data]

# Combine all nodes
all_nodes = product_nodes + review_nodes

# Example: Convert one of the nodes to the desired format
example_product_node = product_nodes[0]
example_review_node = review_nodes[0]