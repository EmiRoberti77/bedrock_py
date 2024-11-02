import boto3
import json
#from similarity import cosineSimilarity
from scipy.spatial.distance import cosine
from jellyfish import jaro_winkler_similarity as get_similarity

def cosineSimilarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)  # 1 - cosine distance gives cosine similarity


client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1") 

facts = [
    'The first computer was invented in the 1940s.',
    'John F. Kennedy was the 35th President of the United States.',
    'The first moon landing was in 1969.',
    'The capital of France is Paris.',
    'Earth is the third planet from the sun.',
]

newFact = 'I like to play computer games'
question = 'Who is the president of USA?'

def getEmbedding(input: str):
    response = client.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": input})
    )

    response_body = json.loads(response["body"].read().decode('utf-8'))
    return response_body["embedding"]

factsWithEmbedding = []

for fact in facts:
    factsWithEmbedding.append({
        "text": fact,
        "embedding": getEmbedding(fact)
    })

newFactEmbedding = getEmbedding(newFact)

similarities = []

for fact in factsWithEmbedding:
    similarities.append({
        "text": fact['text'],
        "similarity": cosineSimilarity(fact['embedding'], newFactEmbedding)  # Corrected syntax here
    })

print(f"Similarities for fact: '{question}' with:")
similarities.sort(key=lambda x: x['similarity'], reverse=True)
for similarity in similarities:
    print(f"  '{similarity['text']}': {similarity['similarity']:.2f}")