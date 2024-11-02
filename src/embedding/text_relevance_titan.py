import boto3
import json
#from similarity import cosineSimilarity
from scipy.spatial.distance import cosine
from jellyfish import jaro_winkler_similarity as get_similarity

def cosineSimilarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)  # 1 - cosine distance gives cosine similarity


client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1") 

facts = [
    'there is a tennis court in basingstoke',
    'ferrari has won the Formula 1 world title',
    'I had a dog called leroy',
    'capital of Italy is Rome',
    'i enjoy going to the gym',
]

question = 'what is my favorite pass time?'

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

newFactEmbedding = getEmbedding(question)

similarities = []

for fact in factsWithEmbedding:
    similarities.append({
        "text": fact['text'],
        "similarity": cosineSimilarity(fact['embedding'], newFactEmbedding)  # Corrected syntax here
    })

print(f"best response to '{question}'")
similarities.sort(key=lambda x: x['similarity'], reverse=True)
for similarity in similarities:
    print(f"  '{similarity['text']}': {similarity['similarity']:.2f}")