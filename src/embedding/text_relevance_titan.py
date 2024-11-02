import boto3
import json
from scipy.spatial.distance import cosine
client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1") 

def cosineSimilarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)  

responses = [
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

responsesWithEmbdedding = []

for response in responses:
    responsesWithEmbdedding.append({
        "response": response,
        "embedding": getEmbedding(response)
    })

newFactEmbedding = getEmbedding(question)

similarities = []

for fact in responsesWithEmbdedding:
    similarities.append({
        "response": fact['response'],
        "similarity": cosineSimilarity(fact['embedding'], newFactEmbedding)  # Corrected syntax here
    })

print(f"best response to '{question}'")
similarities.sort(key=lambda x: x['similarity'], reverse=True)
for similarity in similarities:
    print(f"  '{similarity['response']}': {similarity['similarity']:.2f}")