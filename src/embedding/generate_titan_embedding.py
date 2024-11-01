import boto3
import json

client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
sentence = 'Ferrari iconic color is red'

response = client.invoke_model(
    modelId="amazon.titan-embed-text-v1",
    contentType="application/json",
    accept="*/*",
    body=json.dumps({
        "inputText": sentence
    })
)

response_body = json.load(response["body"])
print(response_body["embedding"])