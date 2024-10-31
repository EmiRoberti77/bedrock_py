import boto3
import json
import pprint

pp = pprint.PrettyPrinter(indent=4)
client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
titan_model_id = "amazon.titan-text-express-v1"
titan_config = json.dumps({
    "inputText":"tell me a story about a dragon",
    "textGenerationConfig":{
        "maxTokenCount":8192,
        "stopSequences":[],
        "temperature":0,
        "topP":1
    }
})

response = client.invoke_model(
    body=titan_config,
    modelId=titan_model_id,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get('body').read())
pp.pprint(response_body.get('results'))