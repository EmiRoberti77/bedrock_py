import boto3
import json
import pprint

pp = pprint.PrettyPrinter(indent=4)
client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
llama_model_id = "meta.llama3-8b-instruct-v1:0"
llama_config = json.dumps({
   "prompt":"give me a dragon story",
    "max_gen_len":512,
    "temperature":0.5,
    "top_p":0.9
    })

response = client.invoke_model(
    body=llama_config,
    modelId=llama_model_id,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get('body').read())
pp.pprint(response_body['generation'])