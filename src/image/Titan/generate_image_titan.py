import boto3
import json
import base64

client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

titan_image_config=json.dumps({
    "taskType":"TEXT_IMAGE",
    "textToImageParams":{
        "text":"a tiger in the forest by a river",
    },
    "imageGenerationConfig":{
        "numberOfImages":1,
        "height":512,
        "width":512,
        "cfgScale":8.0
    }
})

response = client.invoke_model(
    body=titan_image_config,
    modelId="amazon.titan-image-generator-v1",
    accept="application/json",
    contentType="application/json")

response_body = json.loads(response.get("body").read())
base64_image = response_body.get("images")[0]
decodedd_base64Image = base64.b64decode(base64_image)

with open('tiger.png', "wb") as f:
    f.write(decodedd_base64Image)

print('image complete')