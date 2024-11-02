from langchain_aws import BedrockLLM as Bedrock
from langchain_core.prompts import ChatPromptTemplate
import boto3

AWS_REGION = "us-east-1"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
model = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock)


def invoke_model():
    response = model.invoke("whats is the fastest man made vehicle?")
    print(response)


def fist_chain():
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Write a short description for the product provided by the user",
            ),
            ("human", "{product_name}"),
        ]
    )
    chain = template.pipe(model)

    response = chain.invoke({"product_name": "car"})
    print(response)


#invoke_model()
fist_chain()

