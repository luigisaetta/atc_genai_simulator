"""
Sample code for ATC Simulator, based on OCI GenAI
"""

from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.messages import HumanMessage

MODEL_NAME = "meta.llama-3.1-70b-instruct"
ENDPOINT = "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"
COMPARTMENT_OCID = "ocid1.compartment.oc1..aaaaaaaaushuwb2evpuf7rcpl4r7ugmqoe7ekmaiik3ra3m7gec3d234eknq"

chat = ChatOCIGenAI(
    model_id=MODEL_NAME,
    service_endpoint=ENDPOINT,
    compartment_id=COMPARTMENT_OCID,
    model_kwargs={"temperature": 0.1, "max_tokens": 600},
)

messages = [HumanMessage(content="Tell me a joke.")]

for r in chat.stream(messages):
    print(r.content, end="", flush=True)
