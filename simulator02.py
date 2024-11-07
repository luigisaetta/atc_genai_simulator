"""
Sample code for ATC Simulator, based on OCI GenAI

Use the samples.jsonl to build the prompt
"""

from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from prompt import PROMPT
from utils import load_samples
from config import MODEL_NAME, ENDPOINT
from config_private import COMPARTMENT_OCID

# configs
file_path = "samples.jsonl"

# instantiate the chat model
chat = ChatOCIGenAI(
    model_id=MODEL_NAME,
    service_endpoint=ENDPOINT,
    compartment_id=COMPARTMENT_OCID,
    model_kwargs={"temperature": 0.7, "max_tokens": 600},
)

# read the jsonl file
# prompt are messages from pilot, completion from atc


# List to memorize all data
data = load_samples(file_path)

# we have more than 1000 samples in the file
# but with too many samples it ignores the scenario
N_MESSAGES = 100

SCENARIO = "The pilot wants to land to the FCO airport. It is windy and it is raining"

messages = []

# the prompt
messages.append(SystemMessage(PROMPT + SCENARIO))

for entry in data[:N_MESSAGES]:
    ai_msg = AIMessage(content=entry["AIMessage"])
    human_msg = HumanMessage(content=entry["HumanMessage"])
    messages.append(ai_msg)
    messages.append(human_msg)

print("")

for r in chat.stream(messages):
    print(r.content, end="", flush=True)

print("")
print("")
