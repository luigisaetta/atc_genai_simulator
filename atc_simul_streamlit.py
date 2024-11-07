"""
Simulator3
"""

import time
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st

from prompt import PROMPT
from utils import load_samples, get_console_logger
from config import ENDPOINT, MODEL_NAME, MAX_TOKENS, TEMPERATURE
from config_private import COMPARTMENT_OCID


# the file with the samples
FILE_PATH = "samples.jsonl"

N_MESSAGES = 100

DEFAULT_SCENARIO = (
    "The pilot wants to land to the FCO airport. It is windy and it is raining"
)

# the logger
logger = get_console_logger()


def init_conversation():
    """
    To init the conversation using samples from file
    """
    if "scenario_text" not in st.session_state:
        st.session_state.scenario_text = DEFAULT_SCENARIO

    # the prompt
    data = load_samples(FILE_PATH)

    sample_messages = [SystemMessage(PROMPT + st.session_state.scenario_text)]

    # load the samples from file
    for entry in data[:N_MESSAGES]:
        sample_messages.append(AIMessage(content=entry["AIMessage"]))
        sample_messages.append(HumanMessage(content=entry["HumanMessage"]))

    st.session_state.sample_messages = sample_messages


# when push the button to reset chat
def reset_conversation():
    """
    reset the chat history
    """
    if "scenario_text" not in st.session_state:
        st.session_state.scenario_text = DEFAULT_SCENARIO

    # the prompt
    data = load_samples(FILE_PATH)

    sample_messages = [SystemMessage(PROMPT + st.session_state.scenario_text)]

    # load the samples from file
    for entry in data[:N_MESSAGES]:
        sample_messages.append(AIMessage(content=entry["AIMessage"]))
        sample_messages.append(HumanMessage(content=entry["HumanMessage"]))

    st.session_state.sample_messages = sample_messages

    # scenario
    if "scenario_text" not in st.session_state:
        st.session_state.scenario_text = DEFAULT_SCENARIO

    # the prompt
    data = load_samples(FILE_PATH)

    sample_messages = [SystemMessage(PROMPT + st.session_state.scenario_text)]

    # load the samples from file
    for entry in data[:N_MESSAGES]:
        sample_messages.append(AIMessage(content=entry["AIMessage"]))
        sample_messages.append(HumanMessage(content=entry["HumanMessage"]))

    st.session_state.sample_messages = sample_messages
    # here we put the msg in the simulated conversation
    # reset message history
    st.session_state.messages = []


def stream_output(v_ai_msg):
    """
    format the output when using streaming
    """
    text_placeholder = st.empty()
    formatted_output = ""

    for chunk in v_ai_msg:
        formatted_output += chunk.content
        text_placeholder.markdown(formatted_output, unsafe_allow_html=True)

    text_placeholder.markdown(formatted_output, unsafe_allow_html=True)

    return formatted_output


#
# Main
#
st.title("ATC 🛫 Simulator")

init_conversation()
st.sidebar.button("Clear Chat History", on_click=reset_conversation)

st.sidebar.title("Scenario")
new_scenario_text = st.sidebar.text_area(
    "Describe the scenario", st.session_state.scenario_text, height=200
)  # Altezza personalizzabile

if st.sidebar.button("Update Scenario"):
    # Aggiorna lo stato solo quando il bottone viene premuto
    st.session_state.scenario_text = new_scenario_text
    st.sidebar.success("Scenario updated!")

#
# instantiate the chat model
#
chat = ChatOCIGenAI(
    model_id=MODEL_NAME,
    service_endpoint=ENDPOINT,
    compartment_id=COMPARTMENT_OCID,
    model_kwargs={"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
)


# Initialize session, chat history
if "messages" not in st.session_state:
    reset_conversation()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    ROLE = "human" if isinstance(message, HumanMessage) else "ai"
    AVATAR = "🛫" if isinstance(message, AIMessage) else None

    # need to add avatar here
    with st.chat_message(ROLE, avatar=AVATAR):
        st.markdown(message.content)

# React to user input
if question := st.chat_input("Hello..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(question)

    try:
        time_start = time.time()

        # add current message
        st.session_state.messages.append(HumanMessage(content=question))

        # we call here the Chat model
        # few shot examples: every time we send in the prompt the samples
        ai_msg = chat.stream(
            st.session_state.sample_messages + st.session_state.messages
        )
        # Display assistant response in chat message container
        # the assistant should be the pilot
        with st.chat_message("assistant", avatar="🛫"):
            output = stream_output(ai_msg)

        st.session_state.messages.append(AIMessage(content=output))

        time_elapsed = time.time() - time_start

        logger.info("")
        logger.info("Elapsed time: %3.1f sec.", time_elapsed)
        logger.info("")

    except Exception as e:
        logger.error("Exception in handling response from chat..")
        logger.error(e)
