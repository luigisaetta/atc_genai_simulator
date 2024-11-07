"""
Utils
"""

import json
import logging


def load_samples(file_name):
    """
    Read the samples for the prompt from the json file
    """
    data = []

    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            # load the row as json
            entry = json.loads(line.strip())

            # rename field
            if "prompt" in entry:
                # prompt is the pilot -> AI
                entry["AIMessage"] = entry.pop("prompt")
            if "completion" in entry:
                # completion is ATC -> human
                entry["HumanMessage"] = entry.pop("completion")

            # Aggiungi l'elemento aggiornato alla lista
            data.append(entry)

    return data


def get_console_logger():
    """
    To get a logger to print on console
    """
    logger = logging.getLogger("ConsoleLogger")

    # to avoid duplication of logging
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False

    return logger
