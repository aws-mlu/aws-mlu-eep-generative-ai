"""Small helpers for using the Strands Agents SDK in the Module 3 labs."""
import json
import re

from strands import Agent


def ask_llm(model, prompt, system_prompt=None):
    """Send a single prompt to the model and return the response text.

    A fresh Agent is created for every call so that independent prompts
    do not share conversation history.
    """
    agent = Agent(model=model, system_prompt=system_prompt, callback_handler=None)
    return str(agent(prompt)).strip()


def parse_json_response(text):
    """Extract and parse the JSON object or array contained in an LLM response.

    Handles responses wrapped in Markdown code fences or surrounded by extra text.
    Raises ValueError if no valid JSON can be found.
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    starts = [i for i in (candidate.find("{"), candidate.find("[")) if i != -1]
    if not starts:
        raise ValueError("No JSON object found in the model response.")
    start = min(starts)
    end = max(candidate.rfind("}"), candidate.rfind("]"))
    try:
        return json.loads(candidate[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"The model response is not valid JSON: {e}") from e
