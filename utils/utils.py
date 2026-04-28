import re
import json

def parse_string(val):
    if isinstance(val, str):
        return json.loads(val)
    return val

def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return text