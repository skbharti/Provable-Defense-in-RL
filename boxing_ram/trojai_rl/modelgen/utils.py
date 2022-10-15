import json


def is_jsonable(arg):
    try:
        json.dumps(arg)
        return True
    except (TypeError, OverflowError):
        return False
