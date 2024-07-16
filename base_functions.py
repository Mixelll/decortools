import inspect
import concurrent.futures


def to_list(x, ignore_none=True, none_empty_list=False):
    if ignore_none:
        return [x] if x is not None and not isinstance(x, (list, tuple)) else x
    elif none_empty_list:
        return [] if x is None else [x] if not isinstance(x, (list, tuple)) else x
    else:
        return [x] if not isinstance(x, (list, tuple)) else x


def get_required_param_names_from_func(sig_or_func=None):
    if not isinstance(sig_or_func, inspect.Signature):
        sig_or_func = inspect.signature(sig_or_func)
    req_names = []
    for param in sig_or_func.parameters.values():
        if param.default is inspect.Parameter.empty and param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY):
            req_names.append(param.name)

    return req_names


def func_origin_str(func):
    return func.__module__ + '.' + func.__qualname__


def contains_dict(item):
    """
    Recursively check if there is a dictionary inside a possibly nested list or if the item itself is a dictionary.

    Args:
        item (any): The item to check if it's a dictionary or if it contains a dictionary in nested lists.

    Returns:
        bool: True if a dictionary is found, False otherwise.
    """
    if isinstance(item, dict):
        return True  # The item itself is a dictionary
    elif isinstance(item, list):
        for subitem in item:
            if contains_dict(subitem):  # Recursively check the elements
                return True
    return False  # No dictionary found

