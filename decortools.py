import functools as ft
import inspect
import logging
import traceback

import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser

import base_functions as bf
import datetime_functions as dtf


# Setup logging configuration
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def copy_signature(source_fct):

    def _copy(target_fct):
        target_fct.__signature__ = inspect.signature(source_fct)
        return target_fct
    return _copy


def concurrent_groupby_apply(df, apply_func, groupby, max_workers=4):
    """
    Splits a DataFrame into groups based on a specified column and applies a function concurrently to each group.
    Args:
        df (pandas.DataFrame): The DataFrame to process.
        apply_func (callable): Function to apply to each group.
        groupby (str): Column name to group by.
        max_workers (int): The maximum number of concurrent workers.
    Returns:
        pandas.DataFrame: DataFrame with the results concatenated from each group after processing.
    """
    grouped = df.groupby(groupby)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(apply_func, [group for _, group in grouped]))
    return pd.concat(results)


# def concurrent_df_processing(column_name, max_workers=5):
#     """
#     Decorator to split a DataFrame based on a column and process each split concurrently.
#
#     Args:
#         column_name (str): The name of the column to split the DataFrame on.
#         max_workers (int): The maximum number of concurrent workers.
#
#     Returns:
#         function: A decorated function that processes parts of DataFrame concurrently.
#     """
#
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(df, *args, **kwargs):
#             # Split the DataFrame based on the specified column
#             grouped = df.groupby(column_name)
#             results = []
#             with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#                 # Submit each group to the decorated function
#                 future_to_group = {
#                     executor.submit(func, group, *args, **kwargs): name
#                     for name, group in grouped
#                 }
#                 for future in concurrent.futures.as_completed(future_to_group):
#                     results.append(future.result())
#             # Optionally, you could concatenate results back into a single DataFrame
#             # return pd.concat(results)
#             return results
#
#         return wrapper
#
#     return decorator

def inject_signature(source_func):
    """
    Decorator that replaces the *args and **kwargs parameters of a target function with explicit parameters
    based on the signature of a source function.
    """
    def replace(target_func):
        source_signature = inspect.signature(source_func)
        target_signature = inspect.signature(target_func)

        new_params = []
        for param in target_signature.parameters.values():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                # Add positional parameters from the source function
                new_params.extend(
                    p for p in source_signature.parameters.values()
                    if p.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]
                )
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                # Add keyword parameters from the source function
                new_params.extend(
                    p for p in source_signature.parameters.values()
                    if p.kind == inspect.Parameter.KEYWORD_ONLY
                )
            else:
                new_params.append(param)

        # Update the target function's signature directly
        target_func.__signature__ = target_signature.replace(parameters=new_params)
        ft.update_wrapper(target_func, source_func, updated=())  # Update wrapper without overwriting the signature again
    return replace


def bind_signature(source_func):
    """
    Decorator that sets the signature of the decorated function to match the signature of the source function.
    """
    source_signature = inspect.signature(source_func)

    def decorator(target_func):
        # Define the wrapper function to replace *args and **kwargs with explicit arguments
        @ft.wraps(target_func)
        def wrapper(*args, **kwargs):
            bound_args = source_signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            # Pass all arguments as keyword arguments
            return target_func(**bound_args.arguments)

        # Set the modified signature to the wrapper
        wrapper.__signature__ = source_signature
        return wrapper

    return decorator


def object_manipulator_decorator(target_class, manipulate_func, after=False):
    """
    Decorator that applies a transformation function to instances of a specific class found either in the
    function's inputs or outputs.

    Args:
        target_class (type): The class to target for applying the manipulate_func.
        manipulate_func (callable): Function to apply to instances of target_class.
        after (bool): If True, applies the function after the decorated function executes; otherwise, before.
    """

    def decorator(func):
        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            # Function to apply transformations to instances of target_class
            def apply_transformations(args, kwargs):
                new_args = []
                for arg in args:
                    if isinstance(arg, target_class):
                        new_args.append(manipulate_func(arg))
                    else:
                        new_args.append(arg)

                new_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, target_class):
                        new_kwargs[key] = manipulate_func(value)
                    else:
                        new_kwargs[key] = value

                return tuple(new_args), new_kwargs

            if not after:
                args, kwargs = apply_transformations(args, kwargs)

            # Call the original function
            result = func(*args, **kwargs)

            if after:
                if isinstance(result, target_class):
                    result = manipulate_func(result)
                elif isinstance(result, (list, tuple)):
                    result = type(result)(manipulate_func(item) if isinstance(item, target_class) else item for item in result)

            return result

        return wrapper

    return decorator


def df_manipulator_decorator(manipulate_func, *args_, apply_func_to_series=None, after=True, pass_function=False, **kwargs_):
    """
    Decorator that applies a transformation function to specified DataFrame columns either before or after
    the decorated function executes, based on the 'after' parameter.

    Args:
        manipulate_func (callable): Function to apply to DataFrame column(s).
        apply_func_to_series (list, type, str, or None): A single or a list of column names or data types
                                                        specifying which columns to iterate over.
                                                        If None, applies to all columns.
        after (bool): If True, applies the function after the decorated function executes and replaces
                      the returned DataFrame if applicable; if False, applies it before the function.
        pass_function (bool): If True, passes the decorated function as an argument to manipulate_func.
    """

    def decorator(func):
        @ft.wraps(func)
        # @copy_signature(func)
        def wrapper(*args, **kwargs):

            # Normalize apply_func_to_series to a list if it is a single type or column name
            if apply_func_to_series is not None and not isinstance(apply_func_to_series, (list, tuple)):
                apply_func_to_series_list = [apply_func_to_series]
            else:
                apply_func_to_series_list = apply_func_to_series

            # Function to apply transformations to the DataFrame
            def apply_transformations(df):
                if apply_func_to_series_list is None:
                    if pass_function:
                        df_ = manipulate_func(df, func, *args_, **kwargs_)
                    else:
                        df_ = manipulate_func(df, *args_, **kwargs_)
                    # if df_ is not None:
                    #     df = df_
                else:
                    # Apply function to specified columns by names or types
                    for column in df.columns:
                        column_type = df[column].dtype
                        if column in apply_func_to_series_list or column_type in apply_func_to_series_list:
                            if pass_function:
                                df[column] = manipulate_func(df[column], func, *args_, **kwargs_)
                            else:
                                df[column] = manipulate_func(df[column], *args_, **kwargs_)

            if not after:
                # Analyze the function's parameters to find the DataFrame
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                df_arg_name = None
                for name, value in bound_args.arguments.items():
                    if isinstance(value, pd.DataFrame):
                        df_arg_name = name
                        break

                if not df_arg_name:
                    raise ValueError("No DataFrame argument found in the function.")

                df_to_manipulate = bound_args.arguments[df_arg_name]
                apply_transformations(df_to_manipulate)
                result = func(*bound_args.args, **bound_args.kwargs)

            else:
                result = func(*args, **kwargs)
                apply_transformations(result)

            return result

        return wrapper

    return decorator


def find_variadic_parameters(func):
    """
    Inspects a function and identifies the parameters used for *args and **kwargs.

    Args:
        func (callable): The function to inspect.

    Returns:
        tuple: Contains two elements, where the first element is the name of the *args parameter (or None if not present),
               and the second element is the name of the **kwargs parameter (or None if not present).
    """
    signature = inspect.signature(func)
    var_positional = None
    var_keyword = None

    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            var_positional = name  # Name associated with *args
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            var_keyword = name  # Name associated with **kwargs

    return var_positional, var_keyword


def analyze_function_params(sig_or_func=None):
    """
    Analyze the parameters of a function to determine if they can be passed positionally and their indices or names.

    Args:
        sig_or_func (callable): The function whose parameters to analyze, or its signature.

    Returns:
        dict: Maps each parameter name to a tuple (should_be_positional, index_or_name).
    """
    if not isinstance(sig_or_func, inspect.Signature):
        sig_or_func = inspect.signature(sig_or_func)
    param_details = {}
    positional_index = 0
    for name, param in sig_or_func.parameters.items():
        should_be_positional = param.kind in [inspect.Parameter.POSITIONAL_ONLY]
        if should_be_positional:
            param_details[name] = (True, positional_index)
            positional_index += 1
        else:
            param_details[name] = (False, name)
    return param_details


def fetch_param_values(param_names, input_args, input_kwargs, return_inserter=False, return_dict=False, signature_=None, func=None):
    """
    Fetch parameter values from input_args and input_kwargs for a given list of parameter names based on a function's signature.

    Args:
        param_names (list of str): Names of the parameters to fetch.
        input_args (tuple): Positional arguments with which the function might be called.
        input_kwargs (dict): Keyword arguments with which the function might be called.
        return_inserter (bool): If True, returns a function to reinsert parameters into args and kwargs.
        return_dict (bool): If True, returns a dictionary of parameter values instead of a tuple.
        signature_ (callable): The signature of the function whose parameters to analyze.
        func (callable): The function to analyze.

    Returns:
        dict or (dict, callable): A dictionary mapping each parameter name to its fetched value, and optionally a function to reinsert values.
    """
    param_indices = analyze_function_params(inspect.signature(func) if signature_ is None else signature_)
    values_dict = {}
    positions = {}  # Store positions for possible reinsertion
    # Determine the index for fallback insertion
    for name in param_names:
        if isinstance(name, int):
            if name < len(input_args):
                values_dict[name] = input_args[name]
                positions[name] = ('args', name)
            else:
                values_dict[name] = list(input_kwargs.values())[name - len(input_args)]
        elif name in param_indices:
            should_be_positional, index_or_name = param_indices[name]
            if should_be_positional and len(input_args) > index_or_name:
                values_dict[name] = input_args[index_or_name]
                positions[name] = ('args', index_or_name)
            elif name in input_kwargs:
                values_dict[name] = input_kwargs[name]
                positions[name] = ('kwargs', name)
            else:
                raise ValueError(f"Parameter '{name}' not found in input_args or input_kwargs.")

    def reinsert_parameters(updated_values):
        """
        Updates input_args and input_kwargs with modified values based on their original positions.

        Args:
            updated_values (dict): Dictionary of updated parameter values.

        Returns:
            tuple: Updated (args, kwargs)
        """
        new_args = list(input_args)
        new_kwargs = dict(input_kwargs)

        for key, value in updated_values.items():
            if key in positions:
                arg_type, pos = positions[key]
                if arg_type == 'args':
                    if pos < len(new_args):
                        new_args[pos] = value
                    else:
                        # Expand args list if out of range (rare case)
                        new_args.extend([None] * (pos - len(new_args)))
                        new_args.append(value)
                elif arg_type == 'kwargs':
                    new_kwargs[pos] = value

        return tuple(new_args), new_kwargs

    if return_inserter:
        return values_dict if return_dict else tuple(values_dict.values()), reinsert_parameters
    else:
        return values_dict


def dynamic_date_range_decorator(start_name='start_date', end_name='end_date', result_date_accessor_fn=None, aggregate_fn=None):
    """
    Decorator to apply a function over a dynamic date range specified by the input start and end dates corrected bv the
     start and end dates returned by each subsequent call (inside the wrapper) to the decorated function.

    Args:
        start_name (str): The name or position of the start date parameter.
        end_name (str): The name or position of the end date parameter.
        result_date_accessor_fn (callable): A function to extract the date from the result of the decorated function.
        aggregate_fn (callable): A function to aggregate the results of the decorated function.

    Returns:
        function: A decorated function that operates over specified intervals.
    """

    def decorator(func):
        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            (start_date, end_date), reinsert_parameters_fn = fetch_param_values([start_name, end_name], args, kwargs, func=func, return_inserter=True)
            start_date_cls = start_date.__class__
            start_date, date_format, tz = dtf.str2datetime(start_date, return_format_and_tz=True)
            if end_date is None:
                end_date = dtf.now_as_tz(start_date)
            else:
                end_date = dtf.str2datetime(end_date, date_format=date_format, tz=tz)
            cast_fn = lambda x: x if start_date_cls is datetime else start_date_cls(x.strftime(date_format) if date_format else x)
            results = []
            date_range = [start_date, end_date]
            while date_range[0] < date_range[1] and date_range[1] - date_range[0] > timedelta(days=1):
                updated_args, updated_kwargs = reinsert_parameters_fn({start_name: cast_fn(date_range[0]), end_name: cast_fn(date_range[1])})
                results.append(func(*updated_args, **updated_kwargs))
                returned_dates = result_date_accessor_fn(results[-1])
                if returned_dates.empty:
                    logging.warning(f"No dates returned for the interval {date_range[0]} to {date_range[1]}.")
                    break
                naive_fn = lambda x: x.replace(tzinfo=None) if tz is None else x
                if naive_fn(min(returned_dates)) - date_range[0] < date_range[1] - naive_fn(max(returned_dates)):
                    date_range = [naive_fn(max(returned_dates)), date_range[1]]
                else:
                    date_range = [date_range[0], naive_fn(min(returned_dates))]
            if aggregate_fn:
                return aggregate_fn(results)
            return results

        return wrapper

    return decorator


def date_split_decorator(frequency='1M', divisor=1, start_name='start_date', end_name='end_date', aggregate_fn=lambda x: pd.concat(x)):
    """
    Decorator to apply a function over multiple time intervals within a specified date range.

    Args:
        frequency (str): A pandas frequency string indicating the splitting intervals.
        divisor (int): Number of divisions of the total period if intervals are not explicitly provided.
        start_name (str): The name or position of the start date parameter.
        end_name (str): The name or position of the end date parameter.
        aggregate_fn (callable): A function to aggregate the results of the decorated function.

    Returns:
        function: A decorated function that operates over specified intervals.
    """

    def decorator(func):
        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            (start_date, end_date), reinsert_parameters_fn = fetch_param_values([start_name, end_name], args, kwargs, func=func, return_inserter=True)
            intervals = dtf.create_intervals_from_timestamps(dtf.calculate_date_ranges(start_date, end_date, frequency, divisor))
            results = []
            for start, end in intervals:
                updated_args, updated_kwargs = reinsert_parameters_fn({start_name: start, end_name: end})
                results.append(func(*updated_args, **updated_kwargs))
            if aggregate_fn:
                return aggregate_fn(results)
            return results

        return wrapper

    return decorator


def generate_decorator(funcs, inject=None, execute_on_call=False, output_to_input_map=None, overwrite_output=None):
    """
    Post decorator factory followed by inject_inputs and ExecuteFunctionOnCall decorators.
    """
    if not isinstance(funcs, (list, tuple)):
        funcs = [funcs]
    if inject is None:
        inject = {}
    if execute_on_call:
        inject = {k: ExecuteFunctionOnCall(v) if callable(v) and not bf.get_required_param_names_from_func(v) else v for k, v in inject.items()}

    @inject_inputs(**inject)
    @ExecuteFunctionOnCall.decorator
    @post_decorator_factory(output_to_input_map=output_to_input_map, overwrite_output=bool(overwrite_output))
    @copy_signature(funcs[0])
    def decorator_shell(*args, **kwargs):
        out = [f(*args, **kwargs) for f in funcs]
        match overwrite_output:
            case 'all':
                return out
            case 'first':
                return out[0]
            case _:
                return out[-1]
    return decorator_shell


def inject_execute_on_call(funcs, inject=None, execute_on_call=False, overwrite_output=None):
    """
    inject_inputs and ExecuteFunctionOnCall decorators.
    """
    if not isinstance(funcs, (list, tuple)):
        funcs = [funcs]
    if inject is None:
        inject = {}
    if execute_on_call:
        inject = {k: ExecuteFunctionOnCall(v) if callable(v) and not bf.get_required_param_names_from_func(v) else v for k, v in inject.items()}

    @inject_inputs(**inject)
    @ExecuteFunctionOnCall.decorator
    @copy_signature(funcs[0])
    def decorator_shell(*args, **kwargs):
        out = [f(*args, **kwargs) for f in funcs]
        match overwrite_output:
            case 'all':
                return out
            case 'first':
                return out[0]
            case _:
                return out[-1]
    return decorator_shell


class ExecuteFunctionOnCall:
    """
      Wraps a callable and provides a method to execute it with arbitrary arguments,
    allowing deferred or controlled execution. Also includes a static method decorator
    that enhances any function to automatically execute wrapped callables passed as
    arguments, replacing them with their results before the actual function execution.

    This class is particularly useful for scenarios involving callback functions or
    when functions accept other functions as parameters that should only be executed
    under specific conditions controlled by the main function.
    """
    def __init__(self, func):
        self.func = func

    def execute(self, *args, **kwargs):
        # Execute the stored function with any provided arguments
        return self.func(*args, **kwargs)

    @staticmethod
    def decorator(func):
        sig1 = bf.func_origin_str(func)
        sig2 = bf.func_origin_str(ExecuteFunctionOnCall.decorator)
        debug_str = f"DEBUG: {sig1} @ {sig2}"
        logger.debug(debug_str)

        @ft.wraps(func)
        # @copy_signature(func)
        def wrapper(*args, **kwargs):
            wrapper_debug_str = f"{debug_str} (INSIDE WRAPPER) with args={args}, kwargs={kwargs}"
            logger.debug(wrapper_debug_str)
            new_args = [arg.execute() if isinstance(arg, ExecuteFunctionOnCall) else arg for arg in args]
            new_kwargs = {k: v.execute() if isinstance(v, ExecuteFunctionOnCall) else v for k, v in kwargs.items()}

            try:
                result = func(*new_args, **new_kwargs)
                return result
            except Exception as e:
                logger.error(f"ERROR: {str(e)}. CALLING: {wrapper_debug_str}")
                traceback.print_exc()

        return wrapper


def inject_inputs(*args, args_indices=None, **kwargs):
    """
    Modifies a function's signature by injecting specified positional and keyword arguments,
    and then removes these arguments from the function's formal signature. It raises an error
    if there is a conflict between keyword arguments provided at call time and those specified
    to be injected by the decorator.

    Parameters:
        *args (tuple): Positional arguments to inject into the function's call.
        args_indices (list, optional): Indices at which to insert the positional arguments.
        **kwargs (dict): Keyword arguments to bind directly to the function.

    Raises:
        ValueError: If there is a conflict between keyword arguments provided during the call
                    and those set by the decorator.
        IndexError: If an index in args_indices is out of the allowed range.
    """

    # print('inside def inject_inputs')
    def decorator(func):
        # print('Entering inject_inputs decorator')
        # Retrieve the original function's signature and parameters
        original_sig = inspect.signature(func)
        parameters = list(original_sig.parameters.values())
        # Default to sequential insertion if no indices are provided
        if args_indices is None:
            args_indices_in = range(len(args))
        elif len(args_indices) != len(args):
            raise ValueError("Length of args_indices must match the length of args.")
        else:
            args_indices_in = args_indices

        # Ensure provided indices are within the allowed range of the function's parameters
        if any(index >= len(parameters) for index in args_indices_in):
            raise IndexError(f"One or more indices are out of the allowed range of positional parameters for {func}.")
        required_params = bf.get_required_param_names_from_func(func)

        # @ft.wraps(func)
        @copy_signature(func)
        def wrapper(*wrapper_args, **wrapper_kwargs):
            # print('wrapper start inject_inputs')
            # Insert positional arguments at specified indices
            wrapper_args_list = list(wrapper_args)
            for arg, index in zip(args, args_indices_in):
                if index <= len(wrapper_args_list):
                    wrapper_args_list.insert(index, arg)
                else:
                    wrapper_args_list.append(arg)

            # Handle and validate keyword arguments; check for conflicts
            for key, value in kwargs.items():
                if key in wrapper_kwargs:
                    raise ValueError(f"Conflict detected: '{key}' is provided by both the caller and the decorator.")
                wrapper_kwargs[key] = value
            [wrapper_args_list.insert(idx, wrapper_kwargs.pop(k)) for idx, k in enumerate(required_params) if k in wrapper_kwargs]
            if len(wrapper_args_list) < len(required_params):
                raise ValueError(f"Not enough positional arguments provided to {func}.")
            return func(*wrapper_args_list, **wrapper_kwargs)

        # Update the function's signature by removing parameters that are managed by the decorator
        new_params = [p for i, p in enumerate(parameters) if i not in args_indices_in and p.name not in kwargs]
        wrapper.__signature__ = original_sig.replace(parameters=new_params)
        # print('Exiting inject_inputs decorator')
        return wrapper

    return decorator


class UninitializedArgument:
    pass


def post_decorator_factory(output_to_input_map=None, overwrite_output=False):
    """
    A factory function for a decorator that modifies and forwards the outputs
    of a function to another function based on a specified mapping.

    Args:
        output_to_input_map (dict, optional): Keys are output positions of the decorated function,
                                              and values are either indices or names of parameters in `some_function`.
        overwrite_output (bool, optional): If True, the output of the decorated function will be replaced by the output of `some_function`.
    """
    if output_to_input_map is None:
        output_to_input_map = {0: 0}

    def func2decorator_decorator(func2decorator):
        func_sig = inspect.signature(func2decorator)
        # # accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in func_sig.parameters.values())
        # idx_names = [(idx, param.name) for idx, param in enumerate(func_sig.parameters.values()) if
        #              param.default is inspect.Parameter.empty and
        #              param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)]
        # required_names = [y for _, y in idx_names]
        parameters = list(func_sig.parameters.values())
        required_names = bf.get_required_param_names_from_func(func_sig)
        max_index = len(required_names) - 1

        @ft.wraps(func2decorator)
        def func2decorator_wrapper(*d_args, **d_kwargs):
            def decorator(some_function):
                @ft.wraps(some_function)
                def wrapper(*args, **kwargs):
                    output_from_func = some_function(*args, **kwargs)
                    unpack_flag = False
                    if not isinstance(output_from_func, tuple):
                        output_from_func = (output_from_func,)
                        unpack_flag = True

                    full_args = [UninitializedArgument() for _ in range(max_index + 1)]
                    full_kwargs = dict(d_kwargs)

                    for output_idx, target in output_to_input_map.items():
                        if output_idx >= len(output_from_func):
                            raise IndexError(f"Output index {output_idx} is out of range for the function's output size {len(output_from_func)} of {some_function}.")

                        if isinstance(target, int):
                            if target > max_index:
                                raise IndexError(f"Target position exceeds the calculated maximum index of args and kwargs ({max_index}) in {func2decorator}.")
                            full_args[target] = output_from_func[output_idx]
                        elif isinstance(target, str):
                            full_kwargs[target] = output_from_func[output_idx]

                    not_enough_args_error = ValueError(f"Not enough args to fill the required parameters of {func2decorator}.")
                    additional_args_iter = iter(d_args)
                    for i, value in enumerate(full_args):
                        if isinstance(value, UninitializedArgument):
                            try:
                                full_args[i] = next(additional_args_iter)
                            except StopIteration:
                                if (sum(1 for arg in full_args if not isinstance(arg, UninitializedArgument)) + len(full_kwargs)) >= max_index:
                                    for j, name in enumerate(required_names):
                                        if j > max_index:
                                            break
                                        if isinstance(full_args[j], UninitializedArgument) and name in full_kwargs:
                                            full_args[j] = full_kwargs.pop(name)
                                    if any(isinstance(arg, UninitializedArgument) for arg in full_args):
                                        raise not_enough_args_error
                                else:
                                    raise not_enough_args_error
                    f2d_out = func2decorator(*full_args, **full_kwargs)
                    if overwrite_output:
                        return f2d_out
                    return output_from_func[0] if unpack_flag else output_from_func
                return wrapper
            return decorator
        new_params = [p for i, p in enumerate(parameters) if i not in output_to_input_map.values() and p.name not in output_to_input_map.values()]
        func2decorator_wrapper.__signature__ = func_sig.replace(parameters=new_params)
        return func2decorator_wrapper
    return func2decorator_decorator

