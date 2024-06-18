This module implements the following utilities:
- Inject_inputs - Decorator for parameter injection.
- ExecuteFunctionOnCall - Class of input functions wrapper with an execution decorator, wrapper functions are injected with Inject_inputs.
- dynamic_date_range_decorator - Decorator to apply a function over a dynamic date range specified by start and end dates returned by the decorated function.
     thorough conversion from str formats and back to datetime with splitting by frequency or a divisor
- date_split_decorator - Decorator to apply a function over multiple time intervals within a specified date range.
- df_manipulator_decorator - Decorator that applies a transformation function to specified DataFrame columns either before or after
     the decorated function executes. If the transform function accepts a series instead, an list of column names or types should be provided.
- post_decorator_factory - A factory function for a decorator that modifies and forwards the outputs of a function to another function based on a specified mapping.
- generate_decorator - Post decorator factory followed by inject_inputs and ExecuteFunctionOnCall decorators.

