import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import re
from datetime import datetime, timedelta
import pytz
import tzlocal


def now_as_tz(ts):
    tz = get_timezone(ts)
    return to_timezone(datetime.now(), tz='UTC' if tz is None else tz, naive=tz is None, local=True)


def get_timezone(x):
    typeStr = str(type(x)).lower()
    if isinstance(x, str):
        out = x
    elif typeStr.find('pandas') != -1:
        try:
            out = x.dt.tz
        except:
            out = x.tz
    elif typeStr.find('tzfile') != -1:
        out = x.zone
    else:
        try:
            out = x.tzinfo.zone
        except:
            return
    return out


def to_timezone(inp, tz=None, naive=False, local=False):
    type_str = str(type(inp)).lower()
    localize_tz = str(tzlocal.get_localzone()) if local else 'UTC'
    if tz is None:
        tz = localize_tz
    if 'pandas' in type_str:
        try:
            if inp.tz is None:
                out = inp.tz_localize(localize_tz,
                                      nonexistent='shift_backward').tz_convert(str(tz))
            else:
                out = inp.tz_convert(tz)
            if naive:
                out = out.tz_localize(None)
        except AttributeError:
            if inp.dt.tz is None:
                out = inp.dt.tz_localize(localize_tz, ambiguous='infer',
                                         nonexistent='shift_backward').dt.tz_convert(str(tz))
            else:
                out = inp.dt.tz_convert(tz)
            if naive:
                out = out.dt.tz_localize(None)

    else:
        if inp.tzinfo is None:
            out = pytz.timezone(localize_tz).localize(inp)
        else:
            out = inp
        if isinstance(tz, str):
            out = out.astimezone(pytz.timezone(tz))
        else:
            out = out.astimezone(tz)
        if naive:
            out = out.replace(tzinfo=None)
    return out


def deduce_date_format(date_str, return_tz=False):
    """
    Attempt to deduce the date format based on a comprehensive list of common and less common date string formats,
    including those with timezone information. Optionally returns the timezone.

    Args:
        date_str (str): The date string to analyze.
        return_tz (bool): If True, returns a tuple of (format, timezone).

    Returns:
        str or tuple: The date format string, or tuple of format string and timezone string if return_tz is True.
    """
    # List of predefined common and some uncommon formats, including timezone support
    formats = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y',  # Common date formats
        '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S',  # Common datetime formats
        '%Y-%m-%d %H:%M:%S%z', '%m/%d/%Y %H:%M:%S%z',  # Datetime with timezone
        '%Y-%m-%dT%H:%M:%S%z',  # ISO 8601 with timezone
        '%Y-%m-%dT%H:%M:%SZ',  # ISO 8601 with Zulu (UTC) timezone
        '%Y-%m-%dT%H:%M:%S.%f%z',  # ISO 8601 with microseconds and timezone
        '%Y%m%dT%H%M',  # Specific uncommon format
        '%Y%m%d', '%H%M%S',  # Basic compact date and time formats
        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M',  # ISO 8601 variants
        '%Y%m%d%H%M%S',  # Another compact format
        '%d-%b-%Y',  # Day-Month-Year with month as abbreviation
        '%d %B %Y', '%B %d, %Y'  # Full month name formats
    ]

    # Try predefined formats first
    for fmt in formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            # Check if timezone is needed and is present
            if return_tz and '%z' in fmt or 'Z' in fmt:
                tz = parsed_date.tzinfo
                return fmt, str(tz)
            elif return_tz:
                return fmt, None
            return fmt
        except ValueError:
            continue

    # Custom format detection with regular expressions
    custom_patterns = {
        r'\d{8}T\d{4}': '%Y%m%dT%H%M%S',  # Matches '20230101T000000'
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}': '%Y-%m-%dT%H:%M:%S',  # Matches '2023-01-01T00:00:00'
        r'\d{12}': '%Y%m%d%H%M%S',  # Matches '20230101000000'
        r'\d{2}-\d{2}-\d{4} \d{2}:\d{2}': '%m-%d-%Y %H:%M'  # Matches '01-02-2023 12:00'
    }

    for pattern, fmt in custom_patterns.items():
        if re.match(pattern, date_str):
            if return_tz and '%z' in fmt or 'Z' in fmt:
                parsed_date = datetime.strptime(date_str, fmt)
                tz = parsed_date.tzinfo
                return fmt, str(tz)
            elif return_tz:
                return fmt, None
            return fmt

    raise ValueError("Date format not recognized")


def str2datetime(date_str, date_format=None, return_format_and_tz=False, accept_datetime=True, tz=None, naive=False):
    """
    Convert a string to a datetime object using a specified format or deduced format.

    Args:
        date_str (str): Date string to convert.
        date_format (str): Format of the date string, or None to deduce the format.
        return_format_and_tz (bool): If True, returns the deduced format along with the datetime object.
        accept_datetime (bool): If True, skips conversion if the input is already a datetime object.
        tz (str or bool): Timezone to localize the datetime object to, True for UTC, False for local timezone, or None for no conversion.
        naive (bool): If True, returns a naive datetime object.

    Returns:
        datetime: Converted datetime object.

    Raises:
        ValueError: If the date format is not deducible.
    """
    if accept_datetime and isinstance(date_str, datetime):
        return (date_str, None) if return_format_and_tz else date_str
    if date_format is None:
        date_format, _tz = deduce_date_format(date_str, return_tz=True)
    date_out = datetime.strptime(date_str, date_format)
    if tz is not None:
        date_out = to_timezone(date_out, None if isinstance(tz, bool) else tz, local=tz is False, naive=naive)
    if return_format_and_tz:
        return date_out, date_format, _tz if tz is None else (get_timezone(date_out) if isinstance(tz, bool) else tz)
    return date_out


def series_str2datetime(date_series, date_format=None, return_format_and_tz=False, accept_datetime=True, tz=None, naive=False):
    """
    Convert a series of date strings to datetime objects using a specified format or deduced format.

    Args:
        date_series (pd.Series): Series of date strings to convert.
        date_format (str): Format of the date strings, or None to deduce the format.
        return_format_and_tz (bool): If True, returns the deduced format along with the datetime objects.
        accept_datetime (bool): If True, skips conversion if the input is already a datetime object.
        tz (str or bool): Timezone to localize the datetime objects to, True for UTC, False for local timezone, or None for no conversion.
        naive (bool): If True, returns naive datetime objects.

    Returns:
        pd.Series: Series of converted datetime objects. Optionally returns a tuple with formats and timezones.
    """
    if accept_datetime and pd.api.types.is_datetime64_any_dtype(date_series):
        return (date_series, None) if return_format_and_tz else date_series

    if date_series.empty:
        return date_series

    if date_format is None:
        # Use the first non-null item to deduce format
        first_valid_index = date_series.dropna().index[0]
        date_format, _tz = deduce_date_format(date_series.loc[first_valid_index], return_tz=True)

    # Convert all date strings in the series
    date_out = pd.to_datetime(date_series, format=date_format, errors='raise')

    if tz is not None:
        if isinstance(tz, bool):
            if tz:
                date_out = date_out.dt.tz_localize('UTC')
            else:
                date_out = date_out.dt.tz_localize(None)
        else:
            date_out = date_out.dt.tz_convert(tz)

    if naive:
        date_out = date_out.dt.tz_localize(None)

    if return_format_and_tz:
        return date_out, date_format, _tz if tz is None else tz

    return date_out


def calculate_date_ranges(start_date, end_date=None, interval='1M', divisor=1, date_format=None):
    """
    Calculate and format date ranges within a specified period using a fixed interval or a divisor.

    Args:
        start_date (str or datetime): Start of the period, formatted as a string or a datetime object.
        end_date (str, datetime, None): End of the period, formatted as a string or a datetime object, or None to use current datetime.
        interval (str): Pandas frequency string specifying the interval length (e.g., '2D', '3M').
        divisor (int): Method to divide the period based on divisions desired.
        date_format (str): Format of the date string, or None to deduce the format.

    Returns:
        list of tuples: Each tuple represents a (start, end) interval within the period.

    Raises:
        ValueError: If start_date is after end_date, or if the date format is not deducible.
    """
    original_class = start_date.__class__
    date_format = None

    # Convert start_date if it's a string and deduce format
    if isinstance(start_date, str):
        date_format, tz = deduce_date_format(start_date, return_tz=False)
        start_date = datetime.strptime(start_date, date_format)

    if end_date is None:
        end_date = now_as_tz(start_date)
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, date_format)

    if start_date >= end_date:
        raise ValueError("start_date must be before end_date")

    cast_fn = lambda x: x if original_class is datetime else original_class(x.strftime(date_format) if date_format else x)
    cast_fn_ls = lambda x: list(map(cast_fn, x))
    total_seconds = (end_date - start_date).total_seconds()
    step_seconds = total_seconds / divisor
    if interval is not None:
        date_ranges = pd.date_range(start=start_date, end=end_date, freq=interval).tolist()
        dt_range_step = (date_ranges[1] - date_ranges[0]).total_seconds()
        if dt_range_step < step_seconds:
            return cast_fn_ls(date_ranges)
        else:
            step_seconds = total_seconds / divisor

    date_ranges = [start_date]
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(seconds=step_seconds), end_date)
        date_ranges.append(current_end)
        current_start = current_end
    return cast_fn_ls(date_ranges)


def create_intervals_from_timestamps(timestamps):
    """
    Create intervals from a list of sorted timestamps.

    Args:
        timestamps (list of datetime): A list of datetime objects sorted in ascending order.

    Returns:
        list of tuples: Each tuple contains (start, end) representing consecutive timestamp intervals.
    """
    if not timestamps:
        return []

    intervals = []
    for i in range(len(timestamps) - 1):
        start = timestamps[i]
        end = timestamps[i + 1]
        intervals.append((start, end))

    return intervals


def make_naive_datetime(obj):
    """
    Converts various datetime objects, including pandas Timestamp and Series containing datetime objects,
    from timezone-aware to naive (timezone-unaware).

    Args:
        obj: A datetime-like object (datetime.datetime, pandas.Timestamp, pandas.Series with datetime64).

    Returns:
        The converted naive datetime object.

    Raises:
        TypeError: If the object's type is not supported.
    """
    # Handle datetime.datetime objects
    if isinstance(obj, datetime):
        if obj.tzinfo is not None:
            return obj.replace(tzinfo=None)
        else:
            return obj  # Already naive

    # Handle pandas.Timestamp objects
    elif isinstance(obj, pd.Timestamp):
        if obj.tz is not None:
            return obj.tz_localize(None)
        else:
            return obj  # Already naive

    # Handle pandas.Series objects
    elif isinstance(obj, pd.Series):
        if is_datetime64_any_dtype(obj):
            return obj.dt.tz_localize(None)
        else:
            raise TypeError("Provided pandas Series does not contain datetime-like data.")

    # Raise an error if the object type is unsupported
    else:
        raise TypeError("Unsupported datetime object type.")