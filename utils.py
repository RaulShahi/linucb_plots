import os
import pandas as pd
from datetime import datetime, timedelta
import csv
import seaborn as sns
import numpy as np
import toml
import glob
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import to_rgba
from matplotlib.ticker import MultipleLocator



mode_colors = {
        "mmrrs": "#f0f8ff",
        "kernel_minstrel": "#fff5e6",
        "minstrel": "#ffe6f0",
        "linucb": "#92d5d8",
    }

atten_colors={    "40": "#2ecc71",  # Green (Safe)
    "45": "#f1c40f",  # Yellow (Caution)
    "50": "#e67e22",  # Orange
    "55": "#e74c3c",  # Red (Danger)
    "60": "#c0392b",  # Dark Red
    "65": "#8e44ad",  # Purple (Severe / very weak signal)
    "70": "#2c3e50", # Dark Grey (No connectivity zone)
    }


def hex_to_int(hex_str):
    try:
        return int(hex_str, 16)
    except ValueError as e:
        print(f"Error converting '{hex_str} to integer: {e}'")
        return None


def hex_to_time(hex_time):
    """Converts a hexadeical time representation into a human-readable datetime object.

    Parameters
    ----------
        hex_time (str): The hexadecimal string representing time in nanoseconds since the Unix epoch.

    Returns
    --------
        datetime: The corresponding datetime object if the conversion is successful.
        None: If the input is invalid or the conversion fails, returns None and logs an error.

    Raises
    -------
        ValueError: If the input is not a valid hexadecimal string.
        OSError: If the computed time value is invalid.
    """
    try:
        nanoseconds = int(hex_time, 16)
        seconds = nanoseconds / 1e9
        delta = timedelta(seconds=seconds)
        epoch = datetime(1970, 1, 1)
        actual_time = epoch + delta
        return actual_time

    except (ValueError, OSError) as e:
        print(f"Error converting hex time '{hex_time}': {e}")
        return None


def extract_expected_throughput_in_order(file_path):
    """
    Extracts the expected throughput order from a filename.

    Assumes the filename format is: "<order>_lowest_mode_expected_throughput",
    where the order is the first part of the filename, separated by underscores.

    Parameters
    ----------
        file_path (str): The full path to the file.

    Returns
    --------
        str: The extracted order as a string if successful.
        None: If the filename format is invalid or an error occurs.

    Raises
    -------
        IndexError: If the filename does not contain an underscore-separated order.
        ValueError: If there are issues with string processing.
    """
    filename = os.path.basename(file_path)
    try:
        return filename.split("_")[0]
    except (IndexError, ValueError) as e:
        print(f"Error extracting order from filename '{filename}': {e}")
        return None


def extract_mode_from_filename(filename):
    """
    Extracts the mode from a filename.

    Assumes the filename format is: "1_ap_1-Lowest_Power_orca_trace.csv",
    where the mode is the part after the last underscore (`_`) and before the hyphen (`-`).

    Parameters
    ----------
        filename (str): The full path to the file.

    Returns
    -------
        str: The extracted mode as a string.

    Raises
    ------
        IndexError: If the filename format is invalid and the mode cannot be extracted.
    """

    base_filename = os.path.basename(filename)
    if "_orca_trace.csv" in base_filename:
        base_filename = base_filename.replace("_orca_trace.csv", "")
    parts = base_filename.split("_", 2)
    return parts[-1].split("-", 1)[1]
def extract_attenuations(settings_path):
    """Extract attenuatioin from the settings.toml file to link them with the corresponding ap and sta parts"""

    with open(settings_path, "r") as f:
        config = toml.load(f)

    attenuation_values = []

    if "parts" in config:
        for part in config["parts"]:
            att = part.get("attenuation", {})
            value = att.get("value")
            if value is not None:
                attenuation_values.append(value)
    return attenuation_values


def extract_experiment_order(folder_path):
    """Inorder to process the files in order of their occurence,
    files are sorted by extracting the part number from the filename"""
    files = glob.glob(f"{folder_path}/*.csv")
    ordered_ap_files = []
    ordered_sta_files = []
    measured_throughput_file = None


    for filepath in files:
        filename = os.path.basename(filepath)
        if '1_2a_12_b5_38_76_b5' in filename:
            target_list = ordered_ap_files
        elif '1_26_10_cd_d3_b9_f8' in filename:
            target_list = ordered_sta_files
        elif "throughput" in filename.lower():
            measured_throughput_file = filepath
        else:
            continue
        try:
            parts = filename.split("-")[0]  # e.g., "1_ap_56"
            order = int(parts.split("_")[-1])
            target_list.append((order, filepath))
        except (IndexError, ValueError) as e:
            print(f"Error extracting order from filename '{filename}': {e}")


    ordered_ap_files.sort(key=lambda x: x[0])
    ordered_sta_files.sort(key=lambda x: x[0])


    # Return only filepaths in the sorted order
    return [f[1] for f in ordered_ap_files], [f[1] for f in ordered_sta_files], measured_throughput_file

def experiment_order_key(filepath: str) -> int:
    filename = os.path.basename(filepath)
    left = filename.split("-", 1)[0]          # "1_26_10_cd_d3_b9_f8_3"
    order_str = left.split("_")[-1]           # "3"
    return int(order_str)

def load_or_process_measured_throughput(iteration_files_dict, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached measured throughput from {cache_path}")
        return pd.read_pickle(cache_path)

    print("No measured throughput cache found. Processing raw files...")
    return process_measured_throughput_files(
        iteration_files_dict,
        cache_path=cache_path,
    )

def load_or_process_trace_response(
    iteration_files_dict,
    attenuation_values,
    averaged_cache_path,
    first_iter_cache_path,
):
    if os.path.exists(averaged_cache_path) and os.path.exists(first_iter_cache_path):
        print("Loading cached trace response data...")
        return (
            pd.read_pickle(averaged_cache_path),
            pd.read_pickle(first_iter_cache_path),
        )

    print("No trace response cache found. Processing raw files...")
    return process_trace_response_files(
        iteration_files_dict,
        attenuation_values,
        averaged_cache_path=averaged_cache_path,
        first_iter_cache_path=first_iter_cache_path,
    )

def categorize_files(directory):
    """
    Categorizes CSV files in the given directory into measured throughput,
    expected throughput, and response files based on their filenames.

    Iterates over subfolders in the provided directory, looking for CSV files
    and categorizing them based on keywords in their filenames:
    - Files containing "expected_throughput" are categorized as expected throughput files.
    - Files containing "throughput" are categorized as measured throughput files.
    - Files containing "_cpu_" are categorized as cpuusage files
    - All other files are categorized as response files, excluding the 'ap_orca_header.csv' file.

    Parameters
    ---------
        directory (str): The path to the directory containing iteration folders.

    Returns
    --------
        tuple: A tuple containing three dictionaries:
            - measured_throughput_files: Dictionary with iteration folder names as keys
              and lists of file paths to measured throughput files as values.
            - expected_throughput_files: Dictionary with iteration folder names as keys
              and lists of file paths to expected throughput files as values.
            - response_files: Dictionary with iteration folder names as keys and lists of
              file paths to response files as values.
    """
    measured_throughput_files = {}
    response_files = {}
    expected_throughput_files = {}
    cpu_usage_files = {}

    try:
        for iteration_folder in os.listdir(directory):
            iteration_path = os.path.join(directory, iteration_folder)
            if os.path.isdir(iteration_path):
                print(
                    f"Listing and processing files in iteration folder: {iteration_path}"
                )
                measured_throughput_files[iteration_folder] = []
                response_files[iteration_folder] = []
                expected_throughput_files[iteration_folder] = []
                cpu_usage_files[iteration_folder] = []

                for filename in os.listdir(iteration_path):
                    file_path = os.path.join(iteration_path, filename)

                    if os.path.isfile(file_path) and filename.endswith(".csv"):
                        if "orca_header.csv" in filename:
                            continue
                        if "expected_throughput" in filename.lower():
                            expected_throughput_files[iteration_folder].append(
                                file_path
                            )
                        elif "throughput" in filename.lower():
                            measured_throughput_files[iteration_folder].append(
                                file_path
                            )
                        elif "_cpu_" in filename.lower():
                            cpu_usage_files[iteration_folder].append(
                                file_path
                            )
                        else:
                            response_files[iteration_folder].append(file_path)
        return measured_throughput_files, expected_throughput_files, response_files, cpu_usage_files

    except Exception as e:
        print(f"An error occurred: {e}")



def read_csv_to_dict(file_path, delimiter):
    """
    Reads the trace response CSV files obtained after the experiment.

    This function processes the trace response by filtering the lines based on trace type and power type.
    Specifically, it:
    - Filters out non-relevant rows (non "txs" and "est_tp").
    - Collects the "txs" lines where the packet transmission is successful, extracting the timestamp, rate, and power.
    - Separates "txs" lines based on the power type (e.g., "set_power", "set_power_rc").

    Parameters
    -----------
    file_path : str
        The path to the CSV file to be processed.

    delimiter : str
        The delimiter used in the CSV file (e.g., ',' or ';').

    Returns
    --------
    list of dict
        A list of dictionaries where each dictionary contains the filtered data for each relevant line,
        with keys like "trace_type", "time", "rate", "power", and "power_type".

    Notes
    ------
    - The function assumes that the trace response contains lines with the following relevant types: "txs", "est_tp", "set_power", and "set_power_rc".
    - For the "txs" trace type, only lines with a successful packet transmission are processed, extracting the rate and power.
    - The function also handles a "three-phase power model" by tracking and separating the power type from the power controller.
    """
    filtered_data = []
    current_power_type = 'not_from_power_controller'

    try:
        with open(file_path, "r") as file:
            csv_reader = csv.reader(file, delimiter=delimiter)
            for row in csv_reader:
                if row[2] == "log":
                    current_power_type = row[-1]
                    continue

                if row[2] not in {"txs", "est_tp"}:
                    continue

                if len(row[1]) == 16:
                    try:
                        actual_time = hex_to_time(row[1])

                        if row[2] == "est_tp":
                            filtered_data.append({
                                "trace_type": row[2],
                                "actual_time": actual_time,
                                "est_tp": int(row[4], 16) / 10
                            })

                        elif row[2] == "txs" and len(row) >= 11:
                            for i in range(len(row) - 1, 6, -1):
                                split_values = row[i].split(",")
                                if split_values[-1]:
                                    try:
                                        rate = split_values[0]
                                        power = int(split_values[-1], 16)
                                        filtered_data.append({
                                            "trace_type": row[2],
                                            "actual_time": actual_time,
                                            "rate": rate,
                                            "power": power,
                                            "power_type": current_power_type,
                                            "attempts" : int(row[4]),
                                            "acks" : int(row[5]),
                                        })
                                        break
                                    except ValueError:
                                        pass

                    except ValueError as e:
                        print(f"ValueError processing row {row}: {e}")

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return filtered_data


def process_measured_throughput_files(iteration_files_dict, cache_path=None):
    """
    Processes the measured throughput files for multiple iterations, averaging data based on the
    first iteration's timestamps.

    Parameters
    ----------
    iteration_files_dict : dict
        A dictionary where the keys are iteration identifiers (e.g., "1", "2", etc.), and the values
        are lists of file paths corresponding to throughput data files for each iteration.
        Each file is assumed to have a structure where the first column is time and the second column is throughput.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the averaged throughput data. The `time` column contains the timestamps,
        and the `throughput` column contains the averaged throughput values for each timestamp across iterations.

    Notes
    -----
    - The function assumes that the first iteration's timestamps serve as the reference for alignment.
    - The throughput values are averaged across all available iterations for each timestamp.
    - If a timestamp is missing in a subsequent iteration, it is ignored in the averaging calculation.
    """
    combined_data = {}
    for iteration, throughput_files in iteration_files_dict.items():
        iteration_data = []

        for file_path in throughput_files:
            print(f"Processing throughput file: {file_path} for iteration: {iteration}")
            data = pd.read_csv(
                file_path,
                sep=r"\s+",
                header=None,
                skiprows=1,
                names=["time", "throughput"],
            )
            iteration_data.append(data)

        combined_data[iteration] = pd.concat(iteration_data, ignore_index=True)

    first_iteration_data = combined_data.get("1")
    if first_iteration_data is None:
        first_iteration_data = combined_data.get("01", pd.DataFrame())

    averaged_data = []

    for row_index in range(len(first_iteration_data)):
        base_entry = first_iteration_data.iloc[row_index]
        averaged_entry = {"time": base_entry["time"]}
        values_to_average = [base_entry["throughput"]]

        for iteration, data in combined_data.items():
            if iteration == "1" or iteration == "01":
                continue
            if row_index < len(data):
                values_to_average.append(data.iloc[row_index]["throughput"])

        averaged_entry["throughput"] = sum(values_to_average) / len(values_to_average)
        averaged_data.append(averaged_entry)

    averaged_df = pd.DataFrame(averaged_data)

    if cache_path is not None:
        averaged_df.to_pickle(cache_path)

    return averaged_df


def process_trace_response_files(iteration_files_dict, attenuation_values, averaged_cache_path=None, first_iter_cache_path=None):
    """
    Processes the response CSV files for multiple iterations, averaging data based on the first iteration's row indices.

    This function reads and processes trace response files for multiple iterations. It extracts relevant information such as
    timestamp, mode, trace type (e.g., "txs" or "est_tp"), rate, and power. The data from each iteration is combined and
    averaged for each row index based on the first iteration's data, ensuring that each row corresponds to a common
    timestamp across all iterations.

    Parameters
    ----------
    iteration_files_dict : dict
        A dictionary where keys are iteration identifiers (e.g., "1", "2", etc.) and values are lists of file paths
        to the trace response CSV files corresponding to each iteration.

    Returns
    -------
    list of dict
        A list of dictionaries where each dictionary represents an averaged entry. Each dictionary contains the following keys:
        - `time`: The timestamp of the trace event.
        - `mode`: The mode associated with the trace event.
        - `trace_type`: The type of trace (e.g., "txs", "est_tp").
        - `est_tp`: The estimated throughput (if present).
        - `rate`: The average rate (if present).
        - `power`: The average power (if present).
        - `power_type`: The power type (if present).

    Notes
    -----
    - The function averages data for each row index based on the first iteration's timestamps.
    - If no data is found for an iteration, that iteration is skipped.
    - The `rate` and `power` values are averaged across all available entries for each row index.
    - If the `power_type` and `mode` match, the function ensures that those entries are averaged together.

    """
    combined_data = {}

    for iteration, trace_response_files in iteration_files_dict.items():
        trace_response_files.sort(key=experiment_order_key)
        att_vals = [x for x in attenuation_values for _ in range(2)]

        if len(att_vals) != len(trace_response_files):
            print(attenuation_values)
            print(len(attenuation_values), len(trace_response_files))
            for file in trace_response_files:
                print(file)
            print("Lengths of attenuations and files do not match")

        iteration_data = []

        for file_path, attenuation in zip(trace_response_files, att_vals):
            print(f"Processing response file: {file_path} for iteration: {iteration} at attenuation {attenuation}")
            mode = extract_mode_from_filename(file_path)
            data = read_csv_to_dict(file_path, delimiter=";")

            if not data:
                continue

            for entry in data:
                entry["mode"] = mode
                entry["attenuation"] = attenuation
                iteration_data.append(entry)

        if not iteration_data:
            combined_data[iteration] = []
            continue

        iteration_data.sort(key=lambda x: x["actual_time"])
        base_time = iteration_data[0]["actual_time"]

        for entry in iteration_data:
            entry["time"] = (entry["actual_time"] - base_time).total_seconds()

        combined_data[iteration] = iteration_data

    first_iteration_data = combined_data.get("1") or combined_data.get("01") or []
    averaged_data = []

    for row_index in range(len(first_iteration_data)):
        base_entry = first_iteration_data[row_index]
        averaged_entry = base_entry.copy()

        if "power_type" in base_entry:
            averaged_entry["power_type"] = base_entry["power_type"]

        values_to_average = [base_entry]

        for iteration, data in combined_data.items():
            if iteration == "1" or iteration == "01":
                continue
            if row_index < len(data):
                current_entry = data[row_index]

                if current_entry["trace_type"] == "txs":
                    if (
                        current_entry["mode"] == base_entry["mode"]
                        and (
                            "power_type" not in current_entry
                            or current_entry["power_type"] == base_entry.get("power_type")
                        )
                    ):
                        values_to_average.append(current_entry)

                elif current_entry["trace_type"] == "est_tp":
                    values_to_average.append(current_entry)

        power_values = [value["power"] for value in values_to_average if "power" in value]
        if power_values:
            averaged_entry["power"] = sum(power_values) // len(power_values)

        averaged_data.append(averaged_entry)

    averaged_df = pd.DataFrame(averaged_data)
    first_iteration_df = pd.DataFrame(first_iteration_data)

    if averaged_cache_path is not None:
        averaged_df.to_pickle(averaged_cache_path)

    if first_iter_cache_path is not None:
        first_iteration_df.to_pickle(first_iter_cache_path)

    return averaged_df, first_iteration_df

def get_boxplot_properties():
    """
    Returns properties for customizing a boxplot's appearance.

    This function provides default properties for different components of a boxplot, including the box, median line,
    whiskers, and caps. These properties are used to customize the appearance of boxplots in visualizations.

    Returns
    -------
    tuple of dicts
        A tuple containing four dictionaries:
        - `boxprops` : dict
            Properties for the box (i.e., the main body of the boxplot).
        - `medianprops` : dict
            Properties for the median line in the boxplot.
        - `whiskerprops` : dict
            Properties for the whiskers extending from the box.
        - `capprops` : dict
            Properties for the caps at the ends of the whiskers.
    """


    boxprops = dict(facecolor="none", edgecolor="black")
    medianprops = dict(color="black")
    whiskerprops = dict(color="black")
    capprops = dict(color="black")

    return boxprops, medianprops, whiskerprops, capprops


def bin_time(df, time_column="time", bin_size=10):
    """
    Bins a time column of a DataFrame into intervals of specified size.

    This function divides the data in the given `time_column` of the DataFrame into discrete time bins.
    The size of the bins is determined by the `bin_size` parameter. The binning is performed by creating
    bin edges based on the minimum and maximum time values and then assigning each row to its corresponding
    bin.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be binned.
    time_column : str, optional
        The name of the column containing time values to be binned (default is "time").
    bin_size : int, optional
        The size of each time bin in the same unit as the `time_column` (default is 10).

    Returns
    -------
    pandas.DataFrame
        The original DataFrame with an additional column `binned_time`, which contains the binned time intervals.
    """
    min_time = df[time_column].min()
    max_time = df[time_column].max()

    bin_edges = np.arange(min_time, max_time + bin_size, bin_size)

    rounded_bin_edges = np.round(bin_edges / bin_size) * bin_size

    if rounded_bin_edges[-1] < max_time:
        rounded_bin_edges = np.append(
            rounded_bin_edges, np.ceil(max_time / bin_size) * bin_size
        )
    rounded_bin_edges = rounded_bin_edges[rounded_bin_edges <= max_time]

    df["binned_time"] = pd.cut(
        df[time_column], bins=rounded_bin_edges, duplicates="drop"
    )
    return df


def get_bin_edges(min_time, max_time, bin_size=10):
    """
    Generates bin edges for a time range with a specified bin size.

    Parameters
    ----------
    min_time : int or float
        The starting value of the time range.
    max_time : int or float
        The ending value of the time range.
    bin_size : int or float
        The size of each bin, specifying the interval between consecutive bin edges.

    Returns
    -------
    numpy.ndarray
        An array of bin edges from `min_time` to `max_time` with a step size of `bin_size`.
    """
    return np.arange(min_time, max_time + bin_size, bin_size)


def add_grid_lines_to_separate_modes(ax, df):
    """
    Adds vertical grid lines to the plot at times where the mode changes.

    This function identifies time points where the mode changes in the given
    DataFrame and adds vertical dashed grid lines to the plot at these times.
    The grid lines are drawn at each mode transition, providing a clear separation
    between different modes on the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to draw the grid lines.
    df : pandas.DataFrame
        A DataFrame containing at least two columns: 'time' and 'mode'.
        The 'time' column indicates the time points, and the 'mode' column indicates
        the mode of operation at each time.

    Returns
    -------
    list of int
        A sorted list of unique, rounded time positions where the mode changes.
    list of str
        A list of modes corresponding to the times where the mode changes.
    """
    df_sorted = df.sort_values(by="time", ascending=True)
    prev_mode = None
    prev_time = None
    prev_attenuation = None
    line_positions = []
    modes_between_lines = []
    for idx, row in df_sorted.iterrows():
        current_mode = row["mode"]
        current_time = row["time"]
        current_attenuation = row["attenuation"]

        if current_mode != prev_mode:
            if prev_mode is not None and prev_time is not None and prev_attenuation is not None:
                line_positions.append(prev_time)
                modes_between_lines.append((prev_mode, prev_attenuation))
            prev_mode = current_mode
            prev_attenuation = current_attenuation
        prev_time = current_time

    first_time = df_sorted["time"].iloc[0]
    last_time = df_sorted["time"].iloc[-1]

    line_positions.insert(0, first_time)
    line_positions.append(last_time)
    modes_between_lines.append((df_sorted["mode"].iloc[-1], df_sorted["attenuation"].iloc[-1]))

    rounded_positions = [round(pos) for pos in line_positions]
    for pos in sorted(set(line_positions)):
        ax.axvline(x=pos, color="black", linestyle="--", linewidth=1)

    return sorted(set(rounded_positions)), modes_between_lines


def separating_traces_per_mode(df):
    mode_dfs = {}
    current_rows = []
    current_mode = None
    mode_counter = 0

    for _, row in df.iterrows():
        if row['mode'] == 'fixed-rate':
            continue
        if row['mode'] != current_mode:
            if current_rows:
                mode_key = f"mode_{row['mode']}_{mode_counter}"
                mode_dfs[mode_key] = pd.DataFrame(current_rows)
                mode_counter += 1
            current_rows = []
            current_mode = row['mode']
        current_rows.append(row.to_dict())

    if current_rows:
        mode_key = f"mode_{mode_counter}"
        mode_dfs[mode_key] = pd.DataFrame(current_rows)

    return mode_dfs



def scale_line_positions(line_positions, rate_x_range, power_x_range):
    """
    This function scales a list of line positions from one range (e.g., rate x-axis)
    to another range (e.g., power x-axis) by applying a scaling factor calculated
    from the ratio of the two ranges.

    Parameters
    ----------
    line_positions : list of float
        A list of positions (in the original rate x-axis range) that need to be scaled.
    rate_x_range : float
        The total range of the rate x-axis (the original range for the line positions).
    power_x_range : float
        The total range of the power x-axis (the target range to scale the line positions to).

    Returns
    -------
    list of float
        A list of scaled line positions in the target x-axis range.
    """

    scaling_factor = power_x_range / rate_x_range
    return [pos * scaling_factor for pos in line_positions]


def add_mode_legend(ax=None, fig=None, mode_colors = mode_colors, modes_between_lines = None, location = "upper right"):
    unique_modes = []
    for mode, _ in modes_between_lines:
        if mode not in unique_modes:
            unique_modes.append(mode)

    handles = [
        Patch(facecolor = mode_colors.get(mode, "'f0f0f0"), edgecolor="black", alpha=0.3, label = mode)
        for mode in unique_modes
    ]

    if ax is not None:
        ax.legend(handles = handles, title="Mode", loc=location)
    elif fig is not None:
        fig.legend(handles=handles, title = "Mode", loc = "upper center", ncol= len(unique_modes))

def plot_rate_vs_time(kwargs):
    """
    Plots rate versus time, with separate colors for different modes.

    This function creates a scatter plot showing the relationship between time and rate,
    where different modes are distinguished by different colors. Vertical lines are added
    to separate the different modes, and the x-axis is marked with rounded time positions.

    Parameters
    ----------
    kwargs : dict
        A dictionary containing the following key-value pairs:
        - "df" (pandas DataFrame): The data to be plotted, must include 'time', 'rate', and 'mode' columns.
        - "ax" (matplotlib Axes): The axes on which to plot the data.
        - "rounded_position" (list of float): The x-axis positions for the vertical lines separating modes.
        - "modes_between_lines" (list of str): The list of modes corresponding to the line positions.

    Returns
    -------
    tuple
        A tuple representing the limits of the x-axis (min, max).
    """
    df = kwargs["df"]
    ax = kwargs["ax"]
    rounded_positions = kwargs["rounded_position"]
    modes_between_lines = kwargs["modes_between_lines"]

    df = df.dropna(subset=["rate"])
    df.loc[:,"rate_int"] = df["rate"].apply(hex_to_int)
    df_sorted = df.sort_values(by="rate_int", ascending=True)
    num_modes = len(df_sorted["mode"].unique())
    color_palette = sns.color_palette("tab10", num_modes)
    sns.scatterplot(
        data=df_sorted,
        x="time",
        y="rate",
        alpha=0.7,
        ax=ax,
    )

    for i in range(len(rounded_positions) - 1):
        start, end = rounded_positions[i], rounded_positions[i + 1]
        mode, attenuation = modes_between_lines[i]
        color = mode_colors.get(mode, "#f0f0f0")
        ax.axvspan(start, end, color=color, alpha=0.3)
        ax.text(
            (start + end) / 2, 50, f"{attenuation}dB",
            ha="center", va="top", fontsize=18, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )
    ax.set_xticks(rounded_positions)
    ax.set_xticklabels([f"{int(pos)}" for pos in rounded_positions])
    #ax.xaxis.set_major_locator(MultipleLocator(5))
    #ax.legend(loc="lower left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate")
    ax.set_title("Rate vs Time")
    ax.set_xlim(df["time"].min(), df["time"].max())
    ax.invert_yaxis()

    add_mode_legend(ax=ax, mode_colors=mode_colors, modes_between_lines=modes_between_lines, location="lower left")

    return ax.get_xlim()


def calculate_mean_between_different_parts(mean_values, scaled_positions):
    """
    Calculates the mean of values within intervals defined by the scaled positions.
    This function divides the list of mean values into intervals based on the scaled positions,
    and calculates the mean for each interval. The intervals are defined by consecutive pairs of
    positions in the `scaled_positions` list. Each mean value is calculated by averaging the values
    within the corresponding interval, and the results are returned in a list.

    Parameters
    ----------
    mean_values : list of float
        A list containing the values for which the mean will be calculated over intervals.

    scaled_positions : list of float
        A list of positions that define the intervals within the `mean_values` list.
        These positions are used to split the data into intervals and calculate the mean
        for each one.

    Returns
    -------
    list of float
        A list of the calculated mean values for each interval defined by `scaled_positions`.

    """
    interval_means = []
    covered_indices = [False] * len(mean_values)

    for i in range(len(scaled_positions) - 1):
        start_idx = round(scaled_positions[i])
        end_idx = round(scaled_positions[i + 1])
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(mean_values):
            end_idx = len(mean_values)

        bins_in_interval = [
            mean_values[j] for j in range(start_idx, end_idx) if not covered_indices[j]
        ]
        for j in range(start_idx, end_idx):
            covered_indices[j] = True

        if bins_in_interval:
            interval_mean = sum(bins_in_interval) / len(bins_in_interval)
            interval_means.append(interval_mean)
    return interval_means


def plot_power_vs_time(kwargs):
    """
    Plots a boxplot of power consumption over time with separate sections for different power types.

    This function creates a boxplot for power consumption in different time bins, with each bin corresponding
    to a certain power type. The plot separates different power types (e.g., 'sample_power', 'data_power',
    'reference_power') by different colors and draws vertical lines to separate modes of measurement in the time axis.

    Parameters
    ----------
    kwargs : dict
        A dictionary containing the following key-value pairs:
        - df : pandas.DataFrame
            A DataFrame containing the data with time, power, and power_type columns.
        - ax : matplotlib.axes.Axes
            The Axes object to plot the boxplot on.
        - bin_edges : list of float
            The edges of the time bins for grouping the data.
        - rounded_positions : list of float
            The positions on the x-axis where vertical lines will be drawn to separate modes.
        - modes_between_lines : list of str
            The modes corresponding to the intervals between vertical lines.
        - rate_x_limit : list of float
            The x-axis limits for the rate plot.
        - bin_size : int
            The size of each time bin to group the data.
    Returns
    -------
    tuple of (list of float, pandas.Categorical)
        - scaled_positions : list of float
            The scaled positions of vertical lines, adjusted for the length of the bins.
        - bins : pandas.Categorical
            The time bins for which the boxplots are drawn.

    """
    df = kwargs["df"]
    ax = kwargs["ax"]
    bin_edges = kwargs["bin_edges"]
    rate_line_positions = kwargs["rounded_positions"]
    modes_between_lines = kwargs["modes_between_lines"]
    rate_x_limit = kwargs["rate_x_limit"]
    bin_size = kwargs["bin_size"]
    boxprops, medianprops, whiskerprops, capprops = get_boxplot_properties()

    df = bin_time(df, time_column="time", bin_size=bin_size)
    bins = df["binned_time"].cat.categories
    mean_values = []
    power_type_palette = {
        "not_from_power_controller": "none",
        "sample_power": "orange",
        "data_power": "green",
        "reference_power": "red",
    }
    alpha_value = 0.5

    for bin_idx, bin_edge in enumerate(bin_edges[:-1]):
        bin_data = df[(df["time"] >= bin_edges[bin_idx]) & (df["time"] < bin_edges[bin_idx + 1])]

        if not bin_data.empty:
            grouped = bin_data.groupby("power_type")

            for power_type_idx, (power_type, group) in enumerate(grouped):
                power_color = power_type_palette[power_type]
                facecolor_with_alpha = to_rgba(power_color, alpha=alpha_value)
                edge_color = "black" if power_color == "none" else power_color
                sns.boxplot(
                    data=group,
                    x=[bin_idx] * len(group),
                    y="power",
                    ax=ax,
                    boxprops=dict(facecolor=facecolor_with_alpha, edgecolor= edge_color),
                    medianprops=dict(color=power_color),
                    whiskerprops=dict(color=power_color),
                    capprops=dict(color=power_color),
                    flierprops = dict(marker='o', markerfacecolor=power_color)
                )
                mean_value = group["power"].mean()
                mean_values.append(mean_value)


    scaled_positions = scale_line_positions(
        rate_line_positions, rate_x_limit[1], len(bins)
    )
    interval_means = calculate_mean_between_different_parts(
        mean_values, scaled_positions
    )


    for i in range(len(scaled_positions) - 1):
        start, end = scaled_positions[i], scaled_positions[i + 1]
        mode = modes_between_lines[i]
        color = mode_colors.get(mode, "#f0f0f0")
        ax.axvspan(start, end, color=color, alpha=0.3)
        ax.text(
            (start + end) / 2, df["power"].min(), mode,
            ha="center", va="top", fontsize=28, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

    for i, pos in enumerate(scaled_positions[:-1]):
        next_pos = scaled_positions[i + 1]
        mode = modes_between_lines[i]
        ax.axvline(x=pos, color="black", linestyle="--", linewidth=1)

    ax.axvline(x=scaled_positions[-1], color="black", linestyle="--", linewidth=1)

    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.set_xticklabels([])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power Index")
    ax.set_title("Power vs Time")
    ax.set_xlim(left=0)
    ax.set_xlim(0, len(bins))

    handles = [
        Line2D([0], [0], color=color, lw=4, label=ptype)
        for ptype, color in power_type_palette.items()
    ]
    ax.legend(handles=handles, title="Power Type")

    return scaled_positions, bins


def plot_throughput_vs_time(kwargs):
    """
    Plots a simple line plot of throughput vs time.

    Parameters
    ----------
    kwargs : dict
        - df : pandas.DataFrame with columns ["time", "throughput"]
        - ax : matplotlib.axes.Axes
    """
    df = kwargs["df"]
    ax = kwargs["ax"]

    df = df.sort_values("time")

    ax.plot(df["time"], df["throughput"], linewidth=1)
    rounded_positions = kwargs["rounded_position"]
    modes_between_lines = kwargs["modes_between_lines"]

    ymin, ymax = ax.get_ylim()
    y_pos = (ymin + ymax) / 2

    for i in range(len(rounded_positions) - 1):
        start, end = rounded_positions[i], rounded_positions[i + 1]
        mode, attenuation = modes_between_lines[i]
        color = mode_colors.get(mode, "#f0f0f0")
        ax.axvspan(start, end, color=color, alpha=0.3)
        ax.text(
            (start + end) / 2,y_pos,f"{attenuation}dB",
            ha="center",va="top",fontsize=16, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

    ax.set_xlim(left=0)
    ax.set_xticks(rounded_positions)
    ax.set_xticklabels([f"{int(x)}" for x in rounded_positions], rotation=45)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throughput")
    ax.set_title("Throughput vs Time")


def plot_estimated_throughput(kwargs):
    """
    Line plot of estimated throughput vs time, aggregated into fixed time bins
    so it matches the throughput plot's resolution.
    """
    df = kwargs["df"]
    ax = kwargs["ax"]

    rounded_positions = kwargs["rounded_position"]
    modes_between_lines = kwargs["modes_between_lines"]

    df = df.sort_values("time")
    ax.plot(df["time"], df["est_tp"], linewidth=1)

    ymin, ymax = ax.get_ylim()
    y_pos = (ymin + ymax) / 2

    for i in range(len(rounded_positions) - 1):
        start, end = rounded_positions[i], rounded_positions[i + 1]
        mode, attenuation = modes_between_lines[i]
        color = mode_colors.get(mode, "#f0f0f0")

        ax.axvspan(start, end, color=color, alpha=0.3)
        ax.text(
            (start + end) / 2, y_pos, f"{attenuation}dB",
            ha="center", va="top", fontsize=16, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

    ax.set_xlim(left=0)
    ax.set_xticks(rounded_positions)
    ax.set_xticklabels([f"{int(x)}" for x in rounded_positions], rotation=45)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Estimated Throughput")
    ax.set_title("Estimated Throughput vs Time")

def plot_throughput_vs_time_box_plot(kwargs):
    """
    Plots a boxplot of throughput over time using fixed time bins.

    Parameters
    ----------
    kwargs : dict
        - df : pandas.DataFrame with columns ["time", "throughput"]
        - ax : matplotlib.axes.Axes
        - bin_edges : list of float
        - rounded_position : list of float
        - modes_between_lines : list
        - rate_x_limit : list or tuple
        - bin_size : int or float
    """
    df = kwargs["df"]
    ax = kwargs["ax"]
    bin_edges = kwargs["bin_edges"]
    rate_line_positions = kwargs["rounded_position"]
    modes_between_lines = kwargs["modes_between_lines"]
    rate_x_limit = kwargs["rate_x_limit"]
    bin_size = kwargs["bin_size"]

    df = df.sort_values("time").copy()
    df = bin_time(df, time_column="time", bin_size=bin_size)
    bins = df["binned_time"].cat.categories

    for bin_idx in range(len(bin_edges) - 1):
        bin_data = df[
            (df["time"] >= bin_edges[bin_idx]) &
            (df["time"] < bin_edges[bin_idx + 1])
        ]

        if not bin_data.empty:
            sns.boxplot(
                data=bin_data,
                x=[bin_idx] * len(bin_data),
                y="throughput",
                ax=ax,
                boxprops=dict(facecolor="lightblue", edgecolor="blue"),
                medianprops=dict(color="blue"),
                whiskerprops=dict(color="blue"),
                capprops=dict(color="blue"),
                flierprops=dict(
                    marker="o",
                    markerfacecolor="blue",
                    markeredgecolor="blue"
                )
            )

    scaled_positions = scale_line_positions(
        rate_line_positions, rate_x_limit[1], len(bins)
    )

    for i in range(len(scaled_positions) - 1):
        start, end = scaled_positions[i], scaled_positions[i + 1]
        mode, attenuation = modes_between_lines[i]
        color = mode_colors.get(mode, "#f0f0f0")

        ax.axvspan(start, end, color=color, alpha=0.3)

        ax.text(
            (start + end) / 2,
            30,
            f"{mode}",
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

        ax.text(
            (start + end) / 2,
            0,
            f"{attenuation} dB",
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

    for pos in scaled_positions:
        ax.axvline(x=pos, color="black", linestyle="--", linewidth=1)

    tick_positions = scaled_positions
    tick_labels = [f"{int(pos)}" for pos in rate_line_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throughput")
    ax.set_title(f"Throughput vs Time (Boxplot, bin={bin_size}s)")
    ax.set_xlim(0, len(bins))

def rate_selection_count_plot(kwargs):
    df = kwargs["df"]
    ax = kwargs["ax"]
    mode = kwargs["mode"]
    attenuation = kwargs["attenuation"]

    filtered_df = df[
        (df["attenuation"] == attenuation) &
        (df["mode"] == mode)
    ].copy()

    if filtered_df.empty:
        ax.set_title(f"No data for mode={mode}, attenuation={attenuation}")
        ax.set_xlabel("Rate")
        ax.set_ylabel("Selection Count")
        return

    summary = (
        filtered_df.groupby("rate", as_index=False)
        .agg(
            selection_count=("rate", "size"),
            total_acks=("acks", "sum"),
            total_attempts=("attempts", "sum"),
        )
        .sort_values("rate")
    )

    summary["success_ratio"] = (
        summary["total_acks"] / summary["total_attempts"]
    ).fillna(0)

    summary = summary.sort_values("selection_count", ascending=False).head(20)
    # Bar plot for selection count
    x = range(len(summary))
    ax.bar(x, summary["selection_count"], width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["rate"], rotation=45)
    ax.set_xlabel("Rate")
    ax.set_ylabel("Selection Count")
    ax.set_title(f"Rate Selection Count and Success Ratio ({mode}, att={attenuation})")

    # Second y-axis for success ratio
    ax2 = ax.twinx()
    ax2.plot(
        x,
        summary["success_ratio"],
        marker="o",
        linewidth=3,
        linestyle="--"
)
    ax2.set_ylabel("Success Ratio")
    ax2.set_ylim(0, 1.05)
