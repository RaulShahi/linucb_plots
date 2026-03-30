import argparse
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from utils import *

plt.rcParams.update({
    "font.size": 28,
    "axes.titlesize": 28,
    "axes.labelsize": 28,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 28,
})


def plot_response_files(kwargs):
    """
    Generate plots based on response and throughput data.

    Parameters
    ---------
        trace_response_data : Data containing the trace response.
        measured_throughput_data :  DataFrame containing measured throughput data.
                                    Expected columns include 'time' and 'throughput'

    Returns:
        None : Saves the generated plots as PNG files.
    """
    trace_response_data = kwargs["trace_response_data"]
    measured_throughput_data = kwargs["measured_throughput_data"]
    combined_data = kwargs["combined_data"]


    df = pd.DataFrame(trace_response_data)
    pd.set_option("display.max_rows", None)


    df_txs = df[df["trace_type"] == "txs"]


    df_est_tp = df[df["trace_type"] == "est_tp"]
    #print("est_tp",df_est_tp)
    min_time = df["time"].min()
    max_time = df["time"].max()

    time_range = max_time - min_time
    bin_size = 2
    width_factor = 0.4
    max_pixel_width = 64000  # hard pixel limit for width

    # Dynamically compute figure width and dpi
    #desired_fig_width = time_range * width_factor
    #fig_width = min(desired_fig_width, 50)
    fig_width = 100
    #dpi = min(300, max_pixel_width / fig_width)
    dpi = 300

    pixel_width = fig_width * dpi
    print(f"[INFO] time_range: {time_range:.2f}, fig_width: {fig_width:.2f}, dpi: {dpi:.2f}, pixel_width: {pixel_width:.2f}")

    base_height = 15
    num_subplots = 2
    total_height = base_height * num_subplots

    first_key, first_data = next(iter(combined_data.items()))

    fig1 = plt.figure(figsize=(fig_width, total_height))
    fig1.set_size_inches(fig_width, total_height)
    gs = gridspec.GridSpec(num_subplots, 1, height_ratios=[4, 1])

    ax1 = fig1.add_subplot(111)


    rounded_positions, modes_between_lines = add_grid_lines_to_separate_modes(ax1, df)
    bin_edges = get_bin_edges(min_time, max_time, bin_size)

    rate_x_limit = plot_rate_vs_time(
        {
            "df": pd.DataFrame(first_data),
            "ax": ax1,
            "rounded_position": rounded_positions,
            "modes_between_lines": modes_between_lines,
        }
    )


    tmin = min(measured_throughput_data["time"].min(), df_est_tp["time"].min())
    tmax = max(measured_throughput_data["time"].max(), df_est_tp["time"].max())


    fig1.tight_layout(pad=0.6)
    fig1.savefig("linucb_v2_ra_vs_minstrel_rate_plot.jpg", dpi=dpi, facecolor="white")
    plt.close(fig1)

    fig2 = plt.figure(figsize=(fig_width, total_height))
    fig2.set_size_inches(fig_width, total_height)

    gs = gridspec.GridSpec(num_subplots, 1, height_ratios=[3, 3])
    ax3 = fig2.add_subplot(gs[0])
    plot_throughput_vs_time(
        {
            "df": measured_throughput_data,
            "ax": ax3,
            "rounded_position": rounded_positions,
            "modes_between_lines": modes_between_lines
        }
    )

    ax4 = fig2.add_subplot(gs[1])
    plot_estimated_throughput(
        {
            "df": df_est_tp,
            "ax": ax4,
            "rounded_position": rounded_positions,
            "modes_between_lines": modes_between_lines,
        }
    )

    ax3.tick_params(labelbottom=False)
    ax3.set_xlim(left=0, right=tmax)
    ax3.margins(x=0)
    fig2.savefig("linucb_v2_tp_plot.png",dpi=dpi, facecolor="white")
    plt.close(fig2)


if __name__ == "__main__":
    """
    Processes CSV files in a specified directory and generates plots based on the processed data.
    """
    parser = argparse.ArgumentParser(description="Process CSV files in a directory.")
    parser.add_argument(
        "directory", type=str, help="The directory path to process files from."
    )
    args = parser.parse_args()
    (
        measured_throughput_file_dict,
        expected_throughput_files_dict,
        trace_response_files_dict,
        cpu_usage_files_dict,
    ) = categorize_files(args.directory)


    #ap_cpu_usage, sta_cpu_usage = process_cpu_usage_data(cpu_usage_files_dict)
    measured_throughput_data = process_measured_throughput_files(
        measured_throughput_file_dict
    )
    trace_response_data, combined_data = process_trace_response_files(trace_response_files_dict)
    plot_response_files(
        {
            "trace_response_data": trace_response_data,
            "measured_throughput_data": measured_throughput_data,
            #"ap_cpu_usage": ap_cpu_usage,
            #"sta_cpu_usage": sta_cpu_usage,
            "combined_data" : combined_data
        }
    )
