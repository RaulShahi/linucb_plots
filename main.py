import os
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from utils import *

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
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
    first_iteration_data = kwargs["first_iteration_data"]


    df = trace_response_data
    pd.set_option("display.max_rows", None)

    df_txs = df[df["trace_type"] == "txs"]
    df_est_tp = df[df["trace_type"] == "est_tp"]
    min_time = df["time"].min()
    max_time = df["time"].max()

    time_range = max_time - min_time
    bin_size = 2
    width_factor = 0.4
    max_pixel_width = 64000  # hard pixel limit for width

    # Dynamically compute figure width and dpi
    #desired_fig_width = time_range * width_factor
    #fig_width = min(desired_fig_width, 50)
    fig_width = 50
    #dpi = min(300, max_pixel_width / fig_width)
    dpi = 200

    pixel_width = fig_width * dpi
    print(f"[INFO] time_range: {time_range:.2f}, fig_width: {fig_width:.2f}, dpi: {dpi:.2f}, pixel_width: {pixel_width:.2f}")

    base_height = 15
    num_subplots = 2
    total_height = base_height * num_subplots


    fig1 = plt.figure(figsize=(fig_width, total_height))
    fig1.set_size_inches(fig_width, total_height)
    gs = gridspec.GridSpec(num_subplots, 1, height_ratios=[4, 1])

    ax1 = fig1.add_subplot(111)

    rounded_positions, modes_between_lines = add_grid_lines_to_separate_modes(ax1, df)
    bin_edges = get_bin_edges(min_time, max_time, bin_size)

    rate_x_limit = plot_rate_vs_time(
        {
            "df": first_iteration_data,
            "ax": ax1,
            "rounded_position": rounded_positions,
            "modes_between_lines": modes_between_lines,
        }
    )


    tmin = min(measured_throughput_data["time"].min(), df_est_tp["time"].min())
    tmax = max(measured_throughput_data["time"].max(), df_est_tp["time"].max())


    fig1.tight_layout(pad=0.6)
    fig1.savefig("linucb_v2_ra_vs_minstrel_rate_plot1.png", dpi=dpi, bbox_inches="tight",facecolor="white")
    plt.close(fig1)

    fig2 = plt.figure(figsize=(fig_width, total_height))
    fig2.set_size_inches(fig_width, total_height)

    gs = gridspec.GridSpec(num_subplots, 1, height_ratios=[5, 5])
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


    # for ax in (ax3, ax4):
    #     ax.set_xlim(left=tmin, right=tmax)
    #     ax.margins(x=0)
    #     ax.set_xticks(np.arange(tmin, tmax + 1, 20))

    fig2.savefig("linucb_v2_tp_plot1.png",dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    fig3 = plt.figure(figsize=(fig_width, total_height))
    fig3.set_size_inches(fig_width, total_height)

    gs1 = gridspec.GridSpec(num_subplots, 1, height_ratios=[5,5])
    ax5 = fig3.add_subplot(gs1[0])
    plot_throughput_vs_time_box_plot({
            "df": measured_throughput_data,
            "ax": ax5,
            "rounded_position": rounded_positions,
            "modes_between_lines": modes_between_lines,
            "bin_edges": bin_edges,
            "rate_x_limit": rate_x_limit,
            "bin_size": bin_size,
    })


    fig3.savefig("linucb_v2_tp_boxplot.png",dpi=dpi, bbox_inches="tight", facecolor="white")
    # plt.close(fig3)

    # fig4 = plt.figure(figsize=(fig_width, total_height))
    # fig4.set_size_inches(fig_width, total_height)

    # ax6 = fig4.add_subplot(111)

    # rate_selection_count_plot({
    #     "ax":ax6,
    #     "df": df_txs,
    #     "attenuation": 70,
    #     "mode": "linucb"
    # })

    # fig4.savefig("Rate_selection_count_at_70db.png", dpi=dpi,bbox_inches="tight", facecolor="white")
    # plt.close(fig4)





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
    ) = categorize_files(f"{args.directory}/measurement_data")

    settings_path = f"{args.directory}/settings.toml"


    attenuation_values = extract_attenuations(settings_path)
     # Cache directory inside the experiment directory
    cache_dir = os.path.join(args.directory, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    measured_throughput_cache = os.path.join(
        cache_dir, "cached_measured_throughput.pkl"
    )
    trace_response_avg_cache = os.path.join(
        cache_dir, "cached_trace_response_averaged.pkl"
    )
    trace_response_first_iter_cache = os.path.join(
        cache_dir, "cached_trace_response_first_iteration.pkl"
    )

    measured_throughput_data = load_or_process_measured_throughput(
        measured_throughput_file_dict,
        cache_path=measured_throughput_cache,
    )

    trace_response_data, first_iteration_data = load_or_process_trace_response(
        trace_response_files_dict,
        attenuation_values,
        averaged_cache_path=trace_response_avg_cache,
        first_iter_cache_path=trace_response_first_iter_cache,
    )

    plot_response_files(
        {
            "trace_response_data": trace_response_data,
            "measured_throughput_data": measured_throughput_data,
            "first_iteration_data": first_iteration_data,
        }
    )