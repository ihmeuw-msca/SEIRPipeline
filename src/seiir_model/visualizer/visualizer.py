import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from seiir_model.visualizer.versioner import Directories

ODE_BETA_FIT = "ode_beta_fit"
COEFFICIENTS_FIT = "coefficients_fit"
PARAMETERS_FIT = "parameters_fit"
ODE_COMPONENTS_FORECAST = "ode_forecast"


class Visualizer:

    def __init__(self, directories: Directories,
                 groups: list = None, exclude_groups: list = None,
                 col_group="loc_id", col_date='date'
                 ):
        self.directories = directories
        self.col_group = col_group
        self.col_date = col_date
        self.groups = groups
        if exclude_groups is not None:
            for exclude_group in exclude_groups:
                self.groups.remove(exclude_group)
        self.data = {group: {
            ODE_BETA_FIT: [],
            COEFFICIENTS_FIT: [],
            ODE_COMPONENTS_FORECAST: []
        } for group in self.groups}
        self.params_for_draws = []

        # read beta regression draws
        for filename in os.listdir(directories.regression_beta_fit_dir):
            if filename.startswith("fit_draw_") and filename.endswith(".csv"):
                draw_df = pd.read_csv(os.path.join(directories.regression_beta_fit_dir, filename))
                for group in self.groups:
                    self.data[group][ODE_BETA_FIT].append(draw_df[draw_df[col_group] == group])
            else:
                continue

        # read coefficients draws
        for filename in os.listdir(directories.regression_coefficient_dir):
            if filename.startswith("coefficients_") and filename.endswith(".csv"):
                draw_df = pd.read_csv(os.path.join(directories.regression_coefficient_dir, filename))
                for group in self.groups:
                    self.data[group][COEFFICIENTS_FIT].append(draw_df[draw_df['group_id'] == group])
            else:
                continue

        # read params draws
        for filename in os.listdir(directories.regression_parameters_dir):
            if filename.startswith("params_draw_") and filename.endswith(".csv"):
                draw_df = pd.read_csv(os.path.join(directories.regression_parameters_dir, filename))
                self.params_for_draws.append(draw_df)
            else:
                continue

        # read components forecast
        for group in groups:
            path_to_compartments_draws_for_group = os.path.join(directories.forecast_component_draw_dir, str(group))
            if os.path.isdir(path_to_compartments_draws_for_group):
                for filename in os.listdir(path_to_compartments_draws_for_group):
                    if filename.startswith("draw_") and filename.endswith(".csv"):
                        draw_df = pd.read_csv(os.path.join(path_to_compartments_draws_for_group, filename))
                        self.data[group][ODE_COMPONENTS_FORECAST].append(draw_df)
                    else:
                        continue
            else:
                error_msg = f"ODE Components forecast for the group with {col_group} = {group} is not found"
                print("Error: " + error_msg)
                # raise FileNotFoundError(error_msg)

        # TODO: read final draws when the data is ready

    def format_x_axis(self, ax, group,
                      major_tick_interval_days=7, margins_days=5):

        months = mdates.DayLocator(interval=major_tick_interval_days)
        days = mdates.DayLocator()  # Every day
        months_fmt = mdates.DateFormatter('%m/%d')

        # format the ticks
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(months_fmt)
        ax.xaxis.set_minor_locator(days)

        # get times
        past_time = pd.to_datetime(self.data[group][ODE_BETA_FIT][0][self.col_date])
        future_time = pd.to_datetime(self.data[group][ODE_COMPONENTS_FORECAST][0][self.col_date])
        start_date = past_time.to_list()[0]
        now_date = past_time.to_list()[-1]
        end_date = future_time.to_list()[-1]

        # round to nearest years.
        datemin = np.datetime64(start_date, 'D') - np.timedelta64(margins_days, 'D')
        datemax = np.datetime64(end_date, 'D') + np.timedelta64(margins_days, 'D')
        ax.set_xlim(datemin, datemax)

        # format the coords message box
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

        if now_date is not None:
            now_date = np.datetime64(now_date, 'D')
            ylims = ax.get_ylim()
            ax.plot([now_date, now_date], ylims, linestyle="dashed", c='black')
            middle_level = np.mean(ylims)
            ax.text(now_date - np.timedelta64(8, 'D'), middle_level, "Past")
            ax.text(now_date + np.timedelta64(2, 'D'), middle_level, "Future")

    def plot_ode_compartment(self, group, ax,
                             compartment="I1",
                             linestyle='solid',
                             transparency=0.2,
                             color='orange',
                             draws=None
                             ):

        past_time = None
        future_time = None
        mean_trajectory_past = 0
        mean_trajectory_future = 0
        for i, (past_compartments, future_compartments) in enumerate(zip(self.data[group][ODE_BETA_FIT],
                                                                         self.data[group][ODE_COMPONENTS_FORECAST])):
            if draws is not None:
                if i not in draws:
                    continue

            past_time = pd.to_datetime(past_compartments[self.col_date])
            past_compartment_trajectory = past_compartments[compartment]
            ax.plot(past_time, past_compartment_trajectory,
                    linestyle=linestyle, c=color, alpha=transparency)
            #mean_trajectory_past += past_compartment_trajectory.to_numpy()
            future_time = pd.to_datetime(future_compartments[self.col_date])
            future_compartment_trajectory = future_compartments[compartment]
            ax.plot(future_time, future_compartment_trajectory,
                    linestyle=linestyle, c=color, alpha=transparency)
            #mean_trajectory_future += future_compartment_trajectory.to_numpy()

        # TODO: Uncomment this to plot the mean trajectory when all
        # if past_time is not None and future_time is not None:
        #     ax.plot(np.concatenate((past_time, future_time)),
        #             np.concatenate((mean_trajectory_past, mean_trajectory_future)),
        #             linestyle=linestyle, c=color, alpha=1, label=compartment
        #             )


if __name__ == "__main__":
    col_date = "date"
    col_group = "loc_id"
    groups = [102, 524, 532]
    #groups = [524]
    directories = Directories(regression_version="2020_05_02.01", forecast_version="2020_05_02.01")
    visualizer = Visualizer(directories, groups=groups, col_date=col_date, col_group=col_group)
    compartments = ('S', 'E', 'I1', 'I2', 'R', 'beta')
    #compartments = ('S', 'E')
    colors = ('blue', 'orange', 'red', 'purple', 'green', 'blue')

    for group in groups:
        fig = plt.figure(figsize=(12, (len(compartments)+1) * 6))
        grid = plt.GridSpec(len(compartments) + 1, 1, wspace=0.1, hspace=0.4)
        fig.autofmt_xdate()
        for i, compartment in enumerate(compartments):
            ax = fig.add_subplot(grid[i, 0])
            visualizer.plot_ode_compartment(group=group, ax=ax,
                                            compartment=compartment,
                                            linestyle="solid",
                                            transparency=0.1,
                                            color=colors[i],
                                            )
            visualizer.format_x_axis(ax, group, major_tick_interval_days=14)

            ax.grid(True)
            ax.legend(loc="upper left")
            ax.set_title(f"Location {group}: {compartment}")

        plt.savefig(f"trajectories_{group}.pdf")



