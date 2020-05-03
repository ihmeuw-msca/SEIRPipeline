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
OUTPUT_DRAWS_CASES = "output_draws_cases"
OUTPUT_DRAWS_DEATHS = "output_draws_deaths"
OUTPUT_DRAWS_REFF = "output_draws_reff"


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
            ODE_COMPONENTS_FORECAST: [],
            OUTPUT_DRAWS_CASES: None,
            OUTPUT_DRAWS_DEATHS: None,
            OUTPUT_DRAWS_REFF: None

        } for group in self.groups}
        self.params_for_draws = []

        #self.metadata = pd.read_csv("../../../data/covid/metadata-inputs/location_metadata_652.csv")
        # TODO: change it for cluster
        self.metadata = pd.read_csv(directories.get_location_metadata_file(location_set_version_id=652))

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

        #  read final draws
        if os.path.isdir(directories.forecast_output_draw_dir):
            for group in groups:
                self.data[group][OUTPUT_DRAWS_CASES] = pd.read_csv(os.path.join(directories.forecast_output_draw_dir, f"cases_{group}.csv"))
                self.data[group][OUTPUT_DRAWS_DEATHS] = pd.read_csv(os.path.join(directories.forecast_output_draw_dir, f"deaths_{group}.csv"))
                self.data[group][OUTPUT_DRAWS_REFF] = pd.read_csv(os.path.join(directories.forecast_output_draw_dir, f"reff_{group}.csv"))

    def format_x_axis(self, ax, start_date, now_date, end_date,
                      major_tick_interval_days=7, margins_days=5):

        months = mdates.DayLocator(interval=major_tick_interval_days)
        days = mdates.DayLocator()  # Every day
        months_fmt = mdates.DateFormatter('%m/%d')

        # format the ticks
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(months_fmt)
        ax.xaxis.set_minor_locator(days)

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

        for i, (past_compartments, future_compartments) in enumerate(zip(self.data[group][ODE_BETA_FIT],
                                                                         self.data[group][ODE_COMPONENTS_FORECAST])):
            if draws is not None:
                if i not in draws:
                    continue

            past_time = pd.to_datetime(past_compartments[self.col_date])
            past_compartment_trajectory = past_compartments[compartment]
            ax.plot(past_time, past_compartment_trajectory,
                    linestyle=linestyle, c=color, alpha=transparency)
            future_time = pd.to_datetime(future_compartments[self.col_date])
            future_compartment_trajectory = future_compartments[compartment]
            ax.plot(future_time, future_compartment_trajectory,
                    linestyle=linestyle, c=color, alpha=transparency)

        # get times
        past_time = pd.to_datetime(self.data[group][ODE_BETA_FIT][0][self.col_date])
        future_time = pd.to_datetime(self.data[group][ODE_COMPONENTS_FORECAST][0][self.col_date])
        start_date = past_time.to_list()[0]
        now_date = past_time.to_list()[-1]
        end_date = future_time.to_list()[-1]

        visualizer.format_x_axis(ax, start_date, now_date, end_date, major_tick_interval_days=14)

    def create_trajectories_plot(self,
                                 group,
                                 output_dir="plots",
                                 compartments = ('S', 'E', 'I1', 'I2', 'R', 'beta'),
                                 colors = ('blue', 'orange', 'red', 'purple', 'green', 'blue')):
        group_name = self.metadata[self.metadata['location_id'] == group]['location_name'].to_list()[0]
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
            ax.grid(True)
            # ax.legend(loc="upper left")
            ax.set_title(f"Location {group_name}: {compartment}")
        print(f"Trajectories plot for {group} {group_name} is done")

        plt.savefig(os.path.join(output_dir, f"trajectories_{group_name}.pdf"))
        plt.close(fig)

    def create_final_draws_plot(self,
                                group,
                                compartments=('Cases', 'Deaths', 'R_effective'),
                                output_dir = "plots",
                                linestyle="solid",
                                transparency=0.1,
                                color=('orange', 'red', 'blue')):
        compartment_to_col = {
            'Cases': OUTPUT_DRAWS_CASES,
            'Deaths': OUTPUT_DRAWS_DEATHS,
            'R_effective': OUTPUT_DRAWS_REFF
        }
        group_name = self.metadata[self.metadata['location_id'] == group]['location_name'].to_list()[0]
        fig = plt.figure(figsize=(12, (3) * 6))
        grid = plt.GridSpec(3, 1, wspace=0.1, hspace=0.4)
        fig.autofmt_xdate()

        for i, compartment in enumerate(compartments):
            compartment_data = self.data[group][compartment_to_col[compartment]]
            ax = fig.add_subplot(grid[i, 0])
            time = pd.to_datetime(compartment_data[self.col_date])
            start_date = time.to_list()[0]
            end_date = time.to_list()[-1]
            now_date = pd.to_datetime(compartment_data[compartment_data['observed']== 1][self.col_date]).to_list()[-1]
            draw_num = 0
            while f"draw_{draw_num}" in compartment_data.columns:
                draw_name = f"draw_{draw_num}"
                if compartment == "R_effective2":
                    ax.semilogy(time, compartment_data[draw_name], linestyle=linestyle, c=color[i], alpha=transparency)
                else:
                    ax.plot(time, compartment_data[draw_name], linestyle=linestyle, c=color[i], alpha=transparency)
                draw_num += 1
            self.format_x_axis(ax, start_date=start_date, now_date=now_date, end_date=end_date, major_tick_interval_days=14)
            ax.set_title(f"{group_name}: {compartment}")

        print(f"Final draws plot for {group} {group_name} is done")

        plt.savefig(os.path.join(output_dir, f"final_draws_rlinear_{group_name}.pdf"))
        plt.close(fig)

if __name__ == "__main__":
    col_date = "date"
    col_group = "loc_id"
    all_groups = [102, 524, 526, 528,530,532,534,536,538,540,542,544,546,548,550,552,554,556,558,560,562,564,566,568,570,572]
    all_groups += [523,525,527,529,531,533,535,537,539,541,543,545,547,549,551,553,555,557,559,561,563,565,567,569,571,573]
    #groups = all_groups
    groups = [526,528,530,534,572]
    #groups = [524]
    version = "2020_05_03.03"


    directories = Directories(regression_version=version, forecast_version=version)
    visualizer = Visualizer(directories, groups=groups, col_date=col_date, col_group=col_group)

    for group in groups:
        visualizer.create_trajectories_plot(group=group,
                                            # TODO: change when add plotting dirs to the directories object
                                            # output_dir = directories.get_trajectories_plot_dir
                                            output_dir=".")
        visualizer.create_final_draws_plot(group=group,
                                           # TODO: Same
                                           output_dir=".")




