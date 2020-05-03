import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from seiir_model_pipeline.core.versioner import Directories
from seiir_model_pipeline.core.versioner import load_regression_settings
from seiir_model_pipeline.core.versioner import load_forecast_settings

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


class PlotBetaCoef:
    def __init__(self,
                 directory: Directories,
                 location_set_version_id):
        self.directory = directory

        self.path_to_location_metadata = self.directory.get_location_metadata_file(
            location_set_version_id)
        self.path_to_coef_dir = self.directory.regression_coefficient_dir
        self.path_to_savefig = self.directory.regression_diagnostic_dir

        # load settings
        self.settings = load_regression_settings(directory.regression_version)

        # load metadata
        self.location_metadata = pd.read_csv(self.path_to_location_metadata)
        self.id2loc = self.location_metadata.set_index('location_id')[
            'location_name'].to_dict()

        # load coef
        df_coef = [
            pd.read_csv(
                '/'.join([self.path_to_coef_dir, f'coefficients_{i}.csv']))
            for i in range(self.settings.n_draws)
        ]

        # organize information
        self.covs = np.sort(list(self.settings.covariates.keys()))
        self.loc_ids = np.sort(list(df_coef[0]['group_id'].unique()))
        self.locs = np.array([
            self.id2loc[loc_id]
            for loc_id in self.loc_ids
        ])
        self.num_locs = len(self.locs)

        # group coef data
        self.coef_data = {}
        for cov in self.covs:
            coef_mat = np.vstack([
                df[cov].values
                for df in df_coef
            ])
            coef_label = self.locs.copy()
            coef_mean = coef_mat.mean(axis=0)
            sort_idx = np.argsort(coef_mean)
            self.coef_data[cov] = (coef_label[sort_idx], coef_mat[:, sort_idx])

    def plot_coef(self):
        for cov in self.covs:
            plt.figure(figsize=(8, 15))
            plt.boxplot(self.coef_data[cov][1], vert=False, showfliers=False,
                        boxprops=dict(linewidth=0.5),
                        whiskerprops=dict(linewidth=0.5))
            plt.yticks(ticks=np.arange(self.num_locs) + 1,
                       labels=self.coef_data[cov][0])
            coef_mean = self.coef_data[cov][1].mean()
            plt.vlines(coef_mean, ymin=1, ymax=self.num_locs,
                       linewidth=1.0, linestyle='--', color='#003152')
            #             for b in self.settings['covariates'][cov]['bounds']:
            #                 if np.abs(b) >= np.abs(coef_mean)*2:
            #                     continue
            #                 plt.vlines(b, ymin=1, ymax=self.num_locs,
            #                            linewidth=1.0, linestyle='-', color='#8B0000')
            plt.grid(b=True)
            plt.box(on=None)
            plt.title(cov)
            plt.savefig('/'.join([self.path_to_savefig,
                                  f'{cov}_boxplot.pdf']), bbox_inches='tight')


class PlotBetaResidual:
    def __init__(self,
                 directory: Directories,
                 location_set_version_id):
        self.directory = directory
        self.path_to_location_metadata = self.directory.get_location_metadata_file(
            location_set_version_id)
        self.path_to_betas_dir = self.directory.regression_beta_fit_dir
        self.path_to_savefig = self.directory.regression_diagnostic_dir

        # load settings
        self.settings = load_regression_settings(directory.regression_version)

        # load location metadata
        self.location_metadata = pd.read_csv(self.path_to_location_metadata)
        self.id2loc = self.location_metadata.set_index('location_id')[
            'location_name'].to_dict()

        # load the beta data
        df_beta = [
            pd.read_csv(
                '/'.join([self.path_to_betas_dir, f'fit_draw_{i}.csv']))[[
                'loc_id',
                'date',
                'days',
                'beta',
                'beta_pred'
            ]].dropna()
            for i in range(self.settings.n_draws)
        ]

        # location information
        self.loc_ids = np.sort(list(df_beta[0]['loc_id'].unique()))
        self.locs = np.array([
            self.id2loc[loc_id]
            for loc_id in self.loc_ids
        ])
        self.num_locs = len(self.locs)

        # compute RMSE
        self.rmse_data = np.vstack([
            np.array([self.get_rsme(df, loc_id) for loc_id in self.loc_ids])
            for df in df_beta
        ])

    def get_rsme(self, df, loc_id):
        beta = df.loc[df.loc_id == loc_id, 'beta'].values
        pred_beta = df.loc[df.loc_id == loc_id, 'beta_pred'].values

        return np.sqrt(np.mean((beta - pred_beta)**2))

    def plot_residual(self):
        fig, ax = plt.subplots(self.num_locs, 1, figsize=(8, 4*self.num_locs))
        for i, loc in enumerate(self.locs):
            ax[i].hist(self.rmse_data[:, i])
            ax[i].set_title(loc)
        plt.savefig('/'.join([self.path_to_savefig,
                              f'residual_rmse_histo.pdf']), bbox_inches='tight')

        plt.figure(figsize=(8, 15))
        sort_idx = np.argsort(self.rmse_data.mean(axis=0))
        plt.boxplot(self.rmse_data[:, sort_idx], vert=False, showfliers=False,
                    boxprops=dict(linewidth=0.5),
                    whiskerprops=dict(linewidth=0.5))
        plt.grid(b=True)
        plt.box(on=None)
        plt.yticks(ticks=np.arange(self.num_locs) + 1,
                   labels=self.locs[sort_idx])
        plt.title('Beta Regression Residual RMSE')
        plt.savefig('/'.join([self.path_to_savefig,
                              f'residual_rmse_boxplot.pdf']),
                    bbox_inches='tight')


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



