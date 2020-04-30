import pandas as pd

from seiir_model.ode_process import SingleGroupODEProcess
from seiir_model.beta_fit import BetaRegressor


class ModelRunner:
    def __init__(self):
        self.group_processes = dict()

    def fit_betas(self, column_dict, df_dict, alpha, sigma, gamma1, gamma2, solver_dt,
                  spline_options, peaked_dates_dict):

        for group, df in df_dict.keys():
            self.group_processes[group] = SingleGroupODEProcess(
                df=df,
                col_date=column_dict['COL_DATE'],
                col_cases=column_dict['COL_CASES'],
                col_pop=column_dict['COL_POP'],
                col_loc_id=column_dict['COL_LOC_ID'],
                alpha=alpha,
                sigma=sigma,
                gamma1=gamma1,
                gamma2=gamma2,
                solver_dt=solver_dt,
                spline_options=spline_options,
                peak_date=peaked_dates_dict[group]
            )
            self.group_processes[group].process()

    def get_beta_fits(self):
        all_data = []
        for group, process in self.group_processes.items():
            all_data.append(process.create_result_df())
        return pd.concat(all_data)

    def regress(self, cov_model_set):
        # fill this in with the mr data from get_beta_fits
        regressor = BetaRegressor(
            cov_model_set=cov_model_set,
            mr_data=...
        )

    def save_regression_outputs(self):
        pass

    def load_regression_outputs(self):
        pass

    def forecast(self):
        pass

    def run_ode(self):
        pass
