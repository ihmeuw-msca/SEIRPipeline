import pandas as pd

from seiir_model.ode_model import ODEProcess
from seiir_model.regression_model.beta_fit import BetaRegressor, predict
from seiir_model.ode_forecasting import SiierdModelSpecs, ODERunner

class ModelRunner:
    def __init__(self):
        self.ode_process = None

    def fit_beta_ode(self, ode_process_input):
        self.ode_process = ODEProcess(ode_process_input)
        self.ode_process.process()

    def get_beta_ode_fit(self):
        return self.ode_process.create_result_df()

    def fit_beta_regression(self, covmodel_set, mr_data, path, two_stage=False,std=None):
        regressor = BetaRegressor(covmodel_set)
        regressor.fit(mr_data, two_stage, std)
        regressor.save_coef(path)

    def predict_beta_forward(self, covmodel_set, df_cov, df_cov_coef, col_t, col_group, col_scenario):
        regressor = BetaRegressor(covmodel_set)
        regressor.load_coef(df_cov_coef)
        return predict(regressor, df_cov, col_t, col_group, col_scenario)

    def forecast(self, df, col_t, col_beta, model_specs, init_cond, dt=0.1):
        times = df[col_t].to_numpy()
        beta = df[col_beta].to_numpy()
        forecaster = ODERunner(model_specs, init_cond, dt=dt)
        return forecaster.get_solution(times, beta)

    def run_ode(self):
        pass
