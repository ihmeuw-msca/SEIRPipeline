import pandas as pd

from seiir_model.ode_process import ODEProcess
from seiir_model.beta_fit import BetaRegressor


class ModelRunner:
    def __init__(self):
        self.ode_process = None

    def fit_beta_ode(self, ode_process_input):
        self.ode_process = ODEProcess(ode_process_input)
        self.ode_process.process()

    def get_beta_ode_fit(self):
        return self.ode_process.create_result_df()

    def fit_beta_regression(self, cov_model_set):
        pass

    def get_beta_regression_fit(self):
        pass

    def save_regression_outputs(self):
        pass

    def load_regression_outputs(self):
        pass

    def forecast(self):
        pass

    def run_ode(self):
        pass
