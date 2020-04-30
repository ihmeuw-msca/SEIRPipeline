import pandas as pd

from seiir_model.ode_model import ODEProcess
from seiir_model.regression_model import BetaRegressor, predict


class ModelRunner:
    def __init__(self):
        self.ode_process = None

    def fit_beta_ode(self, ode_process_input):
        self.ode_process = ODEProcess(ode_process_input)
        self.ode_process.process()

    def get_beta_ode_fit(self):
        return self.ode_process.create_result_df()

    def fit_beta_regression(
        self, 
        covmodel_set, 
        mr_data=None, 
        cov_coef_path=None, 
        two_stage=False,
        std=None,
    ):
        assert mr_data is not None or cov_coef_path is not None
        assert mr_data is None or cov_coef_path is None
        self.regressor = BetaRegressor(covmodel_set)
        if cov_coef_path is not None:
            self.regressor.load_coef(cov_coef_path)
        else:
            self.regressor.fit(mr_data, two_stage, std)

    def predict_beta_forward(self, df_cov, col_t, col_group, col_scenario):
        return predict(self.regressor, df_cov, col_t, col_group, col_scenario)

    def get_beta_regression_coef(self):
        return self.regressor.cov_coef

    def save_beta_regression_coef(self):
        self.regressor.save_coef()

    def load_regression_outputs(self, path):
        self.regressor.load_coef(path)

    def forecast(self):
        pass

    def run_ode(self):
        pass
