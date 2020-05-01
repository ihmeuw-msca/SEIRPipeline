import pandas as pd

from seiir_model.ode_model import ODEProcess
from seiir_model.regression_model.beta_fit import BetaRegressor, predict
from seiir_model.ode_forecasting import ODERunner


class ModelRunner:
    def __init__(self):
        self.ode_model = None

    def fit_beta_ode(self, ode_process_input):
        self.ode_model = ODEProcess(ode_process_input)
        self.ode_model.process()

    def get_beta_ode_fit(self, path=None):
        if self.ode_model is None:
            assert path is not None, 'Must fit_beta_ode or provide the path ' \
                                     'to the fit result.'
            return pd.read_csv(path)
        else:
            return self.ode_model.create_result_df()

    def get_beta_ode_params(self, path=None):
        if self.ode_model is None:
            assert path is not None, 'Must fit_beta_ode or provide the path ' \
                                     'to the fit parameters.'
            return pd.read_csv(path)
        else:
            return self.ode_model.create_params_df()

    def save_beta_ode_result(self, fit_file, params_file):
        """Save result from beta ode fit.

        Args:
            fit_file (str): fit file path to save to
            params_file (str): params file to save to
        """
        assert self.ode_model is not None, 'Must fit_beta_ode first.'
        # save ode fit
        self.get_beta_ode_fit().to_csv(fit_file, index=False)
        # save other parameters
        self.get_beta_ode_params().to_csv(params_file, index=False)

    def fit_beta_regression(self, covmodel_set, mr_data, path, two_stage=False,std=None):
        regressor = BetaRegressor(covmodel_set)
        regressor.fit(mr_data, two_stage, std)
        regressor.save_coef(path)

    def predict_beta_forward(self, covmodel_set, df_cov, df_cov_coef, col_t, col_group):
        regressor = BetaRegressor(covmodel_set)
        regressor.load_coef(df=df_cov_coef)
        return predict(regressor, df_cov, col_t, col_group)

    @staticmethod
    def forecast(df, col_t, col_beta, model_specs, init_cond, dt=0.1):
        times = df[col_t].to_numpy()
        beta = df[col_beta].to_numpy()
        forecaster = ODERunner(model_specs, init_cond, dt=dt)
        return forecaster.get_solution(times, beta)
