import pandas as pd

from seiir_model.ode_model.ode_process import ODEProcess
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

    def predict_beta_forward(self, covmodel_set, df_cov, df_cov_coef, col_t, col_group, col_beta='beta_pred'):
        regressor = BetaRegressor(covmodel_set)
        regressor.load_coef(df=df_cov_coef)
        return predict(regressor, df_cov, col_t, col_group, col_beta)

    @staticmethod
    def forecast(model_specs, init_cond, times, betas,  dt=0.1):
        """
        Solves ode for given time and beta

        Arguments:
            model_specs (SiierdModelSpecs): specification for the model. See
                seiir_model.ode_forecasting.SiierdModelSpecs
                for more details.
                example:
                    model_specs = SiierdModelSpecs(
                        alpha=0.9,
                        sigma=1.0,
                        gamma1=0.3,
                        gamma2=0.4,
                        N=100,  # <- total population size
                    )

            init_cond (np.array): vector with five numbers for the initial conditions
                The order should be exactly this: [S E I1 I2 R].
                example:
                    init_cond = [96, 0, 2, 2, 0]

            times (np.array): array with times to predict for
            betas (np.array): array with betas to predict for
            dt (float): Optional, step of the solver. I left it sticking outside
                in case it works slow, so you can decrease it from the IHME pipeline.

        Returns:
            result (DataFrame):  a dataframe with columns ["S", "E", "I1", "I2", "R", "t", "beta"]
            where t and beta are times and beta which were provided, and others are solution
            of the ODE
        """
        forecaster = ODERunner(model_specs, init_cond, dt=dt)
        return forecaster.get_solution(times, betas)
