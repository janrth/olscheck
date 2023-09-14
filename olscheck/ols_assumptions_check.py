import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from typing import List


class OlsCheck:
    """
    Checks all relevant assumptions of OLS for pooled regression.
    """

    def __init__(self):
        pass

    def _compute_residuals(self, df: pd.DataFrame, y_true_col: str, fittedvalues_col: str) -> pd.DataFrame:
        """
        Compute residuals and add them as a column to the DataFrame.
        """
        df['residuals'] = df[y_true_col] - df[fittedvalues_col]
        return df

    def _cooks_dist_line(self, factor: float, features: List[str], leverage: pd.Series) -> pd.Series:
        """
        Helper function for plotting Cook's distance curves.
        """
        n = len(features)
        formula = lambda x: np.sqrt((factor * n * (1 - x)) / x)
        x = np.linspace(0.001, max(leverage), 50)
        y = formula(x)
        return x, y

    def _cooks_distance(self, df:pd.DataFrame, leverage: pd.Series, features: List[str], fittedvalues: pd.Series, y_true: pd.Series) -> pd.Series:
        """
        Helper function to calculate Cook's distance.
        """
        residuals = df['residuals']
        n = len(residuals)
        p = len(features)
        mse = np.dot(residuals, residuals) / (n - p)
        cooks_distance = ((residuals) ** 2 / (p * mse)) * (leverage / (1 - leverage) ** 2)
        return cooks_distance

    def _residuals_studentized_internal(self, df: pd.DataFrame, features: List[str], constant=True) -> (np.ndarray, np.ndarray):
        # QR decomposition for leverage calculation
        if constant:
            df = add_constant(df)
            X = df[['const']+features]
        else:
            X = df[features].values
        residuals = df['residuals']
        Q, R = np.linalg.qr(X)
        H = Q @ Q.T
        leverage = np.diagonal(H)
        
        # Compute normalized residuals
        s = np.std(residuals, ddof=1) # using Bessel's correction for sample standard deviation
        res_studentized_internal = residuals / s
        return leverage, res_studentized_internal

    def vif_test(self, df_vif: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the variance inflation factor for all columns in the specified df.
        """
        df_vif = df_vif.dropna()
        df_vif = add_constant(df_vif)
        vif = pd.DataFrame()
        vif["variables"] = df_vif.columns
        vif["vif"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
        return vif.sort_values('vif', ascending=False).reset_index(drop=True)

    def leverage_plot(self, df: pd.DataFrame, features:List[str], fittedvalues_col: str, y_true_col: str, axis_lim=True, constant=True, ax=None):
        """
        Residual vs Leverage plot

        Points falling outside Cook's distance curves are considered observation that can sway the fit
        aka are influential.
        Good to have none outside the curves.
        """
        residuals = df['residuals']
        y_true = df[y_true_col]
        fittedvalues = df[fittedvalues_col]
        
        leverage, res_studentized_internal = self._residuals_studentized_internal(df, features, constant)
        cooks_distance = self._cooks_distance(df, leverage, features, fittedvalues, y_true)
        
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(
            leverage,
            res_studentized_internal,
            alpha=0.5);

        sns.regplot(
            x=leverage,
            y=res_studentized_internal,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        leverage_top_3 = np.flip(np.argsort(cooks_distance), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(
                i,
                xy=(leverage[i], res_studentized_internal[i]),
                color = 'C3')

        xtemp, ytemp = self._cooks_dist_line(0.5, features, leverage) # 0.5 line
        ax.plot(xtemp, ytemp, label="Cook's distance", lw=1, ls='--', color='red')
        xtemp, ytemp = self._cooks_dist_line(1, features, leverage) # 1 line
        ax.plot(xtemp, ytemp, lw=1, ls='--', color='red')

        ax.axhline(0, ls='dotted', color='black', lw=1.25)
        ax.set_xlim(0, max(leverage)+0.01)
        if axis_lim:
            ax.set_ylim(min(res_studentized_internal)-0.1, max(res_studentized_internal)+0.1) 
        ax.set_title('Residuals vs Leverage', fontweight="bold")
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        ax.legend(loc='upper right')
        return res_studentized_internal

    def histogram_residuals(self, df: pd.DataFrame, ax=None):
        """
        Plots a histogram of residuals with +/- 1 and 2 standard deviations.
        """
        residuals = df['residuals']
        if ax is None:
            fig, ax = plt.subplots()
        sns.histplot(residuals, ax=ax)
        ax.set_title('Residuals', fontweight="bold")
        return ax

    def residual_plot(self, df: pd.DataFrame, fittedvalues_cols:str, ax=None):
        """
        Plots the fitted values vs residuals.
        """
        residuals = df['residuals']
        fittedvalues = df[fittedvalues_cols]
        if ax is None:
            fig, ax = plt.subplots()
        sns.residplot(x=fittedvalues, y=residuals, lowess=True,
                      scatter_kws={'alpha': 0.5},
                      line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
                      ax=ax)
        ax.set_title('Residuals vs Fitted', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        return ax

    def scale_location_plot(self, df: pd.DataFrame, fittedvalues_cols:str, features: List[str], ax=None):
        """
        Sqrt(Standardized Residual) vs Fitted values plot.
        """
        residuals = df['residuals']
        fittedvalues = df[fittedvalues_cols]
        leverage, res_studentized_internal = self._residuals_studentized_internal(df, features)
        if ax is None:
            fig, ax = plt.subplots()
        residual_norm_abs_sqrt = np.sqrt(np.abs(res_studentized_internal))
        ax.scatter(fittedvalues, residual_norm_abs_sqrt, alpha=0.5)
        sns.regplot(x=fittedvalues, y=residual_norm_abs_sqrt, scatter=False, ci=False, lowess=True, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}, ax=ax)
        ax.set_title('Scale-Location', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$')
        return ax

    def qq_plot(self, df: pd.DataFrame, features: List[str], ax=None):
        """
        Standarized Residual vs Theoretical Quantile plot.
        """
        residuals = df['residuals']
        leverage, res_studentized_internal = self._residuals_studentized_internal(df, features)
        if ax is None:
            fig, ax = plt.subplots()
        QQ = ProbPlot(res_studentized_internal)
        QQ.qqplot(line='45', alpha=0.5, lw=1, ax=ax)
        ax.set_title('Normal Q-Q', fontweight="bold")
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')
        return ax

    def ols_assumptions(self, df: pd.DataFrame, y_true_col: str, fittedvalues_col: str, features: List[str], plot_context='seaborn-paper', axis_lim=True, constant=True):
        """
        Visualize OLS assumptions.
        """
        df = self._compute_residuals(df, y_true_col, fittedvalues_col)

        # Display VIF results
        print("VIF Results:")
        vif_results = self.vif_test(df[features])
        print(vif_results)
        
        with plt.style.context(plot_context):
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))      
            self.residual_plot(df, fittedvalues_col, ax=ax[0, 0])
            self.qq_plot(df, features, ax=ax[0, 1])
            self.scale_location_plot(df, fittedvalues_col, features, ax=ax[1, 0])
            self.leverage_plot(df, features, fittedvalues_col, y_true_col, axis_lim, constant, ax=ax[1,1])
            plt.show()

