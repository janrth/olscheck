# OlsCheck

`olscheck` is a Python library designed to check and visualize the assumptions of Ordinary Least Squares (OLS). This tool streamlines the process of model diagnostics, offering an all-in-one solution for analyzing residuals, leverage, and multi-collinearity.

![Residuals vs Fitted](images/example_plot.png) 
*Sample of Residuals vs Fitted values visualization*

## Features

- **Histogram of Residuals**: Get a quick glance at the distribution of residuals.
- **Residuals vs Fitted Plot**: Visualize potential non-linear patterns.
- **QQ-Plot**: Check the normality of residuals.
- **Scale-Location Plot**: Confirm homoscedasticity.
- **Leverage Plot**: Identify influential cases.
- **VIF Test**: Assess multi-collinearity among predictors.

## Installation

```bash
pip install olscheck
