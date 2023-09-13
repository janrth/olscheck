import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegressionDataGenerator:
    def __init__(self, n_points=1000, slope=2, intercept=5, noise_std=1.5):
        self.n_points = n_points
        self.slope = slope
        self.intercept = intercept
        self.noise_std = noise_std
        #np.random.seed(42)

    def generate_data(self):
        # Generate X1 values
        X1 = np.linspace(0, 10, self.n_points)
        
        # Generate X2 values as a function of X1
        noise_for_X2 = np.random.normal(0, self.noise_std, self.n_points)
        X2 = 2 * X1 + noise_for_X2
    
        # Simulate a linear relation with added noise using both X1 and X2
        noise_for_y = np.random.normal(0, self.noise_std, self.n_points)
        y = self.slope * X1 + 1.5 * X2 + self.intercept + noise_for_y
    
        # Store data in a pandas DataFrame
        df = pd.DataFrame({
            'X1': X1,
            'X2': X2,
            'y': y
        })
        return df
      





