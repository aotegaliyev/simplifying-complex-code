import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import time
import warnings

warnings.filterwarnings('ignore')

class CurvePrediction:
    def __init__(self, x, y, step=3):
        self.x = np.array(x)
        self.y = np.array(y)
        self.step = step
        self.predicted_x = None
        self.predicted_y = None

    def linear(self, x, a, b):
        return a * x + b

    def sigmoid(self, x, a, b):
        return 1 / (1 + np.exp(-a * (x - b)))

    def extended_sigmoid(self, x, a, b, c, d):
        return a + b / (1 + np.exp(-c * (x - d)))

    def exponential(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def parabola(self, x, a, b, c):
        return a * x**2 + b * x + c

    def tanh(self, x, a, b):
        return a * np.tanh(x) + b

    def log(self, x, a, b):
        return a * np.log(x) + b

    def predict_y(self, func, x_norm, y_norm, real_y, p0=None, maxfev=500000, step=3):
        popt, _ = curve_fit(func, x_norm[:-step], y_norm, p0=p0, maxfev=maxfev)
        predicted_y = func(x_norm, *popt) * max(real_y)
        mse = mean_squared_error(real_y, predicted_y[:-step])
        return predicted_y, mse

    @staticmethod
    def timing(func):
        def wrap(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f'{end_time - start_time:.2f} sec')
            return result
        return wrap

    @timing
    def predict(self, curve_name=None):
        x = self.x
        y = self.y
        step = self.step

        # Normalize x, y
        new_x = np.append(x, [x[-1] + i for i in range(1, step + 1)])
        x_norm = new_x / max(new_x)
        y_norm = y / max(y)
        predicted_x = new_x
        print(x_norm, y_norm, y, step, sep='\n')
        # Curve prediction logic
        predictions = []
        if curve_name is None:
            predictions.append(self.predict_y(self.linear, x_norm, y_norm, y, step=step))
            predictions.append(self.predict_y(self.sigmoid, x_norm, y_norm, y, step=step))
            predictions.append(self.predict_y(self.extended_sigmoid, x_norm, y_norm, y, step=step))
            predictions.append(self.predict_y(self.exponential, x_norm, y_norm, y, p0=[1, 0, 1], step=step))
            predictions.append(self.predict_y(self.parabola, x_norm, y_norm, y, step=step))
            predictions.append(self.predict_y(self.tanh, x_norm, y_norm, y, step=step))
            predictions.append(self.predict_y(self.log, x_norm, y_norm, y, step=step))
        else:
            curve_func = getattr(self, curve_name)
            predictions.append(self.predict_y(curve_func, x_norm, y_norm, y, step=step))

        mean_sq_errors = [p[1] for p in predictions]
        min_index = mean_sq_errors.index(min(mean_sq_errors))
        self.predicted_y = predictions[min_index][0]
        self.predicted_x = predicted_x

    def plot(self):
        real_x = self.x
        real_y = self.y
        predicted_x = self.predicted_x
        predicted_y = self.predicted_y

        plt.figure(figsize=(13, 7))
        y_avg = sum(predicted_y) / (2 * len(predicted_y))
        x_avg = sum(predicted_x) / (2 * len(predicted_x))

        plt.axis(xmin=min(predicted_x) - x_avg / 3,
                 xmax=max(predicted_x) + x_avg,
                 ymin=min(predicted_y) - y_avg / 3,
                 ymax=max(predicted_y) + y_avg)

        plt.plot(real_x, real_y, 'm', label="Real values", linewidth=2)
        plt.plot(predicted_x, predicted_y, 'b', label="Predicted values", linewidth=2)

        plt.xlabel('Time', fontsize=16)
        plt.ylabel('Value', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid()
        plt.show()


# Example usage:
x = [i for i in range(1, 6)]
y = [100000, 120000, 200000, 250000, 270000]

curve_prediction = CurvePrediction(x, y, step=3)
curve_prediction.predict(curve_name='exponential')
curve_prediction.plot()
