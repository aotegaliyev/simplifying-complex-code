from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


class CurveStrategy(ABC):
    def __init__(self, p0: list[float] | None = None) -> None:
        self.p0 = p0

    @abstractmethod
    def fit_curve(self, x: np.ndarray, *params: float) -> callable:
        """Abstract method for the curve function to be implemented by subclasses."""
        pass


class LinearCurveStrategy(CurveStrategy):
    def fit_curve(self, x: np.ndarray, a: float, b: float) -> callable:
        return a * x + b


class SigmoidCurveStrategy(CurveStrategy):
    def fit_curve(self, x: np.ndarray, a: float, b: float) -> callable:
        return 1 / (1 + np.exp(-a * (x - b)))


class ExponentialCurveStrategy(CurveStrategy):
    def __init__(self) -> None:
        super().__init__(p0=[1, 0, 1])

    def fit_curve(self, x: np.ndarray, a: float, b: float, c: float) -> callable:
        return a * np.exp(-b * x) + c


class ParabolaCurveStrategy(CurveStrategy):
    def fit_curve(self, x: np.ndarray, a: float, b: float, c: float) -> callable:
        return a * x**2 + b * x + c


class ExtendedSigmoidCurveStrategy(CurveStrategy):
    def fit_curve(self, x: np.ndarray, a: float, b: float, c: float, d: float) -> callable:
        return a + b / (1 + np.exp(-c * (x - d)))


class TanhCurveStrategy(CurveStrategy):
    def fit_curve(self, x: np.ndarray, a: float, b: float) -> callable:
        return a * np.tanh(x) + b


class LogCurveStrategy(CurveStrategy):
    def fit_curve(self, x: np.ndarray, a: float, b: float) -> callable:
        return a * np.log(x) + b


class CurveFittingService:
    def __init__(self, strategy: CurveStrategy) -> None:
        self.strategy = strategy

    def fit(self, x: np.ndarray, y: np.ndarray, step: int = 3) -> tuple[np.ndarray, float]:
        popt, _ = curve_fit(self.strategy.fit_curve, x[:-step], y, p0=self.strategy.p0, maxfev=500000)
        predicted_y = self.strategy.fit_curve(x, *popt) * max(y)
        mse = mean_squared_error(y, predicted_y[:-step])
        return predicted_y, mse


class PlottingService:
    def plot(self, real_x: np.ndarray, real_y: np.ndarray, predicted_x: np.ndarray, predicted_y: np.ndarray) -> None:
        plt.figure(figsize=(13, 7))
        y_avg = sum(predicted_y) / (2 * len(predicted_y))
        x_avg = sum(predicted_x) / (2 * len(predicted_x))

        plt.axis(xmin=min(predicted_x) - x_avg / 3,
                 xmax=max(predicted_x) + x_avg,
                 ymin=min(predicted_y) - y_avg / 3,
                 ymax=max(predicted_y) + y_avg)

        plt.plot(real_x, real_y, 'm', label="Real values", linewidth=2)
        plt.plot(predicted_x, predicted_y, 'b', label="Predicted values", linewidth=2)

        plt.ylabel('Value', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid()
        plt.show()


class DataPredictionFacade:
    def __init__(self, x: list[float], y: list[float], strategy: CurveStrategy | None = None, step: int = 3) -> None:
        self.x = np.array(x)
        self.y = np.array(y)
        self.step: int = step
        self.predicted_x = np.concatenate((self.x, self.x[-1] + np.arange(1, step + 1)))
        self.predicted_y = None
        self.strategy = strategy
        self.strategies = [
            LinearCurveStrategy(),
            SigmoidCurveStrategy(),
            ExtendedSigmoidCurveStrategy(),
            ExponentialCurveStrategy(),
            ParabolaCurveStrategy(),
            TanhCurveStrategy(),
            LogCurveStrategy(),
        ]

    def predict_and_plot(self) -> None:
        x_norm = self.predicted_x / max(self.predicted_x)
        y_norm = self.y / max(self.y)

        best_mse = float('inf')
        best_prediction = None

        if self.strategy is None:
            # Loop through all strategies and find the best one
            for strategy in self.strategies:
                fitting_service = CurveFittingService(strategy)
                predicted_y, mse = fitting_service.fit(x_norm, y_norm, step=self.step)

                if mse < best_mse:
                    best_mse = mse
                    best_prediction = predicted_y

        else:
            fitting_service = CurveFittingService(self.strategy)
            best_prediction, _ = fitting_service.fit(x_norm, y_norm, step=self.step)

        self.predicted_y = best_prediction * max(self.y)

        # Plot the results using PlottingService
        plot_service = PlottingService()
        plot_service.plot(self.x, self.y, self.predicted_x, self.predicted_y)


# Example
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
linear_y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Adding Gaussian noise to the y values
noise_strength = 1
noisy_linear_y = linear_y + np.random.normal(0, noise_strength, len(linear_y))

facade = DataPredictionFacade(x, noisy_linear_y, step=4)
facade.predict_and_plot()

facade = DataPredictionFacade(x, noisy_linear_y, strategy=LinearCurveStrategy(), step=4)
facade.predict_and_plot() 
