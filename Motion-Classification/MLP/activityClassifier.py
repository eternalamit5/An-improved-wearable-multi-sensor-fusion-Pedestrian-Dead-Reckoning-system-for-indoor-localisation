from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd
from glob import glob
import numpy as np
from FirFilter import FIR
import statistics
from sklearn import preprocessing
from scipy.fft import rfft, ifft, fftfreq
from activityAnalysis import MeasurementExtractor
import seaborn as sn


class ActivityFeatureExtraction:
    def __init__(self, measurement: pd.DataFrame,
                 activity: str,
                 user_id: str,
                 experiment_id: str,
                 sample_frequency: float = 100,
                 cutoff_frequency: float = 4.5,
                 window_size: int = 100):
        self._measurement = measurement
        self._user_id = user_id
        self._experiment_id = experiment_id
        self._activity = activity
        self._sample_frequency = sample_frequency
        self._cutoff_frequency = cutoff_frequency
        self._window_size = window_size
        self._filtered_measurement = pd.concat([self._accelerometer_fir(),
                                                self._gyroscope_fir()], axis=1)

    @property
    def measurement(self) -> pd.DataFrame:
        return self._measurement

    @property
    def filtered_measurement(self) -> pd.DataFrame:
        return self._filtered_measurement

    @property
    def sample_frequency(self) -> float:
        return self._sample_frequency

    @property
    def cutoff_frequency(self) -> float:
        return self._cutoff_frequency

    @property
    def window_size(self) -> int:
        return self._window_size

    def show(self, title):
        acc_fir_x = FIR(signal=self.measurement['acc_x'].to_numpy(),
                        filter_type="lowpass",
                        sample_rate_hz=self.sample_frequency,
                        cutoff_freq_hz=self.cutoff_frequency)
        acc_fir_x.show(title=title + " accelerometer x")

        acc_fir_y = FIR(signal=self.measurement['acc_y'].to_numpy(),
                        filter_type="lowpass",
                        sample_rate_hz=self.sample_frequency,
                        cutoff_freq_hz=self.cutoff_frequency)
        acc_fir_y.show(title=title + " accelerometer y")

        acc_fir_z = FIR(signal=self.measurement['acc_z'].to_numpy(),
                        filter_type="lowpass",
                        sample_rate_hz=self.sample_frequency,
                        cutoff_freq_hz=self.cutoff_frequency)
        acc_fir_z.show(title=title + " accelerometer z")

        gyro_fir_x = FIR(signal=self.measurement['gyro_x'].to_numpy(),
                         filter_type="lowpass",
                         sample_rate_hz=self.sample_frequency,
                         cutoff_freq_hz=self.cutoff_frequency)
        gyro_fir_x.show(title=title + " gyroscope x")

        gyro_fir_y = FIR(signal=self.measurement['gyro_y'].to_numpy(),
                         filter_type="lowpass",
                         sample_rate_hz=self.sample_frequency,
                         cutoff_freq_hz=self.cutoff_frequency)
        gyro_fir_y.show(title=title + " gyroscope y")

        gyro_fir_z = FIR(signal=self.measurement['gyro_z'].to_numpy(),
                         filter_type="lowpass",
                         sample_rate_hz=self.sample_frequency,
                         cutoff_freq_hz=self.cutoff_frequency)
        gyro_fir_z.show(title=title + " gyroscope z")

    def _accelerometer_fir(self):
        fir_x = FIR(signal=self.measurement['acc_x'].to_numpy(),
                    filter_type="lowpass",
                    sample_rate_hz=self.sample_frequency,
                    cutoff_freq_hz=self.cutoff_frequency)

        fir_y = FIR(signal=self.measurement['acc_y'].to_numpy(),
                    filter_type="lowpass",
                    sample_rate_hz=self.sample_frequency,
                    cutoff_freq_hz=self.cutoff_frequency)

        fir_z = FIR(signal=self.measurement['acc_z'].to_numpy(),
                    filter_type="lowpass",
                    sample_rate_hz=self.sample_frequency,
                    cutoff_freq_hz=self.cutoff_frequency)

        result = pd.DataFrame({"acc_x": fir_x.filter(),
                               "acc_y": fir_y.filter(),
                               "acc_z": fir_z.filter()})
        return result

    def _gyroscope_fir(self):
        fir_x = FIR(signal=self.measurement['gyro_x'].to_numpy(),
                    filter_type="lowpass",
                    sample_rate_hz=self.sample_frequency,
                    cutoff_freq_hz=self.cutoff_frequency)

        fir_y = FIR(signal=self.measurement['gyro_y'].to_numpy(),
                    filter_type="lowpass",
                    sample_rate_hz=self.sample_frequency,
                    cutoff_freq_hz=self.cutoff_frequency)

        fir_z = FIR(signal=self.measurement['gyro_z'].to_numpy(),
                    filter_type="lowpass",
                    sample_rate_hz=self.sample_frequency,
                    cutoff_freq_hz=self.cutoff_frequency)

        result = pd.DataFrame({"gyro_x": fir_x.filter(),
                               "gyro_y": fir_y.filter(),
                               "gyro_z": fir_z.filter()})
        return result

    def _accelerometer_mean(self) -> dict:
        return {
            "acc_x": self.filtered_measurement["acc_x"].mean(),
            "acc_y": self.filtered_measurement["acc_y"].mean(),
            "acc_z": self.filtered_measurement["acc_z"].mean()
        }

    def _gyroscope_mean(self) -> dict:
        return {
            "gyro_x": self.filtered_measurement["gyro_x"].mean(),
            "gyro_y": self.filtered_measurement["gyro_y"].mean(),
            "gyro_z": self.filtered_measurement["gyro_z"].mean()
        }

    def _accelerometer_variance(self) -> dict:
        return {
            "acc_x": self.filtered_measurement["acc_x"].var(),
            "acc_y": self.filtered_measurement["acc_y"].var(),
            "acc_z": self.filtered_measurement["acc_z"].var()
        }

    def _gyroscope_variance(self) -> dict:
        return {
            "gyro_x": self.filtered_measurement["gyro_x"].var(),
            "gyro_y": self.filtered_measurement["gyro_y"].var(),
            "gyro_z": self.filtered_measurement["gyro_z"].var()
        }

    def _accelerometer_standard_deviation(self) -> dict:
        return {
            "acc_x": self.filtered_measurement["acc_x"].std(),
            "acc_y": self.filtered_measurement["acc_y"].std(),
            "acc_z": self.filtered_measurement["acc_z"].std()
        }

    def _gyroscope_standard_deviation(self) -> dict:
        return {
            "gyro_x": self.filtered_measurement["gyro_x"].std(),
            "gyro_y": self.filtered_measurement["gyro_y"].std(),
            "gyro_z": self.filtered_measurement["gyro_z"].std()
        }

    def _accelerometer_quantile(self, quant) -> dict:
        return {
            "acc_x": self.filtered_measurement["acc_x"].quantile(quant),
            "acc_y": self.filtered_measurement["acc_y"].quantile(quant),
            "acc_z": self.filtered_measurement["acc_z"].quantile(quant)
        }

    def _gyroscope_quantile(self, quant) -> dict:
        return {
            "gyro_x": self.filtered_measurement["gyro_x"].quantile(quant),
            "gyro_y": self.filtered_measurement["gyro_y"].quantile(quant),
            "gyro_z": self.filtered_measurement["gyro_z"].quantile(quant)
        }

    def _accelerometer_percentile(self, value) -> dict:
        return {
            "acc_x": np.percentile(self.filtered_measurement["acc_x"].to_numpy(), value),
            "acc_y": np.percentile(self.filtered_measurement["acc_y"].to_numpy(), value),
            "acc_z": np.percentile(self.filtered_measurement["acc_z"].to_numpy(), value),
        }

    def _gyroscope_percentile(self, value) -> dict:
        return {
            "gyro_x": np.percentile(self.filtered_measurement["gyro_x"].to_numpy(), value),
            "gyro_y": np.percentile(self.filtered_measurement["gyro_y"].to_numpy(), value),
            "gyro_z": np.percentile(self.filtered_measurement["gyro_z"].to_numpy(), value),
        }

    def _dominating_frequencies(self, signal_param: str):
        fft_freq_bin = fftfreq(self.window_size, 1 / (self.cutoff_frequency * 2))[0:int(self.window_size / 2)]
        frequency_resp = dict(zip(fft_freq_bin, abs(rfft(self.filtered_measurement[signal_param].to_numpy()))))
        sorted_frequency_resp = {k: v for k, v in
                                 sorted(frequency_resp.items(), key=lambda item: item[1], reverse=True)[:5]}
        dominating_freq = list(sorted_frequency_resp.keys())
        return dominating_freq[0:5]

    def _accelerometer_fft(self):
        result = {
            "fft": {
                "real": {
                    "acc_x": rfft(self.filtered_measurement["acc_x"].to_numpy()),
                    "acc_y": rfft(self.filtered_measurement["acc_y"].to_numpy()),
                    "acc_z": rfft(self.filtered_measurement["acc_z"].to_numpy()),
                },
                "absolute_magnitude": {
                    "acc_x": abs(rfft(self.filtered_measurement["acc_x"].to_numpy())),
                    "acc_y": abs(rfft(self.filtered_measurement["acc_y"].to_numpy())),
                    "acc_z": abs(rfft(self.filtered_measurement["acc_z"].to_numpy())),
                },
                "dominant_frequencies": {
                    "acc_x": self._dominating_frequencies('acc_x'),
                    "acc_y": self._dominating_frequencies('acc_y'),
                    "acc_z": self._dominating_frequencies('acc_z'),
                }
            }
        }
        return result

    def _gyroscope_fft(self):
        result = {
            "fft": {
                "real": {
                    "gyro_x": rfft(self.filtered_measurement["gyro_x"].to_numpy()),
                    "gyro_y": rfft(self.filtered_measurement["gyro_y"].to_numpy()),
                    "gyro_z": rfft(self.filtered_measurement["gyro_z"].to_numpy())
                },
                "absolute_magnitude": {
                    "gyro_x": abs(rfft(self.filtered_measurement["gyro_x"].to_numpy())),
                    "gyro_y": abs(rfft(self.filtered_measurement["gyro_y"].to_numpy())),
                    "gyro_z": abs(rfft(self.filtered_measurement["gyro_z"].to_numpy()))
                },
                "dominant_frequencies": {
                    "gyro_x": self._dominating_frequencies('gyro_x'),
                    "gyro_y": self._dominating_frequencies('gyro_y'),
                    "gyro_z": self._dominating_frequencies('gyro_z'),
                }
            }
        }
        return result

    def extract_features(self):
        return {
            "activity": self._activity,
            "user_id": self._user_id,
            "experiment_id": self._experiment_id,

            "acc_x_mean": self._accelerometer_mean()['acc_x'],
            "acc_y_mean": self._accelerometer_mean()['acc_y'],
            "acc_z_mean": self._accelerometer_mean()['acc_z'],

            "acc_x_variance": self._accelerometer_variance()['acc_x'],
            "acc_y_variance": self._accelerometer_variance()['acc_y'],
            "acc_z_variance": self._accelerometer_variance()['acc_z'],

            "acc_x_std": self._accelerometer_standard_deviation()['acc_x'],
            "acc_y_std": self._accelerometer_standard_deviation()['acc_y'],
            "acc_z_std": self._accelerometer_standard_deviation()['acc_z'],

            "acc_x_percentile_1": self._accelerometer_percentile(0.25)['acc_x'],
            "acc_y_percentile_1": self._accelerometer_percentile(0.25)['acc_y'],
            "acc_z_percentile_1": self._accelerometer_percentile(0.25)['acc_z'],

            "acc_x_percentile_2": self._accelerometer_percentile(0.75)['acc_x'],
            "acc_y_percentile_2": self._accelerometer_percentile(0.75)['acc_y'],
            "acc_z_percentile_2": self._accelerometer_percentile(0.75)['acc_z'],

            "acc_x_dominant_frequency_1": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_x"][0],
            "acc_x_dominant_frequency_2": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_x"][1],
            "acc_x_dominant_frequency_3": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_x"][2],
            "acc_x_dominant_frequency_4": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_x"][3],
            "acc_x_dominant_frequency_5": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_x"][4],

            "acc_y_dominant_frequency_1": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_y"][0],
            "acc_y_dominant_frequency_2": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_y"][1],
            "acc_y_dominant_frequency_3": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_y"][2],
            "acc_y_dominant_frequency_4": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_y"][3],
            "acc_y_dominant_frequency_5": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_y"][4],

            "acc_z_dominant_frequency_1": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_z"][0],
            "acc_z_dominant_frequency_2": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_z"][1],
            "acc_z_dominant_frequency_3": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_z"][2],
            "acc_z_dominant_frequency_4": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_z"][3],
            "acc_z_dominant_frequency_5": self._accelerometer_fft()["fft"]["dominant_frequencies"]["acc_z"][4],

            "gyro_x_mean": self._gyroscope_mean()["gyro_x"],
            "gyro_y_mean": self._gyroscope_mean()["gyro_y"],
            "gyro_z_mean": self._gyroscope_mean()["gyro_z"],

            "gyro_x_variance": self._gyroscope_variance()["gyro_x"],
            "gyro_y_variance": self._gyroscope_variance()["gyro_y"],
            "gyro_z_variance": self._gyroscope_variance()["gyro_z"],

            "gyro_x_std": self._gyroscope_standard_deviation()["gyro_x"],
            "gyro_y_std": self._gyroscope_standard_deviation()["gyro_y"],
            "gyro_z_std": self._gyroscope_standard_deviation()["gyro_z"],

            "gyro_x_percentile_1": self._gyroscope_percentile(0.25)["gyro_x"],
            "gyro_y_percentile_1": self._gyroscope_percentile(0.25)["gyro_y"],
            "gyro_z_percentile_1": self._gyroscope_percentile(0.25)["gyro_z"],

            "gyro_x_percentile_2": self._gyroscope_percentile(0.75)["gyro_x"],
            "gyro_y_percentile_2": self._gyroscope_percentile(0.75)["gyro_y"],
            "gyro_z_percentile_2": self._gyroscope_percentile(0.75)["gyro_z"],

            "gyro_x_dominant_frequency_1": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_x"][0],
            "gyro_x_dominant_frequency_2": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_x"][1],
            "gyro_x_dominant_frequency_3": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_x"][2],
            "gyro_x_dominant_frequency_4": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_x"][3],
            "gyro_x_dominant_frequency_5": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_x"][4],

            "gyro_y_dominant_frequency_1": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_y"][0],
            "gyro_y_dominant_frequency_2": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_y"][1],
            "gyro_y_dominant_frequency_3": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_y"][2],
            "gyro_y_dominant_frequency_4": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_y"][3],
            "gyro_y_dominant_frequency_5": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_y"][4],

            "gyro_z_dominant_frequency_1": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_z"][0],
            "gyro_z_dominant_frequency_2": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_z"][1],
            "gyro_z_dominant_frequency_3": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_z"][2],
            "gyro_z_dominant_frequency_4": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_z"][3],
            "gyro_z_dominant_frequency_5": self._gyroscope_fft()["fft"]["dominant_frequencies"]["gyro_z"][4]
        }


class ActivityClassifier:
    def __init__(self, input_vector, output_vector):
        self._classifier = None
        self._input_vector = input_vector
        self._output_vector = output_vector
        self._input_vector_normalizer = preprocessing.Normalizer().fit(input_vector)
        self._input_vector_norm = self._input_vector_normalizer.transform(input_vector)
        self._label_encoder = preprocessing.LabelEncoder()
        self._output_vector_norm = self._label_encoder.fit(output_vector)
        self._initialize_classifier()

    @property
    def classifier(self):
        return self._classifier

    @property
    def input_vector(self):
        return self._input_vector

    @property
    def output_vector(self):
        return self._output_vector

    @property
    def input_vector_normalizer(self):
        return self._input_vector_normalizer

    @property
    def input_vector_norm(self):
        return self._input_vector_norm

    @property
    def label_encoder(self):
        return self._label_encoder

    @property
    def output_vector_norm(self):
        return self._output_vector_norm

    def grid_search_cv(self):
        parameter_space = {
            'hidden_layer_sizes': [(10, 30, 10), (20,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }
        mlp_gs = MLPClassifier(max_iter=100)
        clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
        clf.fit(self._input_vector_norm, self._output_vector)
        print('Best parameters found:\n', clf.best_params_)

    def _initialize_classifier(self, hidden_layer_sizes=(20,),
                               max_iter=5000,
                               alpha=0.0001,
                               learning_rate='constant',
                               activation='tanh',
                               solver='adam',
                               random_state=2,
                               tol=0.000000001,
                               shuffle=True):
        self._classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                         max_iter=max_iter,
                                         alpha=alpha,
                                         activation=activation,
                                         solver=solver,
                                         verbose=10,
                                         random_state=random_state,
                                         tol=tol,
                                         shuffle=shuffle)

    def classify(self, test_size, random_state):
        input_train_set, input_test_set, output_train_set, output_test_set = \
            train_test_split(self._input_vector_norm, self._output_vector,
                             test_size=test_size,
                             random_state=random_state)
        self.classifier.fit(input_train_set, output_train_set)
        prediction = self.classifier.predict(input_test_set)
        result = {
            "prediction": prediction,
            "output_test_set": output_test_set
        }
        return result

    @staticmethod
    def show_stats(classification_result):
        print(accuracy_score(classification_result["output_test_set"], classification_result["prediction"]))
        print(confusion_matrix(classification_result["output_test_set"], classification_result["prediction"]))
        print(classification_report(classification_result["output_test_set"], classification_result["prediction"]))
        cf_matrix = confusion_matrix(classification_result["output_test_set"], classification_result["prediction"])
        sn.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
                   fmt='.2%', cmap='Blues')
