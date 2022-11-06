import math

import numpy as np
import statsmodels.api as sm
import logging
import pandas as pd
import matplotlib.pyplot as plt

# logger for this file
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.FileHandler('/tmp/tracker.log')
handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(levelname)-8s-[%(filename)s:%(lineno)d]-%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# df = pd.read_csv("HeadingData.csv", delimiter=",")


class RAKF1D:

    def __init__(self,
                 initial_state,
                 system_model,
                 system_model_error,
                 measurement_error,
                 state_error_variance,
                 residual_threshold, adaptive_threshold,
                 estimator_parameter_count,
                 gamma,
                 model_type="uwb_imu"):
        """Initializes RAKF 1 Dimensional instance

        Args:
            initial_state (float): Initial system state
            system_model (float): System model equation coefficient (F)
            system_model_error (float): System model error (variance of model error) (Q)
            measurement_error (float): measurement model error (variance of measurement error) (R in kalman filter)
            state_error_variance (float): Initial state error variance (P)
            residual_threshold (float): residual threshold value (c)
            adaptive_threshold (float): Adaptive threshold value (co)
            estimator_parameter_count (int, optional): Sample count for parameter estimation method. Defaults to 1.
            model_type (str, optional): Type of motion model. Defaults to "constant-position".
        """
        try:
            # timestamp
            self.time_previous = -1.0  # float

            # model type
            self.model_type = model_type  # string

            # states
            self.state_model_prediction = 0.0  # float
            self.state_model = float(initial_state)  # float state_model =Xk

            # system model
            self.system_model = float(system_model)  # float
            self.system_model_error = float(system_model_error)  # float

            # measurement
            self.state_measurement_relation = 1.0  # float
            self.measurement_standard_deviation = float(math.sqrt(measurement_error))  # float
            self.measurement_prediction = 0.0  # float

            # residual
            self.residual_threshold = float(residual_threshold)  # c # float
            self.residual_weight = 0.0  # float
            self.residual_measurement = 0.0  # float
            self.residual_measurement_dash = 0.0  # float
            self.gamma = float(gamma)  # float

            # state error variance
            self.state_error_variance_prediction = 0.0  # float
            self.state_error_variance = state_error_variance  # P # float

            # state estimation
            self.state_estimation = 0.0  # float
            self.delta_state_estimate = 0.0  # float

            # gain
            self.gain = 0.0  # float

            # adaptive
            self.adaptive_factor = 0.0  # float
            self.adaptive_threshold = float(adaptive_threshold)  # co

            # parameter estimation
            self.estimator_parameter_count = int(estimator_parameter_count)  # integer
            self.measurement_buffer = np.zeros(self.estimator_parameter_count)  # np array
            self.residual_weight_buffer = np.ones(self.estimator_parameter_count)  # np array
            self.position_buffer = np.zeros(self.estimator_parameter_count)  # np array
            # self.param_est = sm.WLS(self.measurement_buffer, self.position_buffer, self.residual_weight_buffer)

            self.Angularvelocity_buffer = np.zeros(self.estimator_parameter_count)  # np array
            self.acceleration_buffer = np.zeros(self.estimator_parameter_count)  # np array

        except Exception as e:
            logging.critical(e)
            exit(-1)

    def run(self,
            current_measurement,
            timestamp_ms=0,
            Angularvelocity=0,
            acceleration=0):
        """Runs RAKF 1D algorithm

        Args:
            :param current_measurement:  Measurement
            :param timestamp_ms: Timestamp in milliseconds (since epoch). Defaults to 0.
            :param Angularvelocity: Angular Velocity in rad per second. Defaults to 0.
            :param acceleration: Acceleration in meter per second^2. Defaults to 0.
        Returns:
            eqn_result, variable_result : Equation result and Variable result are dictionaries containing results of
            various parameters used in the algorithm calculation
        """
        try:
            # Get timedelta based on timestamp
            # if self.time_previous < 0.0:
            #     timedelta = 0.0
            # else:
            #     timedelta = timestamp_ms - self.time_previous
            #     timedelta /= 1000.0  # millisec to sec conversion
            # self.time_previous = float(timestamp_ms)
            timedelta = 0.01

            # -----------------  Prediction  -----------------------------------
            # equation 29
            self.state_model_prediction = float(
                (self.system_model * self.state_model) + (Angularvelocity * timedelta) +
                (acceleration * (timedelta ** 2) * 0.5))

            # equation 30
            self.state_error_variance_prediction = float(
                (self.system_model * self.state_error_variance * self.system_model) + self.system_model_error)
            # ----------------  Updating  ---------------------------------------

            # equation 35
            self.measurement_prediction = float(self.state_measurement_relation * self.state_model_prediction)

            # equation 34
            self.residual_measurement = float(current_measurement - self.measurement_prediction)

            # equation 33
            self.residual_measurement_dash = abs(float(self.residual_measurement / self.measurement_standard_deviation))

            # equation 31 & 32
            if self.residual_measurement_dash <= self.residual_threshold:
                self.residual_weight = float(1.0 / self.measurement_standard_deviation)
            else:
                # self.residual_weight = float(self.residual_threshold / (self.residual_measurement_dash * \
                #                                                   self.measurement_standard_deviation)) # as per paper
                self.residual_weight = float(
                    (self.residual_threshold / (self.residual_measurement_dash * 2.0 * self.gamma)) \
                    * (1 / self.measurement_standard_deviation))  # our change

            # equation 37 (different from paper , because angular velocity)
            # update position buffer
            self.position_buffer = np.roll(self.position_buffer, -1)  # Observed Position
            self.position_buffer[self.estimator_parameter_count - 1] = self.state_model

            self.measurement_buffer = np.roll(self.measurement_buffer, -1)
            self.measurement_buffer[self.estimator_parameter_count - 1] = current_measurement

            if self.model_type == 'uwb_imu':
                self.Angularvelocity_buffer = np.roll(self.Angularvelocity_buffer, -1)  # Observed Angularvelocity
                self.Angularvelocity_buffer[self.estimator_parameter_count - 1] = Angularvelocity

                self.acceleration_buffer = np.roll(self.acceleration_buffer, -1)  # Observed Acceleration
                self.acceleration_buffer[self.estimator_parameter_count - 1] = acceleration  # Observed acceleration

                uwb_imu_observation_matrix = np.stack(
                    [self.position_buffer, self.Angularvelocity_buffer, self.acceleration_buffer], axis=1)

                wls_model = sm.WLS(self.measurement_buffer, uwb_imu_observation_matrix,
                                   self.residual_weight_buffer).fit()  # with imu
                self.state_estimation = float(wls_model.predict(
                    [uwb_imu_observation_matrix[-1, 0], uwb_imu_observation_matrix[-1, 1],
                     uwb_imu_observation_matrix[-1, 2]]))
            else:
                wls_model = sm.WLS(self.measurement_buffer, self.position_buffer,
                                   self.residual_weight_buffer).fit()  # without imu only
                self.state_estimation = float(
                    wls_model.predict(self.position_buffer[self.estimator_parameter_count - 1]))

            # equation 36
            self.delta_state_estimate = float(
                (self.state_estimation - self.state_model_prediction) / self.state_error_variance_prediction)

            # equation 38
            if self.delta_state_estimate < self.adaptive_threshold:
                self.adaptive_factor = float(1.0)
            elif self.adaptive_threshold < self.delta_state_estimate < self.residual_threshold:
                self.adaptive_factor = float((self.adaptive_threshold / self.delta_state_estimate * self.gamma))
            else:
                self.adaptive_factor = float(self.delta_state_estimate * self.gamma)

            # equation 39
            reciprocal_adaptive_factor = 1.0 / self.adaptive_factor
            reciprocal_residual_weight = 1.0 / self.residual_weight
            numerator = float(reciprocal_adaptive_factor * self.state_error_variance_prediction *
                              self.state_measurement_relation)
            denominator = float((reciprocal_adaptive_factor * self.state_measurement_relation *
                                 self.state_error_variance_prediction * self.state_measurement_relation) +
                                reciprocal_residual_weight)
            self.gain = float(numerator / denominator)

            # equation 40, Computing the corrected state ˆXk:
            self.state_model = float(self.state_model_prediction + (self.gain * self.residual_measurement))

            # equation 41
            # not done here, as normalization is not need for 1 D

            # equation 42
            self.state_error_variance = float(
                (1 - self.gain * self.state_measurement_relation) * self.state_error_variance_prediction)

            # Activity related to eqn 37 , update parameters in parameter estimation based on states
            # self.param_est.adapt(self.state_model, self.measurement_buffer)
            self.residual_weight_buffer = np.roll(self.residual_weight_buffer, -1)
            self.residual_weight_buffer[self.estimator_parameter_count - 1] = self.residual_weight  # Weight

            return float(self.state_model)
        except Exception as e:
            logging.critical(e)
            exit(-1)

    def update_state(self, state_information):
        # timestamp
        self.time_previous = float(state_information["time_previous"])  # float

        # model type
        self.model_type = state_information["model_type"]  # string

        # states
        self.state_model_prediction = float(state_information["state_model_prediction"])  # float
        self.state_model = float(state_information["state_model"])  # float

        # system model
        self.system_model = float(state_information["system_model"])  # float
        self.system_model_error = float(state_information["system_model_error"])  # float

        # measurement
        self.state_measurement_relation = float(state_information["state_measurement_relation"])  # float
        self.measurement_standard_deviation = float(state_information["measurement_standard_deviation"])  # float
        self.measurement_prediction = float(state_information["measurement_prediction"])  # float

        # residual
        self.residual_threshold = float(state_information["residual_threshold"])  # c # float
        self.residual_weight = float(state_information["residual_weight"])  # float
        self.residual_measurement = float(state_information["residual_measurement"])  # float
        self.residual_measurement_dash = float(state_information["residual_measurement_dash"])  # float
        self.gamma = float(state_information["gamma"])  # float

        # state error variance
        self.state_error_variance_prediction = float(state_information["state_error_variance_prediction"])  # float
        self.state_error_variance = float(state_information["state_error_variance"])  # P # float

        # state estimation
        self.state_estimation = float(state_information["state_estimation"])  # float
        self.delta_state_estimate = float(state_information["delta_state_estimate"])  # float

        # gain
        self.gain = float(state_information["gain"])  # float

        # adaptive
        self.adaptive_factor = float(state_information["adaptive_factor"])  # float
        self.adaptive_threshold = float(state_information["adaptive_threshold"])  # co

        # parameter estimation
        self.estimator_parameter_count = int(state_information["estimator_parameter_count"])  # integer

        self.measurement_buffer = np.array(state_information["measurement_buffer"], dtype=np.float32)  # np array
        self.residual_weight_buffer = np.array(state_information["residual_weight_buffer"],
                                               dtype=np.float32)  # np array
        self.position_buffer = np.array(state_information["position_buffer"], dtype=np.float32)  # np array

        self.Angularvelocity_buffer = np.array(state_information["Angularvelocity_buffer"],
                                               dtype=np.float32)  # np array
        self.acceleration_buffer = np.array(state_information["acceleration_buffer"], dtype=np.float32)  # np array

    def state_to_dict(self):
        state_information = {
            "time_previous": self.time_previous,
            "model_type": self.model_type,
            "state_model_prediction": self.state_model_prediction,
            "state_model": self.state_model,
            "system_model": self.system_model,
            "system_model_error": self.system_model_error,
            "state_measurement_relation": self.state_measurement_relation,
            "measurement_standard_deviation": self.measurement_standard_deviation,
            "measurement_prediction": self.measurement_prediction,
            "residual_threshold": self.residual_threshold,
            "residual_weight": self.residual_weight,
            "residual_measurement": self.residual_measurement,
            "residual_measurement_dash": self.residual_measurement_dash,
            "gamma": self.gamma,
            "state_error_variance_prediction": self.state_error_variance_prediction,
            "state_error_variance": self.state_error_variance,
            "state_estimation": self.state_estimation,
            "delta_state_estimate": self.delta_state_estimate,
            "gain": self.gain,
            "adaptive_factor": self.adaptive_factor,
            "adaptive_threshold": self.adaptive_threshold,
            "estimator_parameter_count": self.estimator_parameter_count,
            "measurement_buffer": self.measurement_buffer.tolist(),
            "residual_weight_buffer": self.residual_weight_buffer.tolist(),
            "position_buffer": self.position_buffer.tolist(),
            "Angularvelocity_buffer": self.Angularvelocity_buffer.tolist(),
            "acceleration_buffer": self.acceleration_buffer.tolist()
        }
        return state_information


# df['Yaw'] = df['Yaw'].astype(float)
# yaw= df['Yaw']

# yaw= (df.Yaw).apply(lambda x: float(x))

# yaw=df['Yaw'].to_numpy()


in_file = open('LshapedData3_withAngularVelocity-inRadians.csv', 'r')
in_file.readline()
estimated_heading=0
yaw_list = list()
excel_yaw = 0
excel_reference =0
excel_yaw_list = list()
time = list()
reference_list= list()

for temp_line in in_file.readlines():
    temp_seq = temp_line.split(',')
    yaw = float(temp_seq[6])
    angularVelocity_Z = float(temp_seq[1])
    time.append(float(temp_seq[0]))
    heading_estimate = 0
    heading_estimate = RAKF1D(initial_state=0,
                              system_model=1,
                              system_model_error=5,  # 1 means its trusting model more
                              measurement_error=10.88,
                              state_error_variance=7,
                              residual_threshold=12, #c
                              adaptive_threshold=1,  #c0
                              estimator_parameter_count=10,
                              gamma=1,
                              model_type="uwb_imu")

    heading_estimate.run(current_measurement=yaw, Angularvelocity = angularVelocity_Z)
    estimated_heading = heading_estimate.state_model
    print(estimated_heading)
    excel_yaw = yaw
    excel_yaw_list.append(excel_yaw)
    yaw_list.append(estimated_heading)
    # excel_reference=reference
    reference_list.append(excel_reference)

print("heading", heading_estimate.state_model)
#plt.plot(yaw)
#plt.plot(time, yaw, c='b', marker='x', label='1')
plt.plot(time, yaw_list,   label='Estimated Heading')
plt.plot(time, excel_yaw_list, '--', label='Heading')
# plt.plot(time, reference_list, ':', label='Reference')
plt.legend(loc='upper left')
plt.xlabel('Sample time fine')
plt.ylabel('Heading (Degree)')
plt.show()


in_file.close()
