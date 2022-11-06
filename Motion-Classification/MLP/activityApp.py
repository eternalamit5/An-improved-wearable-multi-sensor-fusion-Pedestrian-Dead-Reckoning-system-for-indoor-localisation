from glob import glob

import pandas as pd

from activityAnalysis import MeasurementExtractor
from activityClassifier import ActivityFeatureExtraction, ActivityClassifier


def app_main():
    files = sorted(glob("../FinaldataSet/*"))
    measurements = list()
    for measurement_file in files:
        measurements.append(MeasurementExtractor(measurement_file=measurement_file,column_labels=['acc_x',
                                                                                                  'acc_y',
                                                                                                  'acc_z',
                                                                                                  'gyro_x',
                                                                                                  'gyro_y',
                                                                                                  'gyro_z',
                                                                                                  'activity']))
    feature_list = list()
    for measurement in measurements:
        m1 = measurement.read_measurement()
        af = ActivityFeatureExtraction(measurement=m1,
                                       activity=m1['activity'][0],
                                       user_id=measurement.user_id,
                                       experiment_id=measurement.experiment_id)
        feature_list.append(af.extract_features())

    # show fir plot
    fir_m = measurements[0].read_measurement()
    fir_af = ActivityFeatureExtraction(measurement=fir_m,
                                   activity=fir_m['activity'][0],
                                   user_id=measurement.user_id,
                                   experiment_id=measurement.experiment_id)
    fir_af.show(title="FIR filter result for")


    vectors = pd.DataFrame(feature_list)

    input_vector = vectors.iloc[:,3:]
    output_vector = vectors.iloc[:,0]

    # classification
    ac = ActivityClassifier(input_vector=input_vector, output_vector=output_vector)

    # grid search cv
    # ac.grid_search_cv()

    res = ac.classify(test_size=0.5, random_state=27)
    ac.show_stats(classification_result=res)


if __name__ == '__main__':
    app_main()
