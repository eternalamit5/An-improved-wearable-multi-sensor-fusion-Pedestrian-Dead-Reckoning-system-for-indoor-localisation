from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from window_slider import Slider
from MLP.FirFilter import FirFilter
import statistics
from sklearn import preprocessing
from scipy.fft import rfft, fftfreq
import seaborn as sn




class HumanMotionClassifierFeatures:
    def __init__(self, dataframe, axis_names, sample_rate_hz, cutoff_freq_hz, window_size, overlap_size):
        self.dataframe = dataframe
        self.sample_rate_hz = sample_rate_hz
        self.cutoff_freq_hz = cutoff_freq_hz
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.axis_names = axis_names

        self.fir_filter = FirFilter( dataframe, sample_rate_hz=sample_rate_hz, cutoff_freq_hz=cutoff_freq_hz )

        self.filtered_signal_x, self.raw_signal_x, self.timestamp_x = self.fir_filter.filter( signal_param=self.axis_names[0] )
        self.sliding_window_x = Slider( self.window_size, self.overlap_size )
        self.sliding_window_x.fit( self.filtered_signal_x )

        self.filtered_signal_y, self.raw_signal_y, self.timestamp_y = self.fir_filter.filter( signal_param=self.axis_names[1] )
        self.sliding_window_y = Slider( self.window_size, self.overlap_size )
        self.sliding_window_y.fit( self.filtered_signal_y )

        self.filtered_signal_z, self.raw_signal_z, self.timestamp_z = self.fir_filter.filter( signal_param=self.axis_names[2] )
        self.sliding_window_z = Slider( self.window_size, self.overlap_size )
        self.sliding_window_z.fit( self.filtered_signal_z )

        self.feature_list = []

    def get_features(self,target_name):
        while True:
            window_data_x = self.sliding_window_x.slide()
            window_data_y = self.sliding_window_y.slide()
            window_data_z = self.sliding_window_z.slide()
            features_dict = dict()
            features_dict.update( self.build_feature_set( dataset=window_data_x, col_name=self.axis_names[0], target_name=target_name ) )
            features_dict.update( self.build_feature_set( dataset=window_data_y, col_name=self.axis_names[1], target_name=target_name ) )
            features_dict.update( self.build_feature_set( dataset=window_data_z, col_name=self.axis_names[2], target_name=target_name ) )
            self.feature_list.append( features_dict )
            if self.sliding_window_x.reached_end_of_list(): break
        return pd.DataFrame( self.feature_list )

    def build_feature_set(self, dataset, col_name, target_name):
        # perform FFT and extract dominating frequency
        fft_magnitude = abs( rfft( dataset ) )
        fft_positive_freq_bin = fftfreq( self.window_size, 1 / (self.cutoff_freq_hz * 2) )[0:(int)( self.window_size / 2 )]
        fft_result = dict( zip( fft_positive_freq_bin, fft_magnitude ) )

        # remove dc component
        # fft_result.pop( 0 )

        sorted_fft_result = {k: v for k, v in sorted( fft_result.items(), key=lambda item: item[1], reverse=True )}
        dominating_freq_list = np.array(list( sorted_fft_result.keys() ))
        dominating_norm_freq_mag_list = np.array(list( sorted_fft_result.values() ))
        dominating_norm_freq_mag_list = dominating_norm_freq_mag_list / dominating_norm_freq_mag_list.max()

        features = {col_name + "_mean": statistics.mean( data=dataset ),
                    col_name + "_var": statistics.variance( data=dataset ),
                    col_name + "_sd": statistics.stdev( data=dataset ),
                    # col_name + "_per_25": np.percentile( dataset, 25 ),
                    # col_name + "_per_75": np.percentile( dataset, 75 ),
                    col_name + "_freq_1": dominating_freq_list[0],
                    col_name + "_freq_2": dominating_freq_list[1],
                    col_name + "_freq_3": dominating_freq_list[2],
                    col_name + "_freq_4": dominating_freq_list[3],
                    col_name + "_freq_5": dominating_freq_list[4],
                    col_name + "_freq_mag_1": dominating_norm_freq_mag_list[0],
                    col_name + "_freq_mag_2": dominating_norm_freq_mag_list[1],
                    col_name + "_freq_mag_3": dominating_norm_freq_mag_list[2],
                    col_name + "_freq_mag_4": dominating_norm_freq_mag_list[3],
                    col_name + "_freq_mag_5": dominating_norm_freq_mag_list[4],
                    'target': target_name
                    }
        return features


class HumanMotionClassifier:
    def __init__(self):
        self.feature_df_list = []
        self.normalized_input_vector = []
        self.encoded_output_vector = []

    def classify(self):
        feature_vector = pd.concat( self.feature_df_list )

        # reindex
        feature_vector.reset_index( drop=True, inplace=True )

        # encode output
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit( feature_vector['target'] )
        self.encoded_output_vector = label_encoder.transform( feature_vector['target'] )
        print(self.encoded_output_vector)

        # normalize the input data
        normalizer = preprocessing.Normalizer().fit( feature_vector.drop( ['target'], axis=1 ) )
        self.normalized_input_vector = normalizer.transform( feature_vector.drop( ['target'], axis=1 ) )

        # split the feature vector into train and test data
        x_train, x_test, y_train, y_test = train_test_split( self.normalized_input_vector, self.encoded_output_vector, test_size=0.25, random_state=27 )

        # MLP classifier instance
        mlp_classifier = MLPClassifier( hidden_layer_sizes=(5, 5), max_iter=500, alpha=0.0001, activation='tanh',
                                        solver='adam', verbose=10, random_state=21, tol=0.000000001, shuffle=True )

        # start training
        mlp_classifier.fit( x_train, y_train )

        # start testing
        y_pred = mlp_classifier.predict( x_test )

        # publish test statistics
        print( accuracy_score( y_test, y_pred ) )
        print( confusion_matrix( y_test, y_pred ) )
        print( classification_report( y_test, y_pred ) )

        # Get the confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)
        sn.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
                    fmt='.2%', cmap='Blues')

        # cm= tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
        # plt.figure(figsize = (10,7))
        # sn.figure(cm, annot=True, fmt='d')
        # plt.xlable('Predicted')
        # plt.ylabel('Truth')

    def add_feature_set(self, dataframe):
        self.feature_df_list.append( dataframe )




if __name__ == '__main__':
    WINDOW_SIZE = 64
    OVERLAP_SIZE = 63
    CUTOFF_FREQ = 4.5
    SAMPLE_FREQ = 100
    # get dataset from csv in the form of dataframe
    # she_slow_walk_df = pd.read_csv( '../../DataSet/Test4/she/she_slow_walk_19_dec_2020.csv' )
    # she_norm_walk_df = pd.read_csv( '../../DataSet/Test4/she/she_norm_walk_19_dec_2020.csv' )
    # she_fast_walk_df = pd.read_csv( '../../DataSet/Test4/she/she_fast_walk_19_dec_2020.csv' )
    # she_jog_walk_df = pd.read_csv( '../../DataSet/Test4/she/she_jog_19_dec_2020.csv' )
    # she_run_walk_df = pd.read_csv( '../../DataSet/Test4/she/she_run_19_dec_2020.csv' )
    # she_stand_walk_df = pd.read_csv( '../../DataSet/Test4/she/she_stand_19_dec_2020.csv' )
    # shan_slow_walk_df = pd.read_csv( '../../DataSet/Test4/shan/shan_slow_walk_19_dec_2020.csv' )
    # shan_norm_walk_df = pd.read_csv( '../../DataSet/Test4/shan/shan_norm_walk_19_dec_2020.csv' )
    # shan_fast_walk_df = pd.read_csv( '../../DataSet/Test4/shan/shan_fast_walk_19_dec_2020.csv' )
    # shan_jog_walk_df = pd.read_csv( '../../DataSet/Test4/shan/shan_jog_19_dec_2020.csv' )
    # shan_run_walk_df = pd.read_csv( '../../DataSet/Test4/shan/shan_run_19_dec_2020.csv' )
    # shan_stand_walk_df = pd.read_csv( '../../DataSet/Test4/shan/shan_stand_19_dec_2020.csv' )

    she_slow_walk_df = pd.read_csv( '../../DataSet/Test5/she_walk_lap1_21_dec_2020.csv' )
    she_norm_walk_df = pd.read_csv( '../../DataSet/Test5/she_walk_lap2_21_dec_2020.csv' )
    # she_fast_walk_df = pd.read_csv( '../../DataSet/Test4/she/she_fast_walk_19_dec_2020.csv' )
    she_jog_walk_df = pd.read_csv( '../../DataSet/Test5/she_run_lap1_21_dec_2020.csv' )
    she_run_walk_df = pd.read_csv( '../../DataSet/Test5/she_run_lap2_21_dec_2020.csv' )
    she_stand_walk_df = pd.read_csv( '../../DataSet/Test5/she_stand_21_dec_2020.csv' )

    shan_slow_walk_df = pd.read_csv( '../../DataSet/Test5/shan_walk_lap1_21_dec_2020.csv' )
    shan_norm_walk_df = pd.read_csv( '../../DataSet/Test5/shan_walk_lap2_21_dec_2020.csv' )
    # shan_fast_walk_df = pd.read_csv( '../../DataSet/Test4/shan/shan_fast_walk_19_dec_2020.csv' )
    shan_jog_walk_df = pd.read_csv( '../../DataSet/Test5/shan_run_lap1_21_dec_2020.csv' )
    shan_run_walk_df = pd.read_csv( '../../DataSet/Test5/shan_run_lap2_21_dec_2020.csv' )
    shan_stand_walk_df = pd.read_csv( '../../DataSet/Test5/shan_stand_21_dec_2020.csv' )

    # initialize classifier with feature sets
    hm_clf = HumanMotionClassifier()
    hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=she_slow_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                           window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features( target_name='walk' ) )
    hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=she_norm_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                           window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features( target_name='walk' ) )
    # hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=she_fast_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
    #                                                        window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features( target_name='walk' ) )
    hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=she_jog_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                           window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features( target_name='run' ) )
    hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=she_run_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                           window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features( target_name='run' ) )
    hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=she_stand_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                           window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features( target_name='stand' ) )
    hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=shan_slow_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                           window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features( target_name='walk' ) )
    hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=shan_norm_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                           window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features( target_name='walk' ) )
    # hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=shan_fast_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
    #                                                        window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features( target_name='walk' ) )
    hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=shan_jog_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                           window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features( target_name='run' ) )
    hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=shan_run_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                           window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features( target_name='run' ) )
    hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=shan_stand_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                           window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features( target_name='stand' ) )

    #  classify
    hm_clf.classify()
