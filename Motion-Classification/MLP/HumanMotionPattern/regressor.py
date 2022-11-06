from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from window_slider import Slider
from Tools.Filters.FirFilter import FirFilter
import statistics
from sklearn.preprocessing import StandardScaler
from scipy.fft import rfft, ifft, fftfreq
from classifier import HumanMotionClassifierFeatures
import matplotlib.pyplot as plt
import math



class HumanMotionRegressorFeatures:
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

        self.filtered_signal_gvel, self.raw_signal_gvel, self.timestamp_gvel = self.fir_filter.filter( signal_param=self.axis_names[3] )
        self.sliding_window_gvel = Slider( self.window_size, self.overlap_size )
        self.sliding_window_gvel.fit( self.filtered_signal_gvel )

        self.feature_list = []

    def get_features(self):
        while True:
            if CONFIG == '1':
                window_data_x = self.sliding_window_x.slide()
                window_data_y = self.sliding_window_y.slide()
                window_data_z = self.sliding_window_z.slide()
                window_data_gvel = self.sliding_window_gvel.slide()

                target_value = self.get_target_value( dataset=window_data_gvel )

                features_dict = dict()
                features_dict.update( self.build_feature_set( window_data_x, self.axis_names[0], target_value ) )
                features_dict.update( self.build_feature_set( window_data_y, self.axis_names[1], target_value ) )
                features_dict.update( self.build_feature_set( window_data_z, self.axis_names[2], target_value ) )
                # features_dict.update({'target': target_value})
                self.feature_list.append( features_dict )
                if self.sliding_window_x.reached_end_of_list(): break
            elif CONFIG == '2':
                window_data_x = self.sliding_window_x.slide()
                window_data_y = self.sliding_window_y.slide()
                window_data_z = self.sliding_window_z.slide()
                window_data_gvel = self.sliding_window_gvel.slide()

                target_value = self.get_target_value( dataset=window_data_gvel )
                features_dict = dict()
                features_dict.update( self.build_feature_set_2( dataset_x=window_data_x, dataset_y=window_data_y, dataset_z=window_data_z,
                                                                target_value=target_value,col_name='mag' ) )
                self.feature_list.append( features_dict )
                if self.sliding_window_x.reached_end_of_list(): break
        return pd.DataFrame( self.feature_list )

    @staticmethod
    def get_target_value(dataset):
        return float( statistics.mean( data=dataset ) )

    def build_feature_set_2(self,dataset_x, dataset_y, dataset_z, target_value,col_name):
        dataset = []
        for i in range(0,len(dataset_x)):
            dataset.append(math.sqrt( pow(dataset_x[i],2) + pow(dataset_y[i],2) + pow(dataset_z[i],2)))

        # perform FFT and extract dominating frequency
        fft_magnitude = abs( rfft( dataset ) )
        fft_positive_freq_bin = fftfreq( self.window_size, 1 / (self.cutoff_freq_hz * 2) )[0:(int)( self.window_size / 2 )]
        fft_result = dict( zip( fft_positive_freq_bin, fft_magnitude ) )

        # remove dc component
        fft_result.pop( 0 )

        sorted_fft_result = {k: v for k, v in sorted( fft_result.items(), key=lambda item: item[1], reverse=True )}
        dominating_freq_list = np.array( list( sorted_fft_result.keys() ) )
        dominating_norm_freq_mag_list = np.array( list( sorted_fft_result.values() ) )
        dominating_norm_freq_mag_list = dominating_norm_freq_mag_list / dominating_norm_freq_mag_list.max()

        features = {
            col_name + "_mean": statistics.mean( data=dataset ),
            col_name + "_var": statistics.variance( data=dataset ),
            col_name + "_sd": statistics.stdev( data=dataset ),
            col_name + "_len": pow(max( dataset )-min( dataset ),0.25),
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
            'target': target_value
        }
        return features

    def build_feature_set(self, dataset, col_name, target_value):
        # perform FFT and extract dominating frequency
        fft_magnitude = abs( rfft( dataset ) )
        fft_positive_freq_bin = fftfreq( self.window_size, 1 / (self.cutoff_freq_hz * 2) )[0:(int)( self.window_size / 2 )]
        fft_result = dict( zip( fft_positive_freq_bin, fft_magnitude ) )

        # remove dc component
        # fft_result.pop( 0 )

        sorted_fft_result = {k: v for k, v in sorted( fft_result.items(), key=lambda item: item[1], reverse=True )}
        dominating_freq_list = np.array( list( sorted_fft_result.keys() ) )
        dominating_norm_freq_mag_list = np.array( list( sorted_fft_result.values() ) )
        dominating_norm_freq_mag_list = dominating_norm_freq_mag_list / dominating_norm_freq_mag_list.max()

        features = {
            col_name + "_mean": statistics.mean( data=dataset ),
            col_name + "_var": statistics.variance( data=dataset ),
            col_name + "_sd": statistics.stdev( data=dataset ),
            # col_name + "_diff": pow(max( dataset )-min( dataset ),0.25),
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
            'target': target_value
        }
        return features


class HumanMotionRegressor:
    def __init__(self):
        self.feature_df_list = []
        self.normalized_input_vector = []
        self.normalized_output_vector = []

    def regressor(self):
        feature_vector = pd.concat( self.feature_df_list )


        # internal test
        test_feature = self.feature_df_list[2]

        # reindex
        feature_vector.reset_index( drop=True, inplace=True )

        # normalize input and output
        normalizer = preprocessing.Normalizer().fit( feature_vector.drop( ['target'], axis=1) )
        # normalizer = StandardScaler()
        normalizer.fit( feature_vector.drop( ['target'], axis=1 ) )
        self.normalized_input_vector = normalizer.transform( feature_vector.drop( ['target'], axis=1 ) )
        self.normalized_output_vector = feature_vector['target']
        #
        # split the feature vector into train and test data
        x_train, x_test, y_train, y_test = train_test_split( self.normalized_input_vector, self.normalized_output_vector, test_size=0.25, random_state=27 )

        # MLP classifier instance
        mlp_regressor = MLPRegressor( hidden_layer_sizes=(10, 10, 10), activation='relu', solver='sgd', max_iter=1000, verbose=10 )

        # start training
        mlp_regressor.fit( x_train, y_train )

        # start testing
        y_pred = mlp_regressor.predict( x_test )

        t_pred = mlp_regressor.predict( test_feature.drop( ['target'], axis=1 ) )
        t_org = test_feature['target']

        # publish test statistics
        print( mlp_regressor.score( x_test, y_test ) )
        print( f' t_org: mean={statistics.mean( t_org )}, var={statistics.stdev( t_org )}' )
        print( f' t_pred: mean={statistics.mean( t_pred )}, var={statistics.stdev( t_pred )}' )
        fig, ax = plt.subplots( 2, 1 )
        ax[0].plot( range( 0, len( t_org ) ), t_org, 'b--', label='real' )
        ax[1].plot( range( 0, len( t_pred ) ), t_pred, 'r--', label='NN Prediction' )
        plt.show()

        # fig = plt.figure()
        # ax1 = fig.add_subplot( 111 )
        # ax1.plot( range( 0, len( y_test ) ), y_test, 'b--', label='real' )
        # ax1.plot( range( 0, len( y_pred ) ), y_pred, 'r--', label='NN Prediction' )
        # plt.show()

    def add_feature_set(self, dataframe):
        self.feature_df_list.append( dataframe )


if __name__ == '__main__':
    WINDOW_SIZE = 128
    OVERLAP_SIZE = 64
    CUTOFF_FREQ = 5
    SAMPLE_FREQ = 100
    CONFIG = '1'
    # get dataset from csv in the form of dataframe
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
    hm_clf = HumanMotionRegressor()
    hm_clf.add_feature_set( HumanMotionRegressorFeatures( dataframe=she_slow_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g', 'groundVelocity'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                          window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features() )
    hm_clf.add_feature_set( HumanMotionRegressorFeatures( dataframe=she_norm_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g', 'groundVelocity'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                          window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features() )
    # hm_clf.add_feature_set( HumanMotionRegressorFeatures( dataframe=she_fast_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g', 'groundVelocity'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
    #                                                       window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features() )
    hm_clf.add_feature_set( HumanMotionRegressorFeatures( dataframe=she_jog_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g', 'groundVelocity'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                          window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features() )
    hm_clf.add_feature_set( HumanMotionRegressorFeatures( dataframe=she_run_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g', 'groundVelocity'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                          window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features() )
    hm_clf.add_feature_set( HumanMotionRegressorFeatures( dataframe=she_stand_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g', 'groundVelocity'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                          window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features() )
    hm_clf.add_feature_set( HumanMotionRegressorFeatures( dataframe=shan_slow_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g', 'groundVelocity'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                          window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features() )
    hm_clf.add_feature_set( HumanMotionRegressorFeatures( dataframe=shan_norm_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g', 'groundVelocity'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                          window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features() )
    # hm_clf.add_feature_set( HumanMotionRegressorFeatures( dataframe=shan_fast_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g', 'groundVelocity'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
    #                                                       window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features() )
    hm_clf.add_feature_set( HumanMotionRegressorFeatures( dataframe=shan_jog_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g', 'groundVelocity'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                          window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features() )
    hm_clf.add_feature_set( HumanMotionRegressorFeatures( dataframe=shan_run_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g', 'groundVelocity'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                          window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features() )
    hm_clf.add_feature_set( HumanMotionRegressorFeatures( dataframe=shan_stand_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g', 'groundVelocity'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                          window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features() )

    #  classify
    hm_clf.regressor()
