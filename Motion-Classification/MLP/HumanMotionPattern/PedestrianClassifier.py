from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from window_slider import Slider
from MLP.FirFilter import FirFilter
import statistics
from sklearn import preprocessing
from scipy.fft import rfft, fftfreq
import seaborn as sn

# Import default_timer to compute durations
from timeit import default_timer as timer

start_time = timer()

# Importing numpy
import numpy as np

# Importing Scipy

# Importing Pandas Library
import pandas as pd

# import glob function to scrap files path
from glob import glob




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


    # IMPORTING RAWDATA
    ####################### Scraping RawData files paths########################
    Raw_data_paths = sorted(glob("../../Data/Original-Data/Raw-Data/*"))

    ################# Just to verify if all paths were scraped #################

    # Selecting acc file paths only
    Raw_acc_paths = Raw_data_paths[0:15]

    # Selecting gyro file paths only
    Raw_gyro_paths = Raw_data_paths[15:30]

    # printing info related to acc and gyro files
    print(("RawData folder contains in total {:d} file ").format(len(Raw_data_paths)))
    print(("The first {:d} are Acceleration files:").format(len(Raw_acc_paths)))
    print(("The second {:d} are Gyroscope files:").format(len(Raw_gyro_paths)))
    print("The last file is a labels file")

    # printing 'labels.txt' path
    print("labels file path is:", Raw_data_paths[30])


    # Define import acc and gyro files function

    #    FUNCTION: import_raw_signals(path,columns)
    #    ###################################################################
    #    #           1- Import acc or gyro file                            #
    #    #           2- convert from txt format to float format            #
    #    #           3- convert to a dataframe & insert column names       #
    #    ###################################################################

    def import_raw_signals(file_path, columns):
        ######################################################################################
        # Inputs:                                                                            #
        #   file_path: A string contains the path of the "acc" or "gyro" txt file            #
        #   columns: A list of strings contains the column names in order.                   #
        # Outputs:                                                                           #
        #   dataframe: A pandas Dataframe contains "acc" or "gyro" data in a float format    #
        #             with columns names.                                                    #
        ######################################################################################

        # open the txt file
        opened_file = open(file_path, 'r')

        # Create a list
        opened_file_list = []

        # loop over each line in the opened_file
        # convert each element from txt format to float
        # store each raw in a list
        for line in opened_file:
            opened_file_list.append([float(element) for element in line.split()])

        # convert the list of lists into 2D numpy array(computationally efficient)
        # data=np.array(opened_file_list)

        # Create a pandas dataframe from this 2D numpy array with column names
        data_frame = pd.DataFrame(data=opened_file_list, columns=columns)

        # return the data frame
        return data_frame


    # Importing Files and Storing DataFrames in raw_dic

    ########################################### RAWDATA DICTIONARY ##############################################################

    # creating an empty dictionary where all dataframes will be stored
    raw_dic = {}

    # creating list contains columns names of an acc file
    raw_acc_columns = ['acc_X', 'acc_Y', 'acc_Z']

    # creating list contains gyro files columns names
    raw_gyro_columns = ['gyro_X', 'gyro_Y', 'gyro_Z']


    # Define Import_labels_file function
    #    FUNCTION: import_raw_labels_file(path,columns)
    #    #######################################################################
    #    #      1- Import labels.txt                                           #
    #    #      2- convert data from txt format to int                         #
    #    #      3- convert integer data to a dataframe & insert columns names  #
    #    #######################################################################
    def import_labels_file(path, columns):
        ######################################################################################
        # Inputs:                                                                            #
        #   path: A string contains the path of "labels.txt"                                 #
        #   columns: A list of strings contains the columns names in order.                  #
        # Outputs:                                                                           #
        #   dataframe: A pandas Dataframe contains labels  data in int format                #
        #             with columns names.                                                    #
        ######################################################################################

        # open the txt file
        labels_file = open(path, 'r')

        # creating a list
        labels_file_list = []

        # Store each row in a list ,convert its list elements to int type
        for line in labels_file:
            labels_file_list.append([int(element) for element in line.split()])
        # convert the list of lists into 2D numpy array
        data = np.array(labels_file_list)

        # Create a pandas dataframe from this 2D numpy array with column names
        data_frame = pd.DataFrame(data=data, columns=columns)

        # returning the labels dataframe
        return data_frame


    #################################
    # creating a list contains columns names of "labels.txt" in order
    raw_labels_columns = ['experiment_number_ID', 'user_number_ID', 'activity_number_ID', 'Label_start_point',
                          'Label_end_point']

    # The path of "labels.txt" is last element in the list called "Raw_data_paths"
    labels_path = Raw_data_paths[-1]

    # apply the function defined above to labels.txt
    # store the output  in a dataframe
    Labels_Data_Frame = import_labels_file(labels_path, raw_labels_columns)
    # print(Labels_Data_Frame)

    # loop for to convert  each "acc file" into data frame of floats and store it in a dictionnary.
    for path_index in range(0, 15):
        # extracting the file name only and use it as key:[expXX_userXX] without "acc" or "gyro"
        key = Raw_data_paths[path_index][-16:-4]

        raw_file_name = Raw_data_paths[path_index]
        # get user id and exp id from the file name
        exp_id, user_id = key.split('_')
        exp_id = str(int(exp_id[3:5]))
        user_id = str(int(user_id[4:6]))
        label_entry_count = path_index * 3
        standing_start = Labels_Data_Frame["Label_start_point"][label_entry_count]
        standing_end = Labels_Data_Frame["Label_end_point"][label_entry_count]
        walking_start = Labels_Data_Frame["Label_start_point"][label_entry_count + 1]
        walking_end = Labels_Data_Frame["Label_end_point"][label_entry_count + 1]
        jogging_start = Labels_Data_Frame["Label_start_point"][label_entry_count + 2]
        jogging_end = Labels_Data_Frame["Label_end_point"][label_entry_count + 2]

        # Applying the function defined above to one acc_file and store the output in a DataFrame
        raw_acc_data_frame = import_raw_signals(Raw_data_paths[path_index], raw_acc_columns)
        raw_acc_data_frame_list = list()

        raw_acc_data_frame_stand_df = raw_acc_data_frame.iloc[standing_start:standing_end + 1]
        raw_acc_data_frame_stand_df.insert(0, "activity", ['1'] * len(raw_acc_data_frame_stand_df), True)
        raw_acc_data_frame_list.extend([i for i in raw_acc_data_frame_stand_df.values.tolist()])

        raw_acc_data_frame_walk_df = raw_acc_data_frame.iloc[walking_start:walking_end + 1]
        raw_acc_data_frame_walk_df.insert(0, "activity", ['2'] * len(raw_acc_data_frame_walk_df), True)
        raw_acc_data_frame_list.extend(i for i in raw_acc_data_frame_walk_df.values.tolist())

        raw_acc_data_frame_jog_df = raw_acc_data_frame.iloc[jogging_start:jogging_end + 1]
        raw_acc_data_frame_jog_df.insert(0, "activity", ['3'] * len(raw_acc_data_frame_jog_df), True)
        raw_acc_data_frame_list.extend(i for i in raw_acc_data_frame_jog_df.values.tolist())

        raw_acc_data_frame_new = pd.DataFrame(raw_acc_data_frame_list, columns=['activity', 'acc_x', 'acc_y', 'acc_z'])

        # print(raw_acc_data_frame_new)
        # By shifting the path_index by 15 we find the index of the gyro file related to same experiment_ID
        # Applying the function defined above to one gyro_file and store the output in a DataFrame
        raw_gyro_data_frame = import_raw_signals(Raw_data_paths[path_index + 15], raw_gyro_columns)
        raw_gyro_data_frame_list = list()

        raw_gyro_data_frame_stand_df = raw_gyro_data_frame.iloc[standing_start:standing_end + 1]
        raw_gyro_data_frame_stand_df.insert(0, "activity", ['1'] * len(raw_gyro_data_frame_stand_df), True)
        raw_gyro_data_frame_list.extend([i for i in raw_gyro_data_frame_stand_df.values.tolist()])

        raw_gyro_data_frame_walk_df = raw_gyro_data_frame.iloc[walking_start:walking_end + 1]
        raw_gyro_data_frame_walk_df.insert(0, "activity", ['2'] * len(raw_gyro_data_frame_walk_df), True)
        raw_gyro_data_frame_list.extend(i for i in raw_gyro_data_frame_walk_df.values.tolist())

        raw_gyro_data_frame_jog_df = raw_gyro_data_frame.iloc[jogging_start:jogging_end + 1]
        raw_gyro_data_frame_jog_df.insert(0, "activity", ['3'] * len(raw_gyro_data_frame_jog_df), True)
        raw_gyro_data_frame_list.extend(i for i in raw_gyro_data_frame_jog_df.values.tolist())

        raw_gyro_data_frame_new = pd.DataFrame(raw_gyro_data_frame_list,
                                               columns=['activity_', 'gyro_x', 'gyro_y', 'gyro_z'])
        # print(raw_gyro_data_frame_new)

        # concatenate acc_df and gyro_df in one DataFrame
        raw_signals_data_frame = pd.concat([raw_acc_data_frame_new, raw_gyro_data_frame_new], axis=1).drop(
            ['activity_'], axis=1)

        # print(raw_signals_data_frame)

        # Store this new DataFrame in a raw_dic , with the key extracted above
        raw_dic[key] = raw_signals_data_frame

    # display(raw_dic['exp02_user02'])




    # initialize classifier with feature sets
    hm_clf = HumanMotionClassifier()

    # can add a for loop of 15 experiments to provide data for the feature set generation

    hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=raw_dic['exp01_user01'], axis_names=('acc_x', 'acc_y', 'acc_z'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                           window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features( target_name='walk' ) )
    hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=raw_dic['exp02_user02'], axis_names=('acc_x', 'acc_y', 'acc_z'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,
                                                           window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE ).get_features( target_name='walk' ) )
    # hm_clf.add_feature_set( HumanMotionClassifierFeatures( dataframe=she_fast_walk_df, axis_names=('accx_g', 'accy_g', 'accz_g'), sample_rate_hz=SAMPLE_FREQ, cutoff_freq_hz=CUTOFF_FREQ,

    #  classify
    hm_clf.classify()
