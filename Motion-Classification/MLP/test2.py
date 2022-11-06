# Import default_timer to compute durations
from timeit import default_timer as timer

start_time = timer()

# Importing numpy
import numpy as np

# Importing Scipy
import scipy as sp

# Importing Pandas Library
import pandas as pd

# import glob function to scrap files path
from glob import glob

# import display() for better visualitions of DataFrames and arrays
from IPython.display import display

# import pyplot for plotting
import matplotlib.pyplot as plt

# IMPORTING RAWDATA
####################### Scraping RawData files paths########################
Raw_data_paths = sorted(glob("../Data/Original-Data/Raw-Data/*"))

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
    raw_acc_data_frame_stand_df.insert(0, "activity", ['1'] * len(raw_acc_data_frame_stand_df) , True)
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
    raw_gyro_data_frame_stand_df.insert(0, "activity", ['1'] * len(raw_gyro_data_frame_stand_df) , True)
    raw_gyro_data_frame_list.extend([i for i in raw_gyro_data_frame_stand_df.values.tolist()])

    raw_gyro_data_frame_walk_df = raw_gyro_data_frame.iloc[walking_start:walking_end + 1]
    raw_gyro_data_frame_walk_df.insert(0, "activity", ['2'] * len(raw_gyro_data_frame_walk_df), True)
    raw_gyro_data_frame_list.extend(i for i in raw_gyro_data_frame_walk_df.values.tolist())

    raw_gyro_data_frame_jog_df = raw_gyro_data_frame.iloc[jogging_start:jogging_end + 1]
    raw_gyro_data_frame_jog_df.insert(0, "activity", ['3'] * len(raw_gyro_data_frame_jog_df), True)
    raw_gyro_data_frame_list.extend(i for i in raw_gyro_data_frame_jog_df.values.tolist())

    raw_gyro_data_frame_new = pd.DataFrame(raw_gyro_data_frame_list, columns=['activity_', 'gyro_x', 'gyro_y', 'gyro_z'])
    # print(raw_gyro_data_frame_new)

    # concatenate acc_df and gyro_df in one DataFrame
    raw_signals_data_frame = pd.concat([raw_acc_data_frame_new, raw_gyro_data_frame_new], axis=1).drop(['activity_'], axis=1)

    #print(raw_signals_data_frame)

    # Store this new DataFrame in a raw_dic , with the key extracted above
    raw_dic[key] = raw_signals_data_frame



#display(raw_dic['exp02_user02'])
