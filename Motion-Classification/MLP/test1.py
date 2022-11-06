# Import default_timer to compute durations
from timeit import default_timer as timer
start_time=timer()

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

#IMPORTING RAWDATA
####################### Scraping RawData files paths########################
Raw_data_paths = sorted(glob("../Data/Original-Data/Raw-Data/*"))

################# Just to verify if all paths were scraped #################

# Selecting acc file paths only
Raw_acc_paths=Raw_data_paths[0:15]

# Selecting gyro file paths only
Raw_gyro_paths=Raw_data_paths[15:30]

# printing info related to acc and gyro files
print (("RawData folder contains in total {:d} file ").format(len(Raw_data_paths)))
print (("The first {:d} are Acceleration files:").format(len(Raw_acc_paths)))
print (("The second {:d} are Gyroscope files:").format(len(Raw_gyro_paths)))
print ("The last file is a labels file")

# printing 'labels.txt' path
print ("labels file path is:",Raw_data_paths[30])

#Define import acc and gyro files function

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

#Importing Files and Storing DataFrames in raw_dic

########################################### RAWDATA DICTIONARY ##############################################################

# creating an empty dictionary where all dataframes will be stored
raw_dic = {}

# creating list contains columns names of an acc file
raw_acc_columns = ['acc_X', 'acc_Y', 'acc_Z']

# creating list contains gyro files columns names
raw_gyro_columns = ['gyro_X', 'gyro_Y', 'gyro_Z']

# loop for to convert  each "acc file" into data frame of floats and store it in a dictionnary.
for path_index in range(0, 15):
    # extracting the file name only and use it as key:[expXX_userXX] without "acc" or "gyro"
    key = Raw_data_paths[path_index][-16:-4]

    # Applying the function defined above to one acc_file and store the output in a DataFrame
    raw_acc_data_frame = import_raw_signals(Raw_data_paths[path_index], raw_acc_columns)

    # By shifting the path_index by 15 we find the index of the gyro file related to same experiment_ID
    # Applying the function defined above to one gyro_file and store the output in a DataFrame
    raw_gyro_data_frame = import_raw_signals(Raw_data_paths[path_index + 15], raw_gyro_columns)

    # concatenate acc_df and gyro_df in one DataFrame
    raw_signals_data_frame = pd.concat([raw_acc_data_frame, raw_gyro_data_frame], axis=1)

    # Store this new DataFrame in a raw_dic , with the key extracted above
    raw_dic[key] = raw_signals_data_frame

# raw_dic is a dictionary contains 15 combined DF (acc_df and gyro_df)
print('raw_dic contains %d DataFrame' % len(raw_dic))

# print the first 3 rows of dataframe exp01_user01
display(raw_dic['exp01_user01'])


plt.style.use('bmh') # for better plots

#Define Import_labels_file function
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

#Apply import_labels_file

#################################
# creating a list contains columns names of "labels.txt" in order
raw_labels_columns=['experiment_number_ID','user_number_ID','activity_number_ID','Label_start_point','Label_end_point']

# The path of "labels.txt" is last element in the list called "Raw_data_paths"
labels_path=Raw_data_paths[-1]

# apply the function defined above to labels.txt
# store the output  in a dataframe
Labels_Data_Frame=import_labels_file(labels_path,raw_labels_columns)
#print(Labels_Data_Frame)

# The first 3 rows of labels dataframe
print ("The first 3 rows of  Labels_Data_Frame:" )
display(Labels_Data_Frame.head(3))
display(Labels_Data_Frame)
print(Labels_Data_Frame.shape)

#Define Activity Labels Dic
# Creating a dictionary for all types of activities
# The 3 activities are STANDING, WALKING, JOGGING
Acitivity_labels=AL={
        1: 'STANDING', 2: 'WALKING', 3: 'JOGGING',
       }


#RawData Exploration
# Each acc file and gyro file having the same exp_ID have also the same number of rows

# a list contains the number of rows per dataframe
rows_per_df=[len(raw_dic[key]) for key in sorted(raw_dic.keys())]

# a list contains exp ids
exp_ids=[i for i in range(1,16)]

# useful row is row that was captured while the user was performing an activity
# some rows in acc and gyro files are not associated to an activity id

# list that will contain the number of useful rows per dataframe
useful_rows_per_df=[]

for i in range(1,16):# iterating over exp ids
    # selecting start-end rows of each activity of the experiment
    start_end_df= Labels_Data_Frame[Labels_Data_Frame['experiment_number_ID']==i][['Label_start_point','Label_end_point']]
    # sum of start_labels and sum of end_labels
    start_sum,end_sum=start_end_df.sum()
    # number of rows useful rows in [exp i] dataframe
    useful_rows_number=end_sum-start_sum+len(start_end_df)
    # storing row numbers in a list
    useful_rows_per_df.append(useful_rows_number)

# visualizing row numbers per dataframe

# plt.bar(exp_ids,rows_per_df) # ploting the bar plot
#
# plt.xlabel('experience identifiers(experiment ids)') # Set X axis info
# plt.ylabel('number of rows') # Set Y axis info
# plt.title('number of rows per experience') # Set the title of the bar plot
# plt.show() # Show the figure

#Detrimining Number of rows and Mean time per each activity

# A list will contain number of rows per activity
rows_per_activity = []

# a list will contain the number of times each activity was performed in the protocol of all experiences
count_act = []

for i in range(1, 4):  # iterating over activity ids
    # a dataframe contains start and end labels for all experiences while users were performing the same activity
    start_end_df = Labels_Data_Frame[Labels_Data_Frame['activity_number_ID'] == i][
        ['Label_start_point', 'Label_end_point']]

    # add to the list the number of times this activity was performed in all experiences
    count_act.append(len(start_end_df))

    # start_sum is the sum of all start_label values in start_end_df
    # end_sum is the sum of all end_label values in start_end_df
    start_sum, end_sum = start_end_df.sum()

    # number of rows related to the activity
    number_of_rows = end_sum - start_sum + len(start_end_df)

    # storing number of rows in a list
    rows_per_activity.append(number_of_rows)

# mean duration in seconds of each activity:
time_per_activity = [rows_per_activity[i] / (float(50) * count_act[i]) for i in range(len(rows_per_activity))]

# activity ids from 1 to 3
activity_ids = [i for i in range(1, 4)]

#Detailed Visualizations
# Two full samples:
sample01_01 = raw_dic['exp01_user01'] # acc and gyro signals of exp 01 user 01
sample12_12 = raw_dic['exp12_user12'] # acc and gyro signals of exp 12 user 12

sampling_freq = 100  # 100 Hz(hertz) is sampling frequency: the number of captured values of each axial signal per second.


def visualize_triaxial_signals(data_frame, exp_id, act, sig_type, width, height):
    #################################### INPUTS ####################################################################
    # inputs: Data_frame: Data frame contains acc and gyro signals                                                 #
    #         exp_id: integer from 1 to 15 (the experience identifier)                                             #
    #         width: integer the width of the figure                                                               #
    #         height: integer the height of the figure                                                             #
    #         sig_type: string  'acc' to visualize 3-axial acceleration signals or 'gyro' for 3-axial gyro signals #
    #         act: possible values: string: 'all' (to visualize full signals) ,                                    #
    #              or integer from 1 to 3 to specify the activity id to be visualized                             #
    #                                                                                                              #
    #              if act is from 1 to 6 it will skip the first 250 rows(first 5 seconds) from                     #
    #              the starting point of the activity and will visualize the next 400 rows (next 8 seconds)        #
    #              if act is between 7 and 12  the function will visualize all rows(full duration) of the activity.
    #  if act from 1 to 3 visualize all rows(full duration) of the activity
    #################################################################################################################

    keys = sorted(raw_dic.keys())  # list contains 'expXX_userYY' sorted from 1 to 15
    key = keys[exp_id - 1]  # the key associated to exp_id (experience)
    exp_id = str(exp_id)
    user_id = key[-2:]  # the user id associated to this experience in string format

    if act == 'all':  # to visualize full signal
        # selecting all rows in the dataframe to be visualized , the dataframe stored in raw_dic and has the same key
        data_df = data_frame
    else:  # act is an integer from 1 to 12 (id of the activity to be visualized )
        # Select rows in labels file having the same exp_Id and user_Id mentioned above + the activity id (act)
        # selecting the first result in the search made in labels file
        # and select the start point and end point of this row related to this activity Id (act)
        start_point, end_point = Labels_Data_Frame[
            (Labels_Data_Frame["experiment_number_ID"] == int(exp_id)) &
            (Labels_Data_Frame["user_number_ID"] == int(user_id)) &
            (Labels_Data_Frame["activity_number_ID"] == act)

            ][['Label_start_point', 'Label_end_point']].iloc[0]

        # if act is between 1 and 3 select the full duration of the first result(row)
        data_df = data_frame[start_point:end_point]

    ##################################

    columns = data_df.columns  # a list contain all column names of the  (6 columns in total)

    if sig_type == 'acc':  # if the columns to be visualized are acceleration columns

        # acceleration columns are the first 3 columns acc_X, acc_Y and acc_Z
        X_component = data_df[columns[0]]  # copy acc_X
        Y_component = data_df[columns[1]]  # copy acc_Y
        Z_component = data_df[columns[2]]  # copy acc_Z

        # accelerations legends
        legend_X = 'acc_X'
        legend_Y = 'acc_Y'
        legend_Z = 'acc_Z'

        # the figure y axis info
        figure_Ylabel = 'Acceleration in g'

        # select the right title in each case

        if act == 'all':
            title = "acceleration signals for all activities performed by user " + user_id + ' in experiment ' + exp_id

        elif act in [1, 2, 3]:
            title = "acceleration signals while user " + user_id + ' was performing activity: ' + str(act) + '(' + AL[
                act] + ')'

    elif sig_type == 'gyro':  # if the columns to be visualized are gyro columns

        # gyro columns are the last 3 columns gyro_X, gyro_Y and gyro_Z
        X_component = data_df[columns[3]]  # copy gyro_X
        Y_component = data_df[columns[4]]  # copy gyro_Y
        Z_component = data_df[columns[5]]  # copy gyro_Z

        # gyro signals legends
        legend_X = 'gyro_X'
        legend_Y = 'gyro_Y'
        legend_Z = 'gyro_Z'

        # the figure y axis info
        figure_Ylabel = 'Angular Velocity in radian per second [rad/s]'

        # select the right title in each case
        if act == 'all':
            title = "gyroscope signals for all activities performed by user " + user_id + ' in experiment ' + exp_id
        elif act in [1, 2, 3]:
            title = "gyroscope signals while user " + user_id + ' was performing activity: ' + str(act) + '(' + AL[
                act] + ')'

    # chosing colors : red for X component blue for Y component and green for Z component
    colors = ['r', 'b', 'g']
    len_df = len(data_df)  # number of rows in this dataframe to be visualized(depends on 'act' variable)

    # converting row numbers into time duration (the duration between two rows is 1/50=0.02 second)
    time = [1 / float(sampling_freq) * j for j in range(len_df)]

    # Define the figure and setting dimensions width and height
    fig = plt.figure(figsize=(width, height))

    # ploting each signal component
    _ = plt.plot(time, X_component, color='r', label=legend_X)
    _ = plt.plot(time, Y_component, color='b', label=legend_Y)
    _ = plt.plot(time, Z_component, color='g', label=legend_Z)

    # Set the figure info defined earlier
    _ = plt.ylabel(figure_Ylabel)  # set Y axis info
    _ = plt.xlabel('Time in seconds (s)')  # Set X axis info (same label in all cases)
    _ = plt.title(title)  # Set the title of the figure

    # localise the figure's legends
    _ = plt.legend(loc="upper left")  # upper left corner

    # showing the figure
    plt.show()

#Visualize acc and gyro signals for both samples
################# plotting acc signals for the first sample ######################
# figure parameters : width=18 height=5
# exp_id=1


#                          DataFrame  , exp_Id, act , sig_type  ,Width,height
visualize_triaxial_signals(sample01_01,   1   ,'all',    'acc'  ,  18 ,  5   )
# sig_type='acc' to visulize acceleration signals
# act='all' to visualize full duration of the dataframe

################# plotting gyro signals for the first sample ######################
# figure parameters : width=18 height=5
# exp_id=1
# act='all' to visualize full duration of the dataframe
visualize_triaxial_signals(sample01_01,1,'all','gyro',18,5) # sig_type='gyro' to visualize gyro signals

#Define a look up function to explore labels file
########################FUNCTION: look_up(exp_ID,user_ID,activity_ID)#########################


def look_up(exp_ID, activity_ID):
    ######################################################################################
    # Inputs:                                                                            #
    #   exp_ID  : integer , the experiment Identifier from 1 to 15 (15 included)         #
    #                                                                                    #
    #   activity_ID: integer  the activity Identifier from 1 to 3 (3 included)         #
    # Outputs:                                                                           #
    #   dataframe: A pandas Dataframe which is a part of Labels_Data_Frame contains      #
    #             the activity ID ,the start point  and the end point  of this activity  #
    ######################################################################################
    user_ID = int(sorted(raw_dic.keys())[exp_ID - 1][-2:])
    # To select rows in labels file of a fixed activity in a fixed experiment
    return Labels_Data_Frame[
        (Labels_Data_Frame["experiment_number_ID"] == exp_ID) &
        (Labels_Data_Frame["user_number_ID"] == user_ID) &
        (Labels_Data_Frame["activity_number_ID"] == activity_ID)

        ]

for activity_Id in range(1,4):# iterating throw activity ids from 1 to 3
    # expID=12
    # It returns all Label_start_point and Label_end_point of this (activityID,expID)
    print('Activity number '+str(activity_Id))
    display(look_up(12,activity_Id)) # display the results of each search

#Visualize signals related to Basic Activities for sample N° 2

# visualize  activities from 1 to 3
for act in range(1,4): # Iterating throw each activity Id from 1 to 3
    visualize_triaxial_signals(sample01_01,1,act,'acc',14,2) # visualize acc signals related to this activity
    visualize_triaxial_signals(sample01_01,1,act,'gyro',14,2) # visualize gyro signals reated to this activity

