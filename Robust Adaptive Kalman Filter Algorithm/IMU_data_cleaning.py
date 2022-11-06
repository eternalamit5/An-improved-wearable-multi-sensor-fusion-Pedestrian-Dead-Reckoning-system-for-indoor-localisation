# simple arranging of data as per sensor.csv file
# first delete start 6 lines from .txt file

import pandas as pd

df = pd.read_csv("RawData/staticTestmagnet2.txt", delimiter=",")


titles= list(df.columns)
print(titles)

df.drop(df.columns[[0,2,3,4,8,9,10]], axis=1, inplace=True) # column 1 = index 0

titles= list(df.columns)
print(titles)

df=df[titles]
print(df)

df.to_csv("staticTestmagnet2.csv", index=False,sep=',')


# df = pd.read_csv("HeadingData.csv", delimiter=",")
# yaw = df.Yaw
#
# print (yaw)
#
# import csv
# from collections import defaultdict
#
# columns = defaultdict(list) # each value in each column is appended to a list
#
# with open('HeadingData.csv') as f:
#     reader = csv.DictReader(f) # read rows into a dictionary format
#     for row in reader: # read a row as {column1: value1, column2: value2,...}
#         for (k,v) in row.items(): # go over each column name and value
#             columns[k].append(v) # append the value into the appropriate list
#                                  # based on column name k
#
# yaw=(columns['Yaw'])
# print(yaw)

# in_file = open('HeadingData.csv', 'r')
# in_file.readline()
# for temp_line in in_file.readlines():
#     temp_seq = temp_line.split(',')
#     yaw = float(temp_seq[3])
#     print(yaw)
#
# in_file.close()