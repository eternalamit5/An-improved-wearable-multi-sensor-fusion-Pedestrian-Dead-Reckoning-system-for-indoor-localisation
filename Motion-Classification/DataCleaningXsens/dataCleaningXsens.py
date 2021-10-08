

import pandas as pd
Acc_df = pd.read_csv("RawData/running01.txt", delimiter=",")

Gyro_df = pd.read_csv("RawData/running01.txt", delimiter=",")

# print(df)

# step 1 # delete uneccessary column
Acc_df.drop(Acc_df.columns[[0, 1, 5, 6, 7, 8, 9, 10]], axis=1, inplace=True) # column 1 = index 0
Gyro_df.drop(Gyro_df.columns[[0, 1, 2, 3, 4, 8, 9, 10]], axis=1, inplace=True) # column 1 = index 0
# print(df)

# delete first 200 rows to get start time

Acc_df = Acc_df.iloc[200:]
Gyro_df = Gyro_df.iloc[200:]
print("after row deletion")
print(Acc_df)
print(Gyro_df)


# convert acc data into g's
Acc_df['Acc_X'] = (Acc_df['Acc_X'] / 9.80665)
Acc_df['Acc_Y'] = (Acc_df['Acc_Y'] / 9.80665)
Acc_df['Acc_Z'] = (Acc_df['Acc_Z'] / 9.80665)


print("after spliting acc and gyro")
print(Acc_df)
print(Gyro_df)

# remove column name and generate .txt file

Acc_df.to_csv(r'running_acc_Cleaned.txt', header=None, index=None, sep=' ', mode='a')
Gyro_df.to_csv(r'running_gyro_Cleaned.txt', header=None, index=None, sep=' ', mode='a')

# df.to_csv("calibration test1_xsens_Coordinate.csv", index=False, sep=',')