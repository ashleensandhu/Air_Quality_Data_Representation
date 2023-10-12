import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Loading the dataset.
csv_file = 'https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/air-quality/AirQualityUCI.csv'
df = pd.read_csv(csv_file, sep=';')

# Dropping the 'Unnamed: 15' & 'Unnamed: 16' columns.
df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'], axis=1) 

# Dropping the null values.
df = df.dropna()

# Creating a Pandas series containing 'datetime' objects.
dt_series = pd.Series(data = [item.split("/")[2] + "-" + item.split("/")[1] + "-" + item.split("/")[0] for item in df['Date']], index=df.index) + ' ' + pd.Series(data=[str(item).replace(".", ":") for item in df['Time']], index=df.index)
dt_series = pd.to_datetime(dt_series)

# Remove the Date & Time columns from the DataFrame and insert the 'dt_series' in it.
df = df.drop(columns=['Date', 'Time'], axis=1)
df.insert(loc=0, column='DateTime', value=dt_series)

# Get the Pandas series containing the year values as integers.
year_series = dt_series.dt.year

# Get the Pandas series containing the month values as integers.
month_series = dt_series.dt.month

# Get the Pandas series containing the day values as integers.
day_series = dt_series.dt.day

# Get the Pandas series containing the days of a week, i.e., Monday, Tuesday, Wednesday etc.
day_name_series = dt_series.dt.day_name()

# Add the 'Year', 'Month', 'Day' and 'Day Name' columns to the DataFrame.
df['Year'] = year_series
df['Month'] = month_series
df['Day'] = day_series
df['Day Name'] = day_name_series

# Sort the DataFrame by the 'DateTime' values in the ascending order. Also, display the first 10 rows of the DataFrame.
df = df.sort_values(by='DateTime')

# Create a function to replace the commas with periods in a Pandas series.
def comma_to_period(series):
    new_series = pd.Series(data=[float(str(item).replace(',', '.')) for item in series], index=df.index)
    return new_series

# Apply the 'comma_to_period()' function on the ''CO(GT)', 'C6H6(GT)', 'T', 'RH' and 'AH' columns.
cols_to_correct = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH'] # Create a list of column names.
for col in cols_to_correct: # Iterate through each column
    df[col] = comma_to_period(df[col]) # Replace the original column with the new series.

# Remove all the columns from the 'df' DataFrame containing more than 10% garbage value.
df = df.drop(columns=['NMHC(GT)', 'CO(GT)', 'NOx(GT)', 'NO2(GT)'], axis=1)

# Create a new DataFrame containing records for the years 2004 and 2005.
aq_2004_df = df[df['Year'] == 2004]
aq_2005_df = df[df['Year'] == 2005]

# Replace the -200 value with the median values for each column having indices between 1 and -4 (excluding -4) for the 2004 year DataFrame.
for col in aq_2004_df.columns[1:-4]:
  median = aq_2004_df.loc[aq_2004_df[col] != -200, col].median()
  aq_2004_df[col] = aq_2004_df[col].replace(to_replace=-200, value=median)

# Repeat the same exercise for the 2005 year DataFrame.
for col in aq_2005_df.columns[1:-4]:
  median = aq_2005_df.loc[aq_2005_df[col] != -200, col].median()
  aq_2005_df[col] = aq_2005_df[col].replace(to_replace=-200, value=median)

# Group the DataFrames about the 'Month' column.
group_2004_month = aq_2004_df.groupby(by='Month')
group_2005_month = aq_2005_df.groupby(by='Month')

# Concatenate the two DataFrames for 2004 and 2005 to obtain one DataFrame.
df = pd.concat([aq_2004_df, aq_2005_df])

# Information of the DataFrame.
df.info()

# S1.2: Create a pie chart to display the percentage of data collected in 2004 and 2005 without calculating the percentage values for slices.
years=df['Year'].value_counts()
plt.figure(dpi=108)
plt.title("Percentage of Data Collected in 2004 and 2005")
plt.pie(years, labels=['2004', '2005'], wedgeprops={'edgecolor':'red'}, autopct='%1.2f%%', explode=(0, 0.15), shadow=True)
plt.show()

# S1.6: Get the month names from the 'DateTime' column for each record.
df['DateTime'].dt.month_name()
# S1.7: Add the 'Month Name' column to the 'df' DataFrame and print the first five rows of the updated DataFrame.
df['Month Name'] = df['DateTime'].dt.month_name()
df.head()

months_2005 = df.loc[df["Year"]==2005,'Month Name'].value_counts()
months_2005
months_2004 = df.loc[df["Year"]==2004,'Month Name'].value_counts()
months_2004

# T1.2: Create a pie chart for the 2005 displaying all the months. Label the slices with the month names.
plt.figure(dpi=108)
plt.title("2005 months")
plt.pie(months_2005, labels=months_2005.index, wedgeprops={'edgecolor':'red'}, explode= np.linspace(0, 0.5, 4), autopct='%1.2f%%', startangle=30, shadow=True)
plt.show()
#start angle places the first slice at an angle of whaever angle your provide with respect to the horizontal axis in the anti clockwise direction

# T2.1 Create a NumPy array containing 10,000 random normally distributed numbers having a mean of 165 cm and a standard deviation.
heights = np.random.normal(165, 15, 10000)

# T2.2: Calculate the mean and standard deviation of the normally distributed heights.
print(np.mean(heights))
print(np.std(heights))

# T2.3: Create a histogram for the heights.
plt.figure(figsize=(15, 5))
plt.title("Finding Height")
plt.hist(heights, bins='sturges', edgecolor='black')
plt.xlabel('heights')
plt.ylabel("Number of Observations")
plt.axvline(x=np.mean(heights), color='red', label=f'mean height={np.mean(heights):.2f} cm', linewidth=2)
plt.legend()
plt.show()

# T2.4: Create a bell curve using the 'distplot()' function.
plt.figure(figsize=(15, 5))
plt.title("Bell Curve For Height")
sns.distplot(heights, bins='sturges', hist=False)
plt.xlabel('heights')
plt.ylabel("Number of Observations")
plt.axvline(x=np.mean(heights), color='red', label=f'mean height={np.mean(heights):.2f} cm', linewidth=2)
plt.legend()
plt.grid(which='major', axis='y', color='gray')
plt.show()

# S2.1: Create a bell curve with the vertical lines denoting mean value and the one-sigma interval.
plt.figure(figsize=(15, 5))
plt.title("Bell Curve For Height")
sns.distplot(heights, bins='sturges', hist=False)
plt.ylabel("Probability Density")
plt.axvline(x=np.mean(heights), color='red', label=f'mean height={np.mean(heights):.2f} cm', linewidth=2)
plt.axvline(np.mean(heights)-np.std(heights), color='green', label=f'mu-sigma={np.mean(heights)-np.std(heights):.2f} cm', linewidth=2)
plt.axvline(np.mean(heights)+np.std(heights), color='green', label=f'mu+sigma={np.mean(heights)+np.std(heights):.2f} cm', linewidth=2)
plt.legend()
plt.grid(which='major', axis='y', color='gray')
plt.show()

# S2.3: Calculate the mean and the median height values.
print(np.mean(heights))
print(np.std(heights))

# S2.4: Create a bell curve with the vertical lines denoting mean value and the two-sigma interval.
plt.figure(figsize=(15, 5))
plt.title("Bell Curve For Height")
sns.distplot(heights, bins='sturges', hist=False)
plt.ylabel("Probability Density")
plt.axvline(x=np.mean(heights), color='red', label=f'mean height={np.mean(heights):.2f} cm', linewidth=2)
plt.axvline(np.mean(heights)-(2*np.std(heights)), color='green', label=f'mu-2 sigma={np.mean(heights)-2*np.std(heights):.2f} cm', linewidth=2)
plt.axvline(np.mean(heights)+2*np.std(heights), color='green', label=f'mu+2 sigma={np.mean(heights)+2*np.std(heights):.2f} cm', linewidth=2)
plt.legend()
plt.grid(which='major', axis='y', color='gray')
plt.show()

# S2.6: Create a bell curve with the vertical lines denoting mean value and the three-sigma interval.
plt.figure(figsize=(15, 5))
plt.title("Bell Curve For Height")
sns.distplot(heights, bins='sturges', hist=False)
plt.ylabel("Probability Density")
plt.axvline(x=np.mean(heights), color='red', label=f'mean height={np.mean(heights):.2f} cm', linewidth=2)
plt.axvline(np.mean(heights)-(3*np.std(heights)), color='green', label=f'mu-3 sigma={np.mean(heights)-3*np.std(heights):.2f} cm', linewidth=2)
plt.axvline(np.mean(heights)+3*np.std(heights), color='green', label=f'mu+3 sigma={np.mean(heights)+3*np.std(heights):.2f} cm', linewidth=2)
plt.legend()
plt.grid(which='major', axis='y', color='gray')
plt.show()

# S3.1: Compute the one-sigma interval for the relative humidity values.
plt.figure(figsize=(15, 5))
plt.title("Histogram for RH")
plt.hist(df["RH"], bins='sturges', edgecolor='black')
plt.ylabel("Probability Density")
plt.axvline(x=df["RH"].mean(), color='red', label=f'mean height={df["RH"].mean():.2f} cm', linewidth=2)
plt.axvline(df["RH"].mean()-df["RH"].std(), color='green', label=f'mu-sigma={df["RH"].mean()-df["RH"].std():.2f} cm', linewidth=2)
plt.axvline(df["RH"].mean()+df["RH"].std(), color='green', label=f'mu+sigma={df["RH"].mean()+df["RH"].std():.2f} cm', linewidth=2)
plt.legend()
plt.grid(which='major', axis='y', color='gray')
plt.show()
# S3.2: Create a histogram for relative humidity values and find out whether it follows a bell curve or not.

# T3.1: Create 3 arrays having normally distributed random values. They should have the same length, same mean but different standard deviations.
array1=np.random.normal(150, 10, 10000)
array2=np.random.normal(150, 30, 10000)
array3=np.random.normal(150, 50, 10000)
# T3.2: Create bell curves as well for above three arrays.
plt.figure(figsize=(15, 5), dpi=96)
plt.title("bell Curve")
sns.distplot(array1, hist=False, bins='sturges', label='First Array')
sns.distplot(array2, hist=False, bins='sturges', label='Second Array')
sns.distplot(array3, hist=False, bins='sturges', label='Third Array')
plt.axvline(150, color='red', label=f'mean=150', linewidth=2)
plt.ylabel("probability Density")
plt.legend(loc='upper left')
plt.grid(which='major', axis='y', color='gray')
plt.show()