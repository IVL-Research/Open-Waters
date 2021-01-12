# Time Series Constrictor
This class is used to easily manage time series data, based on pandas dataframe.  

Data is stored in a dataframe. Whenever preprocessing methods are used, the idea is that no results should be overwritten but
rather stored in new columns so that what has been done to achieve the data can be reproduced easily.

Metadata and description data contains information about for example parameters for different functions. 
These two dictionaries are written to separate sheets in the excel file when the "write_to_excel"-function
is used.

## Example usage
```
from TimeSeriesConstrictor import TimeSeriesConstrictor
tsc = TimeSeriesConstrictor()

## Data import/expot methods ##

# read data from excel file, data is stored in tsc.dataframe
tsc.read_excel("foo.xlsx")

# import data from previously exported tsc data
tsc.read_tsc_excel("foo.xlsx")

# read data from csv file, data is stored in tsc.dataframe
tsc.read_csv("foo.csv")

# write all data, including metadata and descriptions, to excel
tsc.write_to_excel("my_new_datafile.xlsx")

# write plots of all data including metadata and descriptions, to pptx
tsc.write_summary_pptx("my_result_summary.pptx")

## Pre-processing methods ##

# example usage of the outlier detection method
tsc.outlier_detection("foo") # foo is column name in dataframe

# example usage of the one hot encoder method
tsc.one_hot_encoder("foo") # foo is column name in dataframe

# example usage of the smoothing method
tsc.smoothing("foo") # foo is column name in dataframe

# example usage of the find frozen values method
tsc.find_frozen_values("foo") # foo is column name in dataframe

# example usage of the out of range method
tsc.out_of_range("foo", min_limit=0, max_limit=100) # foo is column name in dataframe

# see result of pre-processing methods, the results are stored in the dataframe as a new column
print(tsc.dataframe)

## Plot methods ##

# interactive plot of all signals, directly in the notebook
tsc.plot()

# interactive plot of all signals, saved to html file
# this is good for large datasets. The html file can be opened in a browser 
# and the same interactive plot can be looked at from there
fig = tsc.plot()
fig.write_html("my_interactive_plot.html")

# static plot of specific signal, optional to save to .png
tsc.plot_static("foo")

# interactive plot to tune defined pre-processing method parameters using sliders.
methods_dict = {'outlier_detection':{'median_lim':{'min':0, 'max':5, 'step':0.05,
                                                        'description': '"Median limit"'},
                                        'window_size':{'min':0, 'max':30, 'step':1,
                                                        'description': '"Window size"'}},
                   'out_of_range_detection':{'min_limit':{'min':-100, 'max':100, 'step':5,
                                                        'description': '"Min limit"'},
                                        'max_limit':{'min':0, 'max':300, 'step':5,
                                                        'description': '"Max limit"'}}}
tsc.parameter_tuning(methods_dict)
```

Side note:
The name Constrictor refers to how this class "squeezes" out information from time series data.
