# Time Series Constrictor
This class is used to easily manage time series data, based on pandas dataframe.  

Data is stored in a dataframe. Whenever preprocessing are used, the idea is that no results should be overwritten but
rather stored in new columns so that what has been done to achieve the data can be reproduced easily.

Metadata and description data contains information about for example parameters for different functions. 
These two dictionaries are written to separate sheets in the excel file when the "write_to_excel"-function
is used.

## Example usage
```
from TimeSeriesConstrictor import TimeSeriesConstrictor
tsc = TimeSeriesConstrictor()

# read data from excel file, data is stored in tsc.dataframe
tsc.read_excel("foo.xlsx")

# example usage of the outlier detection method (which is under development)
tsc.outlier_detection("foo") # foo is column name in dataframe

# see result of outlier detection, the results are stored in the dataframe as a new column
print(tsc.dataframe)

# interactive plot of all signals
tsc.plot()

# write all data to excel
tsc.write_to_excel("my_new_datafile.xlsx")

```

Side note:
The name Constrictor refers to how this class "squeezes" out information from time series data.
