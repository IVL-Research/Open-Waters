import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import glob


class TimeSeriesConstrictor:
    """
    A class to contain a pandas dataframe along with functions to modify the data in it.
    If the class methods are used, the original data in the dataframe is always kept as it is.
    Instead, new columns are created with the modified data along with how they were created.
    """

    def __init__(self, dataframe=None, metadata=None, description=None):
        if isinstance(dataframe, type(pd.DataFrame())):
            self.dataframe = dataframe
        if isinstance(metadata, type(dict())):
            self.metadata = metadata
        else:
            self.metadata = dict()
        if isinstance(description, type(dict())):
            self.description = description
        else:
            self.description = dict()

    def plot(self, y_column="all", default_mode="lines", **kwargs):
        """
        Returns interactive plot.
        y_column is the column name in self.dataframe,
        can be either a string or a list of strings.

        """
        fig = go.Figure()

        if y_column == "all":
            y_column = [str(col) for col in self.dataframe.columns]
        elif not isinstance(y_column, type([])):
            y_column = [y_column]
        for col in y_column:
            marker_dict = dict()
            temp_data = self.dataframe[col].dropna()
            mode = default_mode
            try:
                mode = self.metadata[col]["plot_mode"]
                marker_dict["symbol"] = self.metadata[col]["plot_markers"]
            except:
                pass
            fig.add_trace(
                go.Scatter(
                    x=temp_data.index,
                    y=temp_data,
                    name=col,
                    mode=mode,
                    marker=marker_dict,
                )
            )
        fig.update_layout(showlegend=True)

        return fig

    def outlier_detection(
        self,
        target_column,
        outlier_dist="robusst",
        window_size=3,
        median_lim=1,
        mode="run",
        output_column_name="preprocessed",
        outlier_column_name="outliers",
        **kwargs
    ):
        """
        Find outliers.
        Creates one dataframe column called "outliers" where outliers are
        kept and one dataframe column called "preprocessed" where outliers are removed.
        """

        # Constants should always be stated at the top and explained
        CONSTANT_1 = (
            0.675  # Standard distribution constant, portion of data within 1 STD.
        )

        # Create name for column
        new_column = self.create_target_column(outlier_column_name)
        new_column_2 = self.create_target_column(output_column_name)

        # Create metadata dictionary for outliers
        self.metadata[new_column] = dict()
        self.metadata[new_column]["method"] = "outlier_detection"
        self.metadata[new_column]["used_data_column"] = target_column
        self.metadata[new_column]["outlier_dist"] = outlier_dist
        self.metadata[new_column]["window_size"] = window_size
        self.metadata[new_column]["median_lim"] = median_lim
        self.metadata[new_column]["mode"] = mode
        self.metadata[new_column]["plot_mode"] = "markers"
        self.metadata[new_column]["plot_markers"] = "circle-open"

        self.description[new_column] = dict()
        self.description[new_column]["info"] = (
            "Data where all NON-outliers of " + target_column + " are set to nan"
        )

        # Create metadata dictionary for column where outliers have been removed
        self.metadata[new_column_2] = dict()
        self.metadata[new_column_2]["method"] = "outlier_detection"
        self.metadata[new_column_2]["used_data_column"] = target_column
        self.metadata[new_column_2]["outlier_dist"] = outlier_dist
        self.metadata[new_column_2]["window_size"] = window_size
        self.metadata[new_column_2]["median_lim"] = median_lim
        self.metadata[new_column_2]["mode"] = mode

        self.description[new_column_2] = dict()
        self.description[new_column_2]["info"] = (
            "Data where outliers of " + target_column + " are set to nan"
        )

        # Create temporary dataframe to use only inside this method
        outlier_detection_temp_df = pd.DataFrame()

        # Median of previous (window_size) values
        val = (
            self.dataframe[target_column]
            .rolling(window_size)
            .median(**kwargs)
            .shift(1, axis=0)
        )

        if outlier_dist == "robust":
            # Mean absolute deviation
            mad_val = self.dataframe[target_column].mad(**kwargs)

            # Compare value with val
            outlier_detection_temp_df["Test"] = CONSTANT_1 * (
                (self.dataframe[target_column] - val).abs() / mad_val
            )
        elif outlier_dist == "gaussian":
            # Standard deviation
            dataStd = self.dataframe[target_column].std(**kwargs)

            # Compare value with val
            outlier_detection_temp_df["Test"] = (
                self.dataframe[target_column] - val
            ).abs() / dataStd
        else:
            print("No valid outlier_dist specified, doing nothing")
            outlier_detection_temp_df["Test"] = np.nan
            self.metadata[new_column]["outlier_dist"] = "None"
        if mode == "run":
            # Create column where 0="not outlier", 1="outlier"
            outlier_detection_temp_df["anomalyVec"] = np.zeros(len(self.dataframe))
            outlier_detection_temp_df.loc[
                outlier_detection_temp_df["Test"] > median_lim, "anomalyVec"
            ] = 1

            # Create own column with outliers
            self.dataframe[new_column] = self.dataframe[
                target_column
            ] * outlier_detection_temp_df["anomalyVec"].replace(0, np.nan)

            # The number of data points that were not possible to check with current test due to Nans inside the moving window.
            # Nans in original data are exclude from the sum.
            self.metadata[new_column]["NonCheckedData"] = dict()
            self.metadata[new_column]["NonCheckedData"] = (
                outlier_detection_temp_df["Test"].isna().sum()
                - self.dataframe[target_column].isna().sum()
            )
            self.description[new_column][
                "NonCheckedData"
            ] = "number of data points that were not possible to \
            evaluate with the current test due to Nans inside the moving window. Nans in original data are exclude from the sum."
            self.metadata[new_column]["AnomalyProportion"] = (
                outlier_detection_temp_df["anomalyVec"].sum()
                / outlier_detection_temp_df["anomalyVec"].count()
            )

            # Create column where outliers have been removed from the target data
            self.dataframe[new_column_2] = self.dataframe[target_column].loc[
                outlier_detection_temp_df["anomalyVec"] < 0.5
            ]

    def read_excel(self, path, index_col=0, **kwargs):
        self.dataframe = pd.read_excel(path, index_col=index_col, **kwargs)

    def read_csv(self, path, index_col=0, **kwargs):
        self.dataframe = pd.read_csv(path, index_col=index_col, **kwargs)

    def find_frozen_values(
        self,
        target_column,
        window_size=3,
        var_lim_low=4,
        mode="run",
        output_column_name="preprocessed",
        outlier_column_name="frozen_values",
        **kwargs
    ):
        """
        Find frozen values.
        Creates one dataframe column called "frozen_values" where data points that are frozen
        are kept and one dataframe column called "preprocessed" where frozen data points
        are removed.
        """

        # Create name for column
        new_column = self.create_target_column(outlier_column_name)
        new_column_2 = self.create_target_column(output_column_name)

        # Create metadata dictionary for frozen values
        self.metadata[new_column] = dict()
        self.metadata[new_column]["method"] = "find_frozen_values"
        self.metadata[new_column]["used_data_column"] = target_column
        self.metadata[new_column]["window_size"] = window_size
        self.metadata[new_column]["var_lim_low"] = var_lim_low
        self.metadata[new_column]["mode"] = mode
        self.metadata[new_column]["plot_mode"] = "markers"
        self.metadata[new_column]["plot_markers"] = "circle-open"

        self.description[new_column] = dict()
        self.description[new_column]["info"] = (
            "Data where all NON-frozen values of " + target_column + " are set to nan"
        )

        # Create metadata dictionary for column where frozen values have been removed from target data
        self.metadata[new_column_2] = dict()
        self.metadata[new_column_2]["method"] = "find_frozen_values"
        self.metadata[new_column_2]["used_data_column"] = target_column
        self.metadata[new_column_2]["window_size"] = window_size
        self.metadata[new_column_2]["var_lim_low"] = var_lim_low
        self.metadata[new_column_2]["mode"] = mode

        self.description[new_column_2] = dict()
        self.description[new_column_2]["info"] = (
            "Data where frozen values of " + target_column + " are set to nan"
        )

        # create temporary dataframe
        frozen_values_temp_df = pd.DataFrame()

        # calc standard deviation of data within the window
        frozen_values_temp_df["Test"] = (
            self.dataframe[target_column].rolling(window_size).std(**kwargs)
        )

        # create column where 0="not frozen value", 1="frozen value"
        frozen_values_temp_df["anomalyVec"] = 0
        frozen_values_temp_df.loc[
            frozen_values_temp_df["Test"] < var_lim_low, "anomalyVec"
        ] = 1

        # Process entire window as frozen
        frozen_values_temp_df["anomalyVec"] = (
            frozen_values_temp_df["anomalyVec"]
            .shift(-window_size + 1, fill_value=0)
            .rolling(window_size)
            .max(**kwargs)
        )

        # first number of data points becomes nan due to window size, these are not detected as frozen and set to 0 here
        frozen_values_temp_df["anomalyVec"][0 : window_size - 1] = 0

        # Add frozen values to new column
        self.dataframe[new_column] = (
            self.dataframe[target_column] * frozen_values_temp_df["anomalyVec"]
        ).replace(0, np.nan)

        # Add column with removed frozen values from raw data
        self.dataframe[new_column_2] = self.dataframe[target_column].loc[
            frozen_values_temp_df["anomalyVec"] < 0.5
        ]

    def out_of_range_detection(self, target_column, min_limit, max_limit):
        """
        Find values outside/inside specified limits.
        Creates one dataframe column called "out_of_range" where data points that are out of range
        are kept and one dataframe column called "preprocessed" where data points that are out of range
        are removed.
        """
        # Create name for column
        new_column = self.create_target_column("out_of_range")
        new_column_2 = self.create_target_column("preprocessed")

        # Create metadata dictionary
        self.metadata[new_column] = dict()
        self.metadata[new_column]["method"] = "out_of_range_detection"
        self.metadata[new_column]["used_data_column"] = target_column
        self.metadata[new_column]["min_limit"] = min_limit
        self.metadata[new_column]["max_limit"] = max_limit
        self.metadata[new_column]["plot_mode"] = "markers"
        self.metadata[new_column]["plot_markers"] = "circle-open"

        self.description[new_column] = dict()
        self.description[new_column]["info"] = (
            "Data where all NON-out of range values of "
            + target_column
            + " are set to nan"
        )

        # Create metadata dictionary
        self.metadata[new_column_2] = dict()
        self.metadata[new_column_2]["method"] = "out_of_range_detection"
        self.metadata[new_column_2]["used_data_column"] = target_column
        self.metadata[new_column_2]["min_limit"] = min_limit
        self.metadata[new_column_2]["max_limit"] = max_limit
        self.description[new_column_2] = dict()
        self.description[new_column_2]["info"] = (
            "Data where out of range values of " + target_column + " are set to nan"
        )
        temp_df = pd.DataFrame()
        temp_df["Time"] = self.dataframe.index
        temp_df = temp_df.set_index("Time")
        temp_df["out_of_range_binary"] = 0
        temp_df["out_of_range_binary"].loc[
            (self.dataframe[target_column] > max_limit)
        ] = 1
        temp_df["out_of_range_binary"].loc[
            (self.dataframe[target_column] < min_limit)
        ] = 1

        self.dataframe[new_column] = (
            temp_df["out_of_range_binary"] * self.dataframe[target_column]
        ).replace(0, np.nan)

        self.dataframe[new_column_2] = self.dataframe[target_column][
            (self.dataframe[target_column] < max_limit)
            & (self.dataframe[target_column] > min_limit)
        ]

    def write_to_excel(self, file_name):
        """
        Store data, metadata and descriptions to excel file
        """
        # create common write for all sheets
        writer = pd.ExcelWriter(file_name, engine="xlsxwriter")

        # write data to sheet "data"
        self.dataframe.to_excel(writer, sheet_name="data")

        # convert metadata dict to dataframe and write to sheet "metadata"
        md_df = pd.DataFrame.from_dict(self.metadata)
        md_df.to_excel(writer, sheet_name="metadata")

        # convert description dict to dataframe and write to sheet "description"
        dscr_df = pd.DataFrame.from_dict(self.description)
        dscr_df.to_excel(writer, sheet_name="description")

        # store file
        writer.save()

    def create_target_column(self, target_column):
        """
        When creating a new column, always call this function first
        like: my_target_column = self.create_target_column(my_target_column)
        It checks if it already exists a column with that name in the
        dataframe, and if it does, adds numbering (_2 / _3 / ...)
        """
        if target_column in self.dataframe.columns:
            new_column_name = target_column + "_2"
            count = 2
            while new_column_name in self.dataframe.columns:
                count = count + 1
                new_column_name = target_column + "_" + str(count)
        else:
            new_column_name = target_column
        return new_column_name

    def read_acurve_excel_export(self, excel_path):
        """
        Read excel file exported from acurve. Stores the data
        in self.dataframe and self.metadata
        """
        df = pd.read_excel(excel_path, header=None)
        df_list = []
        md_dict = dict()
        for i in range(0, len(df.columns), 3):
            temp_df = df[df.columns[i : i + 3]]
            md_dict = self.extract_acurve_excel_metadata(temp_df, md_dict)
            df_temp = self.extract_acurve_excel_data(temp_df)
            df_list.append(df_temp)
        df_merged = df_list[0]
        for df in df_list[1:]:
            df_merged = pd.merge(df_merged, df, on="Time", how="outer")
        df_merged = df_merged.sort_index()

        self.dataframe = df_merged
        self.metadata = md_dict

    def extract_acurve_excel_data(self, dataframe):
        """
        Read data of excel exports from acurve
        """
        N_ROWS_METADATA = 10
        data_name = dataframe[dataframe.columns[2]][0]
        dataframe = dataframe[N_ROWS_METADATA + 1 :]
        dataframe = dataframe.set_index(dataframe.columns[0])
        dataframe.columns = ["nan", data_name]
        dataframe = dataframe.drop(dataframe.columns[0], axis=1)
        dataframe.index = pd.to_datetime(dataframe.index, format="%Y-%M-%D %HH%MM")
        dataframe.index.name = "Time"
        return dataframe

    def extract_acurve_excel_metadata(self, dataframe, md_dict):
        """
        Read metadata of excel exports from acurve
        """
        N_ROWS_METADATA = 10
        data_name = dataframe[dataframe.columns[2]][0]
        col1 = dataframe.columns[0]
        col2 = dataframe.columns[2]
        md_dict[data_name] = dict()
        for idx, data in enumerate(dataframe[col1][0:N_ROWS_METADATA]):
            md_dict[data_name][data] = str(dataframe[col2][idx])
        return md_dict
