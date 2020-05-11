# Authors:
# Ziqiao Gao 2157371827
# Rui Hu 2350308289
# He Chang 5670527576
# Fanlin Qin 5317973858
#
import numpy as np
import pandas as pd


def preprocessing(filepath, outfilepath):
    dataset = pd.read_csv(filepath, sep=',', header=0)
    """
    # if need to drop row contains any NA value
    # axis can be 0 or index, 1 for column
    # how can be any or all
    # thresh: Require that many non-NA values.
    # subset: Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include.
    dataset.dropna(axis='index', how='any', inplace=True)
    """
    # get the columns' names
    columns = dataset.columns.values
    # set columns type
    columns_types = np.array(['Date', 'Location', 'RealNum', 'RealNum', 'PosNum', 'PosNum', 'PosNum',
                              'Direction', 'PosNum', 'Direction', 'Direction', 'PosNum',
                              'PosNum', 'PosNum', 'PosNum', 'PosNum', 'PosNum',
                              'PosNum', 'PosNum', 'RealNum', 'RealNum', 'YN', 'PosNum',
                              'YN'])

    # directions integer mapping
    directions_all = {np.nan: -1, '': -1, 'N': 1, 'NNE': 2, 'NE': 3, 'ENE': 4, 'E': 5, 'ESE': 6, 'SE': 7, 'SSE': 8, 'S': 9,
                      'SSW': 10, 'SW': 11, 'WSW': 12, 'W': 13, 'WNW': 14, 'NW': 15, 'NNW': 16}

    for column, col_type in zip(columns, columns_types):
        #
        #print(column, col_type)
        #
        # for Date column
        if col_type == 'Date':
            # convert object to numpy datetime64
            dataset[column] = pd.to_datetime(dataset[column], format='%Y/%m/%d')
            # Separate Month and Day
            # Month and Day is useful for classify with FNN
            """
            # same effect as the two lines below
            dataset['Month'] = dataset[column].apply(lambda x: x.month).astype(np.int32)
            dataset['Day'] = dataset[column].apply(lambda x: x.day).astype(np.int32)
            """
            dataset.insert(1, 'Month', dataset[column].apply(lambda x: x.month).astype(np.int32))
            dataset.insert(2, 'Day', dataset[column].apply(lambda x: x.day).astype(np.int32))
        if col_type == 'Location':
            # get locations in dataset
            locations = np.unique(dataset[column].to_numpy())
            #
            #print(locations)
            #
            locations_all = {np.nan: -1}
            # map location to integer, start from 1, 0 reserved for NA
            iteration = 1
            for location in locations:
                locations_all[location] = iteration
                iteration += 1
            #
            #print(locations_all)
            #
            dataset[column] = dataset[column].apply(lambda x: locations_all.get(x))
        if col_type == 'PosNum' or col_type == 'RealNum':
            # if the column is attribute with number value
            missing_value = dataset[column].isna().sum()
            if missing_value > 0:
                print("Column: ", column, "has nan value.")
            """
            Should normalize at first, and then set NA to a proper value????
            """
            if col_type == 'PosNum':
                mean = dataset[column].mean(axis=0, skipna=True)
                stdev = dataset[column].std(axis=0, skipna=True)
                dataset[column] = dataset[column].apply(lambda x: 0 if np.isnan(x) else (x - mean)/stdev)
            elif col_type == 'RealNum':
                mean = dataset[column].mean(axis=0, skipna=True)
                stdev = dataset[column].std(axis=0, skipna=True)
                dataset[column] = dataset[column].apply(lambda x: 0 if np.isnan(x) else (x - mean)/stdev)
        if col_type == 'Direction':
            # map direction to integer
            dataset[column] = dataset[column].apply(lambda x: directions_all.get(x) if x in directions_all.keys() else -1)
        if col_type == 'YN':
            # map Yes to 1, No to 0, NA to -1
            if column == 'RainToday':
                dataset[column] = dataset[column].apply(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else -1))
            elif column == 'RainTomorrow':
                index_invalid_data = dataset[column].index[dataset[column].isna()].values
                #
                #print(index_invalid_data, len(index_invalid_data))
                #
                dataset.drop(index_invalid_data, axis=0, inplace=True)
                dataset[column] = dataset[column].apply(lambda x: 1 if x == 'Yes' else 0)
    dataset.drop(columns=['Date', 'RISK_MM'], inplace=True)
    #
    #print(dataset)
    #
    # don't output index
    dataset.to_csv(outfilepath, sep=',', na_rep='NA', index=False)
    return


if __name__ == '__main__':
    preprocessing('../data/weatherAUS.csv', '../data/weatherAUS_APP_NORM.csv')
