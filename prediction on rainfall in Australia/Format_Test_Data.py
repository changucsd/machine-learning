# Authors:
# Ziqiao Gao 2157371827
# Rui Hu 2350308289
# He Chang 5670527576
# Fanlin Qin 5317973858
#

import pandas as pd
import numpy as np

folder_path = "../data/test_data/"
file_paths = ["albury-April-2020", "albury-Dec-2019", "bendigo-April-2020", "bendigo-Dec-2019", "moree-Dec-2019", "perth-April-2020"]
suffix = ".csv"
output_file_path = "../data/weatherAUS_TEST.csv"

if __name__ == '__main__':
    file_list = []
    for file_path in file_paths:
        city = file_path.split('-', 3)[0]
        #
        #print(city)
        #
        dataset = pd.read_csv(folder_path + file_path + suffix, sep=',', skiprows=6, skip_blank_lines=True, header=0, encoding='cp1252', engine='python')
        dataset.fillna('NA', inplace=True)
        #
        #print(dataset.columns, dataset.shape)
        #
        dataset.drop(dataset.columns[0], axis=1, inplace=True)
        dataset.rename(columns={"Date": "Date", "Minimum temperature (°C)": "MinTemp",
                                "Maximum temperature (°C)": "MaxTemp", "Rainfall (mm)": "Rainfall",
                                "Evaporation (mm)": "Evaporation", "Sunshine (hours)": "Sunshine",
                                "Direction of maximum wind gust ": "WindGustDir",
                                "Speed of maximum wind gust (km/h)": "WindGustSpeed",
                                "Time of maximum wind gust": "WindGustTime", "9am Temperature (°C)": "Temp9am",
                                "9am relative humidity (%)": "Humidity9am", "9am cloud amount (oktas)": "Cloud9am",
                                "9am wind direction": "WindDir9am", "9am wind speed (km/h)": "WindSpeed9am",
                                "9am MSL pressure (hPa)": "Pressure9am", "3pm Temperature (°C)": "Temp3pm",
                                "3pm relative humidity (%)": "Humidity3pm", "3pm cloud amount (oktas)": "Cloud3pm",
                                "3pm wind direction": "WindDir3pm", "3pm wind speed (km/h)": "WindSpeed3pm",
                                "3pm MSL pressure (hPa)": "Pressure3pm"}, inplace=True)
        #
        #print(dataset.columns, dataset.shape)
        #
        dataset["WindSpeed9am"] = dataset["WindSpeed9am"].apply(lambda x: 0 if x == "Calm" else x)
        dataset.insert(1, 'Location', city)
        #
        #print(dataset.columns, dataset.shape)
        #
        dataset.insert(22, 'RainToday', dataset["Rainfall"].apply(lambda x: 'Yes' if x > 1 else 'No'))
        RainTomorrow = dataset["RainToday"].tolist()
        RainTomorrow.pop(0)
        RainTomorrow.append('NA')
        dataset.insert(23, 'RISK_MM', 0)
        dataset.insert(24, 'RainTomorrow', RainTomorrow)
        dataset = dataset.reindex(['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                         'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm',
                         'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
                         'Temp3pm', 'RainToday', 'RISK_MM', 'RainTomorrow'], axis="columns")
        file_list.append(dataset)

    concat_dataset = pd.concat(file_list, axis=0, ignore_index=True)
    concat_dataset.to_csv(output_file_path, sep=',', na_rep='NA', index=False)
