#!/usr/bin/env python

# __author__ = "Miranda Li"
# __copyright__ = "Copyright 2022, The Cherry Bloom Prediction Project"
# __license__ = "GPL"
# __version__ = "1.0.1"
# __email__ = "miranda.li@daf.qld.gov.au"

import os, sys
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

import matplotlib
matplotlib.style.use('ggplot')


# ************** process data

# replace the following path with the path where you saved washingtondc.csv, kyoto.csv, liestal.csv
data_folder = r"C:\\Users\\lim\\Documents\\Projects\\sideProjects\\CherryBlossomPeakBloomPrediction\\peak-bloom-prediction-main\\data\\"

data = ["washingtondc", "kyoto", "liestal"]

# temperature.csv can be downloaded from the git repository
climate_data = "temperature"

# load cherry data
df_washingtondc = pd.read_csv(data_folder+data[0]+".csv")
df_kyoto = pd.read_csv(data_folder+data[1]+".csv")
df_liestal = pd.read_csv(data_folder+data[2]+".csv")

# climate data
df_temp = pd.read_csv(data_folder+climate_data+".csv")
df_temp1 = df_temp.loc[df_temp['location'].isin(["washingtondc", "kyoto", "liestal", "vancouver"]) & df_temp['season'].isin(["Winter", "Spring"])]


# combine cherry data from 3 sites
df = pd.concat([df_washingtondc, df_kyoto, df_liestal])
print('Min year', df['year'].min())
print('Max year', df['year'].max())

# add Tmax temperature to df
list_tmax_win = []
list_tmax_spr = []
for index, row in df.iterrows():
    locat = row['location']
    yea = row['year']
    ref_row1 = df_temp1.loc[(df_temp1['location'] == locat) & (df_temp1['year'] == yea) & (df_temp1['season'] == "Winter")]
    ref_row2 = df_temp1.loc[(df_temp1['location'] == locat) & (df_temp1['year'] == yea) & (df_temp1['season'] == "Spring")]
    try:
        tm_w = ref_row1['tmax_avg'].values[0]
    except:
        tm_w = np.nan
    try:
        tm_s= ref_row2['tmax_avg'].values[0]
    except:
        tm_s = np.nan
    list_tmax_win.append(tm_w)
    list_tmax_spr.append(tm_s)


df['tmax_avg_winter'] = list_tmax_win
df['tmax_avg_spring'] = list_tmax_spr

# remove nan
df_temperature = df.dropna()

# visualizing the data


# processing data
# convert string to float
df["location"] = df.location.map({"washingtondc": 1, "kyoto": 2, "liestal": 3})


# using data from 1880 onwards
df_sub = df.loc[df["year"] >= 1880]
X = df_sub[["location", "year"]].values
y = df_sub["bloom_doy"].values # since January 1st of the year until peak bloom 

# using data from whole dataset
# X = df[["location", "year"]].values
# y = df["bloom_doy"].values


# # split dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.1, random_state=13
# )

# create test data for year 2022-2031
year_to_predict = [2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031]

locs = []
years = []
for i in range(len(data)):
    for j in range(len(year_to_predict)):
        locs.append(data[i])
        years.append(year_to_predict[j])

data_p = {'location': locs, 'year': years}

df_pred = pd.DataFrame(data_p)
df_pred["location"] = df_pred.location.map({"washingtondc": 1, "kyoto": 2, "liestal": 3})

# vancouver
locs_van = ["vancouver"] * 10
data_van = {'location': locs_van, 'year': year_to_predict}

df_pred_van = pd.DataFrame(data_van)
df_pred_van["location"] = df_pred_van.location.map({"vancouver": 4})

# training data
X_train = X 
y_train = y 

X_test =  df_pred.values
X_test_van = df_pred_van.values


# ************* fit the model

# using location and year to predict blooming date
params = {
    "n_estimators": 400,
    "max_depth": 3,
    "min_samples_split": 8,
    "learning_rate": 0.01,
    "loss": "squared_error"
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)


# prediction
y_hat = reg.predict(X_test)
# y_hat = np.round(y_hat)

pred_date = []
i = 0
for k in X_test[:,1]:
    x = dt.date(k, 1, 1) + dt.timedelta(days=y_hat[i])
    pred_date.append(x.isoformat())
    i = i+1


# Extrapolating to Vancouver, BC
y_hat_van = reg.predict(X_test_van)
# y_hat_van = np.round(y_hat_van)

pred_date_van = []
i = 0
for k in X_test_van[:,1]:
    x = dt.date(k, 1, 1) + dt.timedelta(days=y_hat_van[i])
    pred_date_van.append(x.isoformat())
    i = i+1


p_doy_washingtondc = []
p_doy_kyoto = []
p_doy_liestal = []
p_location = []
p_year = []
p_doy = []
for i in range(len(X_test[:,0])):
    if X_test[i,0] == 1:
        p_location.append("washingtondc")
        p_doy_washingtondc.append(y_hat[i])
    elif X_test[i,0] == 2:
        p_location.append("kyoto")
        p_doy_kyoto.append(y_hat[i])
    elif X_test[i,0] == 3:
        p_location.append("liestal")
        p_doy_liestal.append(y_hat[i])
    p_year.append(X_test[i,1])
    p_doy.append(y_hat[i])
    


data_merged_pred1 = {"year": year_to_predict, "washingtondc": p_doy_washingtondc, "kyoto": p_doy_kyoto, "liestal": p_doy_liestal, "vancouver": y_hat_van}


df_merged_pred1 = pd.DataFrame(data_merged_pred1)
# df_merged_pred1.to_csv(r"C:\Users\lim\Documents\Projects\sideProjects\CherryBlossomPeakBloomPrediction\peak-bloom-prediction-main\submission\pred_GradientBoostingRegressor.csv",  index = False)


# df_gbr_pred = pd.DataFrame({'location':p_location, 'year':p_year, 'bloom_doy':p_doy})
# fig, ax = plt.subplots(figsize=(10,4))
# for key, grp in df_gbr_pred.groupby(['location']):
#     ax.plot(grp['year'], grp['bloom_doy'], label=key)

# ax.set_ylabel("bloom_doy")
# ax.legend()
# # plt.show()
# plt.savefig(r'C:\Users\lim\Documents\Projects\sideProjects\CherryBlossomPeakBloomPrediction\peak-bloom-prediction-main\submission\pic\bloom_doy_pred.png')


# using location, year and temperature to predict bloom date
# step 1: predict temperature for 2022-2031
# print('df_temperature',df_temperature)
# print(df_temperature[["location", "year", "tmax_avg_winter", "tmax_avg_spring"]])
df_temperature["location"] = df_temperature.location.map({"washingtondc": 1, "kyoto": 2, "liestal": 3})

X_train_climate = df_temperature[["location", "year"]].values
y_train_t_winter = df_temperature['tmax_avg_winter'].values
y_train_t_spring = df_temperature['tmax_avg_spring'].values

# predict temperature from 2022-2031
reg = ensemble.GradientBoostingRegressor(**params)
# winter
reg.fit(X_train_climate, y_train_t_winter)

y_hat_c_winter = reg.predict(X_test)
y_hat_c_winter = np.round(y_hat_c_winter)

y_hat_c_van_winter = reg.predict(X_test_van)
y_hat_c_van_winter = np.round(y_hat_c_van_winter)

# spring
reg.fit(X_train_climate, y_train_t_spring)
y_hat_c_spring = reg.predict(X_test)
y_hat_c_spring = np.round(y_hat_c_spring)

y_hat_c_van_spring = reg.predict(X_test_van)
y_hat_c_van_spring = np.round(y_hat_c_van_spring)

# new training data with climate variables
X_train_c = df_temperature[["location", "year", "tmax_avg_winter", "tmax_avg_spring"]]
y_train_c = df_temperature["bloom_doy"].values


data_pc = {'location': locs, 'year': years, 'tmax_avg_winter': y_hat_c_winter, 'tmax_avg_spring': y_hat_c_spring}
df_pred_c = pd.DataFrame(data_pc)
df_pred_c["location"] = df_pred_c.location.map({"washingtondc": 1, "kyoto": 2, "liestal": 3})

X_test_c = df_pred_c.values

# van
df_pred_van['tmax_avg_winter'] = y_hat_c_van_winter
df_pred_van['tmax_avg_spring'] = y_hat_c_van_spring

X_test_van_c = df_pred_van.values


# step 2: fit to model to predict bloom date
reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train_c, y_train_c)

y_hat_c = reg.predict(X_test_c)
# y_hat_c = np.round(y_hat_c)

p_doy_washingtondc = []
p_doy_kyoto = []
p_doy_liestal = []
p_location = []
p_year = []
p_doy = []
for i in range(len(X_test[:,0])):
    if X_test[i,0] == 1:
        p_doy_washingtondc.append(y_hat_c[i])
        p_location.append("washingtondc")
    elif X_test[i,0] == 2:
        p_doy_kyoto.append(y_hat_c[i])
        p_location.append("kyoto")
    elif X_test[i,0] == 3:
        p_doy_liestal.append(y_hat_c[i])
        p_location.append("liestal")
    p_year.append(X_test[i,1])
    p_doy.append(y_hat_c[i])


y_hat_c_van = reg.predict(X_test_van_c)
# y_hat_c_van = np.round(y_hat_c_van)

data_merged_pred2 = {"year": year_to_predict, "washingtondc": p_doy_washingtondc, "kyoto": p_doy_kyoto, "liestal": p_doy_liestal, "vancouver": y_hat_c_van}


df_merged_pred2 = pd.DataFrame(data_merged_pred2)
# df_merged_pred2.to_csv(r"C:\Users\lim\Documents\Projects\sideProjects\CherryBlossomPeakBloomPrediction\peak-bloom-prediction-main\submission\pred_GradientBoostingRegressor_climate.csv",  index = False)


# ********* random forest
regr = RandomForestRegressor(max_depth=3, random_state=0)
regr.fit(X_train_c, y_train_c)

y_hat_c = regr.predict(X_test_c)
# y_hat_c = np.round(y_hat_c)

p_doy_washingtondc = []
p_doy_kyoto = []
p_doy_liestal = []
p_location = []
p_year = []
p_doy = []
for i in range(len(X_test[:,0])):
    if X_test[i,0] == 1:
        p_doy_washingtondc.append(y_hat_c[i])
        p_location.append("washingtondc")
    elif X_test[i,0] == 2:
        p_doy_kyoto.append(y_hat_c[i])
        p_location.append("kyoto")
    elif X_test[i,0] == 3:
        p_doy_liestal.append(y_hat_c[i])
        p_location.append("liestal")
    p_year.append(X_test[i,1])
    p_doy.append(y_hat_c[i])


y_hat_c_van = regr.predict(X_test_van_c)
# y_hat_c_van = np.round(y_hat_c_van)

data_merged_pred3 = {"year": year_to_predict, "washingtondc": p_doy_washingtondc, "kyoto": p_doy_kyoto, "liestal": p_doy_liestal, "vancouver": y_hat_c_van}


df_merged_pred3 = pd.DataFrame(data_merged_pred3)
# df_merged_pred3.to_csv(r"C:\Users\lim\Documents\Projects\sideProjects\CherryBlossomPeakBloomPrediction\peak-bloom-prediction-main\submission\pred_RandomForestRegressor_climate.csv",  index = False)



# ******** LASSO
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train_c, y_train_c)

y_hat_c = clf.predict(X_test_c)
# y_hat_c = np.round(y_hat_c)

p_doy_washingtondc = []
p_doy_kyoto = []
p_doy_liestal = []
p_location = []
p_year = []
p_doy = []
for i in range(len(X_test[:,0])):
    if X_test[i,0] == 1:
        p_doy_washingtondc.append(y_hat_c[i])
        p_location.append("washingtondc")
    elif X_test[i,0] == 2:
        p_doy_kyoto.append(y_hat_c[i])
        p_location.append("kyoto")
    elif X_test[i,0] == 3:
        p_doy_liestal.append(y_hat_c[i])
        p_location.append("liestal")
    p_year.append(X_test[i,1])
    p_doy.append(y_hat_c[i])


y_hat_c_van = regr.predict(X_test_van_c)
# y_hat_c_van = np.round(y_hat_c_van)

data_merged_pred4 = {"year": year_to_predict, "washingtondc": p_doy_washingtondc, "kyoto": p_doy_kyoto, "liestal": p_doy_liestal, "vancouver": y_hat_c_van}


df_merged_pred4 = pd.DataFrame(data_merged_pred4)
# df_merged_pred4.to_csv(r"C:\Users\lim\Documents\Projects\sideProjects\CherryBlossomPeakBloomPrediction\peak-bloom-prediction-main\submission\pred_LASSO_climate.csv",  index = False)


# get average of different models
washingtondc1 = df_merged_pred1["washingtondc"] 
kyoto1 = df_merged_pred1["kyoto"] 
liestal1 = df_merged_pred1["liestal"] 
vancouver1 = df_merged_pred1["vancouver"] 

washingtondc2 = df_merged_pred2["washingtondc"]
kyoto2 = df_merged_pred2["kyoto"]
liestal2 = df_merged_pred2["liestal"]
vancouver2 = df_merged_pred2["vancouver"]

washingtondc3 = df_merged_pred3["washingtondc"]
kyoto3 = df_merged_pred3["kyoto"]
liestal3 = df_merged_pred3["liestal"]
vancouver3 = df_merged_pred3["vancouver"]

washingtondc4 = df_merged_pred4["washingtondc"]
kyoto4 = df_merged_pred4["kyoto"]
liestal4 = df_merged_pred4["liestal"]
vancouver4 = df_merged_pred4["vancouver"]

list_washingtondc = (washingtondc2 + washingtondc3 + washingtondc4)/3
list_kyoto = (kyoto2 + kyoto3 + kyoto4)/3
list_liestal = (liestal2 + liestal3 + liestal4)/3
list_vancouver = ( vancouver2 + vancouver3 + vancouver4)/3


data_merged_pred = {"year": year_to_predict, "washingtondc": np.round(list_washingtondc, 1), "kyoto": np.round(list_kyoto, 1), \
    "liestal": np.round(list_liestal, 1), "vancouver": np.round(list_vancouver, 1)}


df_merged_pred = pd.DataFrame(data_merged_pred)
df_merged_pred.to_csv(r"C:\Users\lim\Documents\Projects\sideProjects\CherryBlossomPeakBloomPrediction\peak-bloom-prediction-main\submission\cherry_predictions.csv",  index = False)

pred_date_dc = []
i = 0
for k in year_to_predict:
    x = dt.date(k, 1, 1) + dt.timedelta(days=list_washingtondc[i])
    pred_date_dc.append(x.isoformat())
    i = i+1
# print('washingtondc', pred_date_dc)

pred_date_kyoto = []
i = 0
for k in year_to_predict:
    x = dt.date(k, 1, 1) + dt.timedelta(days=list_kyoto[i])
    pred_date_kyoto.append(x.isoformat())
    i = i+1
# print('kyoto', pred_date_kyoto)

pred_date_liestal = []
i = 0
for k in year_to_predict:
    x = dt.date(k, 1, 1) + dt.timedelta(days=list_liestal[i])
    pred_date_liestal.append(x.isoformat())
    i = i+1
# print('liestal', pred_date_liestal)

pred_date_vanc = []
i = 0
for k in year_to_predict:
    x = dt.date(k, 1, 1) + dt.timedelta(days=list_vancouver[i])
    pred_date_vanc.append(x.isoformat())
    i = i+1
# print('vancouver', pred_date_vanc)

date_merged = {"year": year_to_predict, "washingtondc": pred_date_dc, "kyoto": pred_date_kyoto, \
    "liestal": pred_date_liestal, "vancouver": pred_date_vanc}

df_merged_date = pd.DataFrame(date_merged)
df_merged_date.to_csv(r"C:\Users\lim\Documents\Projects\sideProjects\CherryBlossomPeakBloomPrediction\peak-bloom-prediction-main\submission\cherry_bloom_dates_predictions.csv",  index = False)
