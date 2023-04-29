import requests
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pandas as pd
import time


scaler = MinMaxScaler()

allcitieswitharea = {}

with open('cities.json') as json_file:
    allcitieswitharea = json.load(json_file)

df_census = pd.read_excel("census_2021.xlsx")

headers = {"Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAANtViAEAAAAAlS1bO4KSV%2BYb23gsmqrWzfG%2Bmno%3DHIa5PsqBGvD1bNGsDexQecaYUM8KwTAWq7nTtvn6J3TrP43IDA"}
api = "https://api.twitter.com/2/tweets/search/recent?max_results=100&tweet.fields=geo&expansions=geo.place_id&place.fields=contained_within,country,country_code,full_name,geo,id,name,place_type&user.fields=entities,name&query="

sid_obj = SentimentIntensityAnalyzer()


def area_density(city_with_areas):
    areas = []
    densitys = []

    for area in city_with_areas:
        densitys.append(
            int(df_census.loc[df_census['Cities'] == area]["Density per sq km"]))
        areas.append(area)

    list_of_tuples = list(zip(areas, densitys))
    df_city = pd.DataFrame(list_of_tuples,
                           columns=[
                               'Area',
                               'Density per sq km'])

    return df_city


def sentiment_analysis(text, df, colname, val, newcol):
    sentiment_dict = sid_obj.polarity_scores(text)
    if sentiment_dict['compound'] >= 0.05:
        df.loc[df.index[df[colname] == val], newcol] = "Positive"
    elif sentiment_dict['compound'] <= - 0.05:
        df.loc[df.index[df[colname] == val], newcol] = "Negative"
    else:
        df.loc[df.index[df[colname] == val], newcol] = "Neutral"

    return None


def predicted_location_possibility(df_city):
    scaler.fit(df_city[['Sentiment Analysis Compound Val']])
    df_city['Sentiment Analysis Compound Val Sacle'] = scaler.transform(
        df_city[['Sentiment Analysis Compound Val']])

    scaler.fit(df_city[['Density per sq km']])
    df_city['Density per sq km Sacle'] = scaler.transform(
        df_city[['Density per sq km']])

    scaler.fit(df_city[['Total Respone']])
    df_city['Total Respone Sacle'] = scaler.transform(
        df_city[['Total Respone']])

    df_city['Sentiment Analysis Compound Val Sacle'] = df_city['Sentiment Analysis Compound Val Sacle'].apply(
        lambda x: x*0.2)
    df_city['Density per sq km Sacle'] = df_city['Density per sq km Sacle'].apply(
        lambda x: x*0.4)
    df_city['Total Respone Sacle'] = df_city['Total Respone Sacle'].apply(
        lambda x: x*0.4)

    df_city['Predicted Location Possibility Sacle'] = (
        df_city['Sentiment Analysis Compound Val Sacle'] + df_city['Density per sq km Sacle'] + df_city['Total Respone Sacle'])/3

    return df_city

def Grouping_K_means(df_city):
    df_area_encode = pd.get_dummies(df_city['Area'])

    df_area_encode = pd.concat([df_area_encode, df_city.drop(
        ['Area', 'Sentiment Analysis'], axis="columns")], axis="columns")
    df_area_encode = df_area_encode.dropna()
    df_area_encode = df_area_encode.reset_index(drop=True)

    km = KMeans(n_clusters=3)
    y_predicted = km.fit_predict(df_area_encode)
    df_area_encode['cluster'] = y_predicted

    recomended_names = []
    df_recomended_names = pd.DataFrame(index=np.arange(100))
    for i in range(3):
        recomended = df_area_encode[df_area_encode.cluster==i]
        recomended = recomended.drop(['cluster','Density per sq km','Total Respone','Sentiment Analysis Compound Val'],axis="columns")
        name = recomended.apply(lambda x: recomended.columns[x==1], axis=1)
        names = []
        for n in name:
            names.append(n.values[0])
        recomended_names.append(names)

    df_recomended_names['group1'] = pd.Series(recomended_names[0])
    df_recomended_names['group2'] = pd.Series(recomended_names[1])
    df_recomended_names['group3'] = pd.Series(recomended_names[2])

    return df_recomended_names


def getalldata(city, categorie):
    print("\n\n", city, "\n")

    totalresp = 0
    df_all_tweets = pd.DataFrame()
    df_geoloc_density = pd.DataFrame()

    city_with_areas = set(allcitieswitharea[city])
    df_city = area_density(city_with_areas)

    for area in city_with_areas:
        totalrespsamegeoid = 0
        query = categorie + ", \"" + area + "\""
        endpoint = api+query
        resp = requests.get(endpoint, headers=headers).json()
        totalresp = totalresp+resp["meta"]["result_count"]

        if resp["meta"]["result_count"] > 0:
            df_city.loc[df_city.index[df_city['Area'] == area],
                        'Total Respone'] = int(resp["meta"]["result_count"])

            df_current_area = pd.json_normalize(resp['data'])
            df_current_area.insert(2, "Area", area)
            df_all_tweets = df_all_tweets.append(df_current_area)

            if 'includes' in resp:
                df_geoloc_density = df_geoloc_density.append(
                    pd.json_normalize(resp['includes']['places']))
                for geoid in df_current_area['geo.place_id']:
                    for geoid2 in df_geoloc_density["id"]:
                        if geoid2 == geoid:
                            totalrespsamegeoid = totalrespsamegeoid+1
                            df_geoloc_density.loc[df_geoloc_density.index[df_geoloc_density['id']
                                                                          == geoid], 'Total Tweets'] = totalrespsamegeoid
                print(query, " result_count ==",
                      resp["meta"]["result_count"], " Totalrespsamegeoid ==", totalrespsamegeoid)

            print(query, " result_count ==",
                  resp["meta"]["result_count"])

            sentiment_dict = sid_obj.polarity_scores(df_current_area["text"])
            df_city.loc[df_city.index[df_city['Area'] == area],
                        'Sentiment Analysis Compound Val'] = sentiment_dict['compound']

            sentiment_analysis(
                df_current_area["text"], df_city, "Area", area, "Sentiment Analysis")

    for text in df_all_tweets["text"]:
        sentiment_analysis(
            text, df_all_tweets, "text", text, "Sentiment Analysis")

    df_all_tweets = df_all_tweets.drop('id', axis=1)
    df_all_tweets.reset_index(inplace=True)

    df_all_geo_tweets = df_all_tweets[df_all_tweets['geo.place_id'].notna()]
    df_all_geo_tweets.reset_index(inplace=True)

    df_geoloc_density.reset_index(inplace=True)

    df_grouping_k_means = Grouping_K_means(df_city)

    df_city = predicted_location_possibility(df_city)

    data = {"totalresp": totalresp,
            "df_city": df_city,
            "df_all_tweets": df_all_tweets,
            "df_all_geo_tweets": df_all_geo_tweets,
            "df_geoloc_density": df_geoloc_density,
            "df_grouping_k_means": df_grouping_k_means}

    return data
