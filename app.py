import json
from flask import Flask, render_template, jsonify, request
import leisure_hub as leisurehub
from flask_cors import CORS
import time
app = Flask(__name__)
app.config.from_object(__name__)


CORS(app, resources={r'/*': {'origins': '*'}})


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/getnewlocation', methods=["GET", "POST"])
def getnewlocation():
    # city = request.form.get("city")
    # categorie = request.form.get("leisurehub")
    # print(city, categorie)

    city = request.form["city"]
    categorie = request.form["leisurehub"]
    print(city, categorie)

    data = leisurehub.getalldata(city, categorie)
    totalresp = data["totalresp"]
    df_city = data["df_city"]
    df_all_tweets = data["df_all_tweets"]
    df_all_geo_tweets = data["df_all_geo_tweets"]
    df_geoloc_density = data["df_geoloc_density"]
    df_grouping_k_means = data["df_grouping_k_means"]

    print(totalresp)
    print(df_city)
    print(df_all_tweets)
    print(df_all_geo_tweets)
    print(df_geoloc_density)
    print(df_grouping_k_means)

    data = {"totalresp": totalresp,
            "df_city": df_city.to_json(),
            "df_all_tweets": df_all_tweets.to_json(),
            "df_all_geo_tweets": df_all_geo_tweets.to_json(),
            "df_geoloc_density": df_geoloc_density.to_json(),
            "df_grouping_k_means": df_grouping_k_means.to_json()
            }

    # return render_template('index.html', data=data)
    return data


if __name__ == '__main__':
    app.run()
