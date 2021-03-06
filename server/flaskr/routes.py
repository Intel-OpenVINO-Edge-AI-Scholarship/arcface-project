from .create import create_app
from datetime import datetime, timedelta
import math
import sqlite3
import json
from flask import make_response, request
import pandas as pd
import sys

app = create_app(dbfile='people_counter.db')

def database_connection():
    db = sqlite3.connect('people_counter.db')
    return db

@app.route('/ephemeric/<string:lat>/<string:lng>/<string:timestamp>')
def hello(lat, lng, timestamp):
    d = pd.Timestamp(timestamp)
    pos = suncalc.getPosition(d.to_pydatetime(), float(lat), float(lng))
    d = dict(zip(['azimuth', 'altitude'], [pos['azimuth'], pos['altitude']]))
    enc = json.JSONEncoder()
    response = make_response(enc.encode(d))
    response.headers['Content-Type'] = 'application/json'
    return response

def arcface():
    

# @app.route('/ephemeric')
# def hello():
#     cities = json.load(open('city.list.json', 'r'))
#     suncalc.getPosition(d.to_pydatetime(), lat, lng)
#     return 

@app.route('/azimuth_altitude_write', methods=['POST'])
def write_text():
    dataURLArray = request.get_data()
    decoder = json.JSONDecoder()
    dataURLArray = decoder.decode(dataURLArray.decode())
    res = []
    print(dataURLArray, file=sys.stderr)
    for fileno, dataUrl in enumerate(list(dataURLArray.values())):
        f = open('azimuth_altitude_canvas_file' + (fileno+1).__str__(), 'w')
        f.write(dataUrl)
        f.flush()
        f.close()
        res.append(1)
    enc = json.JSONEncoder()
    response = make_response(enc.encode(res))
    response.headers['Content-Type'] = 'application/json'
    return response

