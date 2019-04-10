# -*- coding: utf-8 -*-
"""
bundoran weather report

@author: RUAIRI
"""
import requests
from datetime import datetime
import json


class weather_data:
    #api_address= 'http://api.openweathermap.org/data/2.5/weather?appid=b2d41430bd5b5403291b78b6b1fe12e0&units=metric&q=bundoran'
    new_api= 'http://magicseaweed.com/api/a6e57b6c5788cde8560cac105c2e9344/forecast/?spot_id=50&units=eu'
    api_address= 'http://api.openweathermap.org/data/2.5/weather?appid=b2d41430bd5b5403291b78b6b1fe12e0&units=metric&q=bundoran'
    jsonDayDat=requests.get(api_address).json()
    jsonDat=requests.get(new_api).json()#import and convert to json
  #  print (jsonDat)
    time_stamps=jsonDat[0]['timestamp']
    break_min=jsonDat[0]['swell']['minBreakingHeight']
    break_max=jsonDat[0]['swell']['maxBreakingHeight']
    period_val=jsonDat[0]['swell']['components']['primary']['period']
    print('period',period_val)
    print('Wave_Max',break_max)
    print('Wave_Min',break_min)
    
    
    Sun_set=jsonDayDat['sys']['sunset']
    Sun_rise=jsonDayDat['sys']['sunrise']

    print('Sunrise',datetime.utcfromtimestamp(Sun_rise).strftime('%Y-%m-%d %H:%M:%S'))