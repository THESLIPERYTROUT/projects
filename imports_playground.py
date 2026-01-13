import requests
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


'''def get_weather(lat, lon, units='metric', key='db98887c21d08c99b463ba97957717a4'):  
    url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,current&units={units}&appid={key}"
    weather_call = requests.get(url)
    if weather_call.status_code == 200:  #successful call  
        weather_data = weather_call.json()  #format called data 
    else: 
        return f'Error: {weather_call.status_code} {weather_call.text}' 
    date = (weather_data['hourly'][0]['dt'])
    print(date)
get_weather(39.095472, -77.919111)'''

#print(datetime.fromtimestamp(1741745116.640628, tz=timezone.utc))
#print(datetime.fromtimestamp(1741745116.640628, tz=timezone.utc))
#print(datetime.fromtimestamp(1741745116.640628))
current_timestamp = datetime.now().timestamp()
print(current_timestamp)

if datetime.utcfromtimestamp(current_timestamp) == datetime.fromtimestamp(current_timestamp, tz=timezone.utc):
    print('True')
else: 
    print('False')

'''print(datetime.utcfromtimestamp(current_timestamp))
print(datetime.fromtimestamp(current_timestamp, tz=timezone.utc))
print(datetime.fromtimestamp(current_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
print(type(datetime.fromtimestamp(current_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')))
print(type(datetime.fromtimestamp(current_timestamp, tz=timezone.utc)))'''

current_timestamp = 1741748687

print((datetime.fromtimestamp(current_timestamp, tz=timezone.utc)))
print((datetime.utcfromtimestamp(current_timestamp)))