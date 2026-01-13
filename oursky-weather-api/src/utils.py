def get_weather(lat, lon, units='metric', key='db98887c21d08c99b463ba97957717a4'):  
    url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,current&units={units}&appid={key}"
    weather_call = requests.get(url)
    if weather_call.status_code == 200:  
        weather_data = weather_call.json()  
    else: 
        return f'Error: {weather_call.status_code} {weather_call.text}' 

    hourly_forecast = weather_data['hourly']   
    daily_forecast = weather_data['daily']

    user_utc = int(datetime.now(timezone.utc).timestamp())

    sunrise_1 = round(daily_forecast[0]['sunrise'] / 3600) * 3600
    sunset_1 = round(daily_forecast[0]['sunset'] / 3600) * 3600
    sunrise_2 = round(daily_forecast[1]['sunrise'] / 3600) * 3600
    sunset_2 = round(daily_forecast[1]['sunset'] / 3600) * 3600
    sunrise_3 = round(daily_forecast[2]['sunrise'] / 3600) * 3600
    sunset_3 = round(daily_forecast[2]['sunset'] / 3600) * 3600
    
    last_available_hour = hourly_forecast[-1]['dt']  

    if sunrise_1 < user_utc < sunset_1:
        night1_start_hour = sunset_1
        night1_end_hour = sunrise_2
        night2_start_hour = sunset_2
        night2_end_hour = min(sunrise_3, last_available_hour)
    elif user_utc > sunset_1: 
        night1_start_hour = user_utc
        night1_end_hour = sunrise_2
        night2_start_hour = sunset_2
        night2_end_hour = min(sunrise_3, last_available_hour)
        night3_start_hour = sunset_3
        night3_end_hour = min(last_available_hour, 24 * 3600)
    elif user_utc < sunrise_1:
        night1_start_hour = user_utc
        night1_end_hour = sunrise_1
        night2_start_hour = sunset_1
        night2_end_hour = sunrise_2
        night3_start_hour = sunset_2
        night3_end_hour = min(sunrise_3, last_available_hour)

    if user_utc < sunrise_1:
        sun_cycles = [night1_start_hour, night1_end_hour, night2_start_hour, night2_end_hour, night3_start_hour, night3_end_hour]
    else: 
        sun_cycles = [night1_start_hour, night1_end_hour, night2_start_hour, night2_end_hour]

    night1 = [hour for hour in hourly_forecast if night1_start_hour <= hour['dt'] <= night1_end_hour]
    night2 = [hour for hour in hourly_forecast if night2_start_hour <= hour['dt'] <= night2_end_hour]

    cloud_avg1, cloud_avg2, cloud_hourly = get_cloud_cover(hourly_forecast, night1, night2)
    temp_avg1, temp_avg2, temp_hourly = get_temperature(hourly_forecast, night1, night2)
    wind_avg1, wind_avg2, wind_hourly = get_wind_speed(hourly_forecast, night1, night2)
    precipitation_avg1, precipitation_avg2, precipitation_hourly = get_precipitation_chance(hourly_forecast, night1, night2)

    weather_summary = {
        'Cloud Cover (%)': {
            'Night 1 average': cloud_avg1,
            'Night 2 average': cloud_avg2,
            'Hourly': cloud_hourly,
        },
        'Temperature (C)': {
            'Night 1 average': temp_avg1,
            'Night 2 average': temp_avg2,
            'Hourly': temp_hourly,
        },
        'Wind Speed (km/h)': {
            'Night 1 average': wind_avg1,
            'Night 2 average': wind_avg2,
            'Hourly': wind_hourly,
        },
        'Precipation Chance (%)': {
            'Night 1 average': precipitation_avg1,
            'Night 2 average': precipitation_avg2,
            'Hourly': precipitation_hourly,
        }
    }
    return weather_summary, cloud_hourly, sun_cycles

def get_cloud_cover(hourly_forecast, night1, night2):
    cloud_cover = []
    cloud_cover1 = []
    cloud_cover2 = []
    time_stamps = []
    for hour in hourly_forecast:
        cloud_cover.append(hour['clouds'])
        time_stamps.append(hour['dt'])
    for hour in night1:
        cloud_cover1.append(hour['clouds'])
    for hour in night2:
        cloud_cover2.append(hour['clouds'])

    night1_avg = sum(cloud_cover1) / len(cloud_cover1)
    night2_avg = sum(cloud_cover2) / len(cloud_cover2)
    
    hourly_dict = {time_stamps[i]: value for i, value in enumerate(cloud_cover)}

    return night1_avg, night2_avg, hourly_dict

def get_temperature(hourly_forecast, night1, night2):
    temp = []
    temp1 = []
    temp2 = []
    time_stamps = []
    for hour in hourly_forecast:
        temp.append(hour['temp'])
        time_stamps.append(hour['dt'])
    for hour in night1:
        temp1.append(hour['temp'])
    for hour in night2:
        temp2.append(hour['temp'])

    night1_avg = sum(temp1) / len(temp1)
    night2_avg = sum(temp2) / len(temp2)
    
    hourly_dict = {time_stamps[i]: value for i, value in enumerate(temp)}

    return night1_avg, night2_avg, hourly_dict

def get_wind_speed(hourly_forecast, night1, night2):
    wind_speed = []
    wind_speed1 = []
    wind_speed2 = []
    time_stamps = []
    for hour in hourly_forecast:
        wind_speed.append(hour['wind_speed'])
        time_stamps.append(hour['dt'])
    for hour in night1:
        wind_speed1.append(hour['wind_speed'])
    for hour in night2:
        wind_speed2.append(hour['wind_speed'])

    night1_avg = sum(wind_speed1) / len(wind_speed1)
    night2_avg = sum(wind_speed2) / len(wind_speed2)
    
    hourly_dict = {time_stamps[i]: value for i, value in enumerate(wind_speed)}

    return night1_avg, night2_avg, hourly_dict 

def get_precipitation_chance(hourly_forecast, night1, night2):
    precipitation = []
    precipitation1 = []
    precipitation2 = []
    time_stamps = []
    for hour in hourly_forecast:
        precipitation.append((hour['pop']) * 100)
        time_stamps.append((hour['dt']) * 100)
    for hour in night1:
        precipitation1.append((hour['pop']) * 100)
    for hour in night2:
        precipitation2.append((hour['pop']) * 100)

    night1_avg = sum(precipitation1) / len(precipitation1)
    night2_avg = sum(precipitation2) / len(precipitation2)
    
    hourly_dict = {time_stamps[i]: value for i, value in enumerate(precipitation)}

    return night1_avg, night2_avg, hourly_dict