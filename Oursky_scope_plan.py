import requests
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# function handling API call and data processing
def get_weather(lat, lon, units='metric', key='db98887c21d08c99b463ba97957717a4'):  
    url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,current&units={units}&appid={key}"
    weather_call = requests.get(url)
    if weather_call.status_code == 200:  #successful call  
        weather_data: dict = weather_call.json()  #format called data 
    else: 
        return f'Error: {weather_call.status_code} {weather_call.text}' #return failed call info 

    # Isolate hourly & daily forcast 
    hourly_forecast = weather_data['hourly']   
    daily_forecast = weather_data['daily']

    # Find local sunset and sunrise times, define observable hours (important time logic!)
    user_utc = int(datetime.now(timezone.utc).timestamp())
    #offset = weather_data['timezone_offset'] 

    '''sunsets = []
    for days in daily_forecast:
        if daily_forecast['dt'] < hourly_forecast[-1]['dt']:
            sunsets.append(days['sunset'])
        else: break 
    sunrises = []
    for days in daily_forecast:
        if daily_forecast['dt'] < hourly_forecast[-1]['dt']:
            sunrises.append(days['sunrise'])
        else: break'''

    sunrise_1 = round(daily_forecast[0]['sunrise'] / 3600) * 3600
    sunset_1 = round(daily_forecast[0]['sunset'] / 3600) * 3600
    sunrise_2 = round(daily_forecast[1]['sunrise'] / 3600) * 3600
    sunset_2 = round(daily_forecast[1]['sunset'] / 3600) * 3600
    sunrise_3 = round(daily_forecast[2]['sunrise'] / 3600) * 3600
    sunset_3 = round(daily_forecast[2]['sunset'] / 3600) * 3600
    
    last_available_hour = hourly_forecast[-1]['dt']  #making sure we don't go out of forcast's bounds 
   
    # sunrise < user_utc < sunset case... AKA daytime base case 
    if sunrise_1 < user_utc < sunset_1:
        night1_start_hour = sunset_1
        night1_end_hour = sunrise_2
        night2_start_hour = sunset_2
        night2_end_hour = min(sunrise_3, last_available_hour)
    # sunset_1 < user_utc < midnight case
    elif user_utc > sunset_1: 
        night1_start_hour = user_utc
        night1_end_hour = sunrise_2
        night2_start_hour = sunset_2
        night2_end_hour = min(sunrise_3, last_available_hour)
        night3_start_hour = sunset_3
        night3_end_hour = min(last_available_hour, 24 * 3600)
    # midnight < user_utc < sunrise_1 case
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
    #night3 = [hour for hour in hourly_forecast if night3_start_hour <= hour['dt'] <= night3_end_hour] Will add soon 
    
    # Call our functions to get the data we need  
    cloud_avg1, cloud_avg2, cloud_hourly = get_cloud_cover(hourly_forecast, night1, night2)
    temp_avg1, temp_avg2, temp_hourly = get_temperature(hourly_forecast, night1, night2)
    wind_avg1, wind_avg2, wind_hourly = get_wind_speed(hourly_forecast, night1, night2)
    precipitation_avg1, precipitation_avg2, precipitation_hourly = get_precipitation_chance(hourly_forecast, night1, night2)

    # Create complete dictionary 

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

# function to plot weather data
def plot_weather(weather_summary, site_name, sun_cycles): 
    # x-axis time
    hours = [datetime.fromtimestamp(int(ts), tz=timezone.utc) for ts in weather_summary['Cloud Cover (%)']['Hourly'].keys()]
    #datetime.fromtimestamp(current_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    # Convert night cycle timestamps to Matplotlib float format
    night1_start = mdates.date2num(datetime.fromtimestamp(sun_cycles[0], tz=timezone.utc))
    night1_end = mdates.date2num(datetime.fromtimestamp(sun_cycles[1], tz=timezone.utc))
    night2_start = mdates.date2num(datetime.fromtimestamp(sun_cycles[2], tz=timezone.utc))
    night2_end = mdates.date2num(datetime.fromtimestamp(sun_cycles[3], tz=timezone.utc))
    if len(sun_cycles) > 4:
        night3_start = mdates.date2num(datetime.fromtimestamp(sun_cycles[4], tz=timezone.utc))
        night3_end = mdates.date2num(datetime.fromtimestamp(sun_cycles[5], tz=timezone.utc))
    else: pass 
    
    # y-axis data 
    cloud_cover = list(weather_summary['Cloud Cover (%)']['Hourly'].values())
    precipitation = list(weather_summary['Precipation Chance (%)']['Hourly'].values())

    fig, ax = plt.subplots(figsize=(12, 6))  
    ax.plot(hours, cloud_cover, label='Cloud Cover (%)', color='grey', marker='o', linestyle='-')
    ax.plot(hours, precipitation, label="Precipitation Probability (%)", marker='s', linestyle='--', color='red')
    # night shading
    ax.axvspan(night1_start, night1_end, color='gray', alpha=0.3)
    ax.axvspan(night2_start, night2_end, color='gray', alpha=0.3)
    if len(sun_cycles) > 4:
        ax.axvspan(night3_start, night3_end, color='gray', alpha=0.3)
    else: pass

    # formatting
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('%')
    ax.set_ylim(0, 100)
    ax.set_title(f'{site_name} Weather Data')

    # Use Matplotlib's date formatting on x-axis
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Auto-adjusts tick frequency
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M UTC'))  # Adds date + hour

    plt.xticks(rotation=45)  
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend()
    plt.show()  

# setup weather class to easily store and access data 
class weather:
    def __init__(self, lat, lon, site_name=None):
        self.lat = lat
        self.lon = lon  
        self.name = site_name
        self.data = None
    def fetch_data(self):
        self.data, self.hours, self.suncycles = get_weather(self.lat, self.lon)
        if isinstance(self.data, str):
            print(self.data)
    def __repr__(self):
        return f"Weather Site: {self.name} ({self.lat}, {self.lon})"
# child classes for known sites
class VA(weather):
    def __init__(self):
        super().__init__(39.095472, -77.919111, 'Virginia')
class Namibia(weather):
    def __init__(self):
        super().__init__(-23.236390, 16.361670, "Namibia")
class Chile(weather):
    def __init__(self):
        super().__init__(-30.470556, -70.765000, "Chile") 
class SRO(weather):
    def __init__(self):
        super().__init__(37.070367, -119.413093, 'SRO')
class awoa(weather):
    def __init__(self):
        super().__init__(-31.820398, 117.281526, 'AWOA')
class Wolongbar(weather):
    def __init__(self):
        super().__init__(-28.820056, 153.420472, 'Wolongbar')
class Brazil(weather):
    def __init__(self):
        super().__init__(-21.736300, -41.026200, 'Brazil')
# site functions, may or may not be needed 
def virgina():  
    #lat, lon = 39.095472, -77.919111 
    weather.va = VA()
    weather.va.fetch_data() 
    
    plot_weather(weather.va.data, weather.va.name, weather.va.suncycles)
def sro():  
    #lat, lon = 37.070367, -119.413093
    weather.sro = SRO()
    weather.sro.fetch_data()
    
    plot_weather(weather.sro.data, weather.sro.name, weather.sro.suncycles)
def wolongbar():
    #lat, lon = -28.820056, 153.420472
    weather.wolongbar = Wolongbar()
    weather.wolongbar.fetch_data()

    plot_weather(weather.wolongbar.data, weather.wolongbar.name, weather.wolongbar.suncycles)
def namibia(): 
    #lat, lon = -23.236390, 16.361670 
    weather.namibia = Namibia()
    weather.namibia.fetch_data()    

    plot_weather(weather.namibia.data, weather.namibia.name, weather.namibia.suncycles)
def AOWA():  
    #lat, lon = -31.820398, 117.281526 

    weather.awoa = awoa()
    weather.awoa.fetch_data()
    
    plot_weather(weather.awoa.data, weather.awoa.name, weather.awoa.suncycles)
def brazil():  
    #lat, lon = -21.736300, -41.026200
    weather.brazil = Brazil()
    weather.brazil.fetch_data()

    plot_weather(weather.brazil.data, weather.brazil.name, weather.brazil.suncycles)  
def chile():  
    #lat, lon = -30.470556, -70.765000
    weather.chile = Chile()
    weather.chile.fetch_data()

    plot_weather(weather.chile.data, weather.chile.name, weather.chile.suncycles)

# functions sorting/filtering data from API call returns a (night avg, hourly dict, segment dict)
def get_cloud_cover(hourly_forecast, night1, night2, night3=None):
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
    
    # Create a dictionary for each hour of the night 
    hourly_dict = {time_stamps[i]: value for i, value in enumerate(cloud_cover)}

    #Create a dictionary for start, middle, and end of night !!!(Not used in graphing, maybe useful later on... needs to be patched for having 2 nights)!!!
    '''night_segment = (len(cloud_cover))/3
    if night_segment.is_integer():
        start_segment = cloud_cover[:int(night_segment)]
        middle_segment = cloud_cover[int(night_segment):int(night_segment*2)]
        end_segment = cloud_cover[int(night_segment*2):]
    elif isinstance(night_segment, float):
        start_segment = cloud_cover[:int(night_segment)+1]
        middle_segment = cloud_cover[int(night_segment)+1:int(night_segment*2)+1]
        end_segment = cloud_cover[int(night_segment*2)+1:]


    # Calculate average cloud cover for each segment
    start_avg = round(sum(start_segment) / len(start_segment),2) if start_segment else 0
    middle_avg = round(sum(middle_segment) / len(middle_segment),2) if middle_segment else 0
    end_avg = round(sum(end_segment) / len(end_segment),2) if end_segment else 0
    
    # Create a dictionary for the average cloud cover for each segment
    segment_dict = {'Start': start_avg, 'Middle': middle_avg, 'End': end_avg}'''
    
    return night1_avg, night2_avg, hourly_dict
def get_temperature(hourly_forecast, night1, night2, night3=None):
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
    
    # Create a dictionary for each hour of forcast  
    hourly_dict = {time_stamps[i]: value for i, value in enumerate(temp)}

    #Create a dictionary for start, middle, and end of night !!!(Not used in graphing, maybe useful later on... needs to be patched for having 2 nights)!!!
    '''night_segment = (len(temperature))/3
    if night_segment.is_integer():
        start_segment = temperature[:int(night_segment)]
        middle_segment = temperature[int(night_segment):int(night_segment*2)]
        end_segment = temperature[int(night_segment*2):]
    elif isinstance(night_segment, float):
        start_segment = temperature[:int(night_segment)+1]
        middle_segment = temperature[int(night_segment)+1:int(night_segment*2)+1]
        end_segment = temperature[int(night_segment*2)+1:]


    # Calculate average cloud cover for each segment
    start_avg = round(sum(start_segment) / len(start_segment),2) if start_segment else 0
    middle_avg = round(sum(middle_segment) / len(middle_segment),2) if middle_segment else 0
    end_avg = round(sum(end_segment) / len(end_segment),2) if end_segment else 0
    
    # Create a dictionary for the average temperature for each segment
    segment_dict = {'Start': start_avg, 'Middle': middle_avg, 'End': end_avg}'''

    return night1_avg, night2_avg, hourly_dict
def get_wind_speed(hourly_forecast, night1, night2, night3=None):
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
    
    # Create a dictionary for each hour of the night 
    hourly_dict = {time_stamps[i]: value for i, value in enumerate(wind_speed)}

   #Create a dictionary for start, middle, and end of night !!!(Not used in graphing, maybe useful later on... needs to be patched for having 2 nights)!!!
    '''night_segment = (len(wind_speed))/3
    if night_segment.is_integer():
        start_segment = wind_speed[:int(night_segment)]
        middle_segment = wind_speed[int(night_segment):int(night_segment*2)]
        end_segment = wind_speed[int(night_segment*2):]
    elif isinstance(night_segment, float):
        start_segment = wind_speed[:int(night_segment)+1]
        middle_segment = wind_speed[int(night_segment)+1:int(night_segment*2)+1]
        end_segment = wind_speed[int(night_segment*2)+1:]


    # Calculate average cloud cover for each segment
    start_avg = round(sum(start_segment) / len(start_segment),2) if start_segment else 0
    middle_avg = round(sum(middle_segment) / len(middle_segment),2) if middle_segment else 0
    end_avg = round(sum(end_segment) / len(end_segment),2) if end_segment else 0
    
    # Create a dictionary for the average wind speed for each segment
    segment_dict = {'Start': start_avg, 'Middle': middle_avg, 'End': end_avg}'''
    
    return night1_avg, night2_avg, hourly_dict 
def get_precipitation_chance(hourly_forecast, night1, night2, night3=None):
    precipitation = []
    precipitation1 = []
    precipitation2 = []
    time_stamps = []
    for hour in hourly_forecast:
        precipitation.append((hour['pop'])*100)
        time_stamps.append((hour['dt'])*100)
    for hour in night1:
        precipitation1.append((hour['pop'])*100)
    for hour in night2:
        precipitation2.append((hour['pop'])*100)

    night1_avg = sum(precipitation1) / len(precipitation1)
    night2_avg = sum(precipitation2) / len(precipitation2)
    
    # Create a dictionary for each hour of the night 
    hourly_dict = {time_stamps[i]: value for i, value in enumerate(precipitation)}

    #Create a dictionary for start, middle, and end of night !!!(Not used in graphing, maybe useful later on... needs to be patched for 48 hr forcast)!!!
    '''night_segment = (len(precipitation_chance))/3
    if night_segment.is_integer():
        start_segment = precipitation_chance[:int(night_segment)]
        middle_segment = precipitation_chance[int(night_segment):int(night_segment*2)]
        end_segment = precipitation_chance[int(night_segment*2):]
    elif isinstance(night_segment, float):
        start_segment = precipitation_chance[:int(night_segment)+1]
        middle_segment = precipitation_chance[int(night_segment)+1:int(night_segment*2)+1]
        end_segment = precipitation_chance[int(night_segment*2)+1:]


    # Calculate average cloud cover for each segment
    start_avg = round(sum(start_segment) / len(start_segment),2) if start_segment else 0
    middle_avg = round(sum(middle_segment) / len(middle_segment),2) if middle_segment else 0
    end_avg = round(sum(end_segment) / len(end_segment),2) if end_segment else 0
    
    # Create a dictionary for the average cloud cover for each segment
    segment_dict = {'Start': start_avg, 'Middle': middle_avg, 'End': end_avg}'''
    
    return night1_avg, night2_avg, hourly_dict

#console runtime function 
def run():
    print("Welcome to the Oursky Scope Plan!")
    print("Please select a location:")
    print("1. Virginia")
    print("2. SRO")
    print("3. Wolongbar")
    print("4. Namibia")
    print("5. AOWA")
    print("6. Brazil")
    print("7. Chile")
    choice = input("Enter the number of your choice: ")
    if choice == '1':
        virgina()
    elif choice == '2':
        sro()
    elif choice == '3':
        wolongbar()
    elif choice == '4':
        namibia()
    elif choice == '5':
        AOWA()
    elif choice == '6':
        brazil()
    elif choice == '7':
        chile()
    else:
        print("Invalid choice. Please try again.")           
if __name__ == '__main__':
    while True: 
        run()
        cont = input("Do you want to check another location? (yes/no): ").strip().lower()
        if cont != 'yes':
            print("Thank you for using the Oursky Scope Plan!")
            break
    