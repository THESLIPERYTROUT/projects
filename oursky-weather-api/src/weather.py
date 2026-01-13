class Weather:
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

class VA(Weather):
    def __init__(self):
        super().__init__(39.095472, -77.919111, 'Virginia')

class Namibia(Weather):
    def __init__(self):
        super().__init__(-23.236390, 16.361670, "Namibia")

class Chile(Weather):
    def __init__(self):
        super().__init__(-30.470556, -70.765000, "Chile") 

class SRO(Weather):
    def __init__(self):
        super().__init__(37.070367, -119.413093, 'SRO')

class AOWA(Weather):
    def __init__(self):
        super().__init__(-31.820398, 117.281526, 'AWOA')

class Wolongbar(Weather):
    def __init__(self):
        super().__init__(-28.820056, 153.420472, 'Wolongbar')

class Brazil(Weather):
    def __init__(self):
        super().__init__(-21.736300, -41.026200, 'Brazil')