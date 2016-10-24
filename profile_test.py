import missingno.missingno as msno
import pandas as pd
collisions = pd.read_csv("NYPD_Motor_vehicle_Collisions.csv")

msno.geoplot(collisions.sample(10000), x='LONGITUDE', y='LATITUDE')