import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Load the data
e_pv_sto = np.loadtxt("../../../Hydrogen/PV_STO.txt") 
e_pv_tip = np.loadtxt("../../../Hydrogen/PV_TIP.txt")
e_pv_main_building = np.loadtxt("../../../Hydrogen/PV_Main_building.txt")

# Sum up the data from all three files
data = e_pv_sto + e_pv_tip + e_pv_main_building

# Check the number of hours in your data
num_hours = len(data)

# Generate a date-time index based on the length of your data
date_range = pd.date_range(start='2000-01-01', periods=num_hours, freq='H')

# Create a DataFrame
df = pd.DataFrame(data, columns=['Production'], index=date_range)

# Plot the data
plt.figure(figsize=(10,6))
plt.plot(df.index, df['Production'], color='darkorange')
plt.xlabel('Month')
plt.ylabel('Production [kWh]')
plt.title('Total Simulated Rooftop PV Production (base case scenario)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b')) 
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tight_layout()
plt.show()
