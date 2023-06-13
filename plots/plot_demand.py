import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parse the .txt file for electricity consumption data
with open('../../../Hydrogen/el_demand_2021.txt', 'r') as file:
    consumption_data = [float(line.strip()) for line in file]

# Generate a date-time index from 2021 to 2021, at hourly intervals
date_range = pd.date_range(start='2021-01-01', end='2021-12-31 23:00:00', freq='H')

# Create a DataFrame for electricity consumption
consumption_df = pd.DataFrame(consumption_data, columns=['Consumption'], index=date_range)

# Load the production data
e_pv_sto = np.loadtxt("../../../Hydrogen/PV_STO.txt")
e_pv_tip = np.loadtxt("../../../Hydrogen/PV_TIP.txt")
e_pv_main_building = np.loadtxt("../../../Hydrogen/PV_Main_building.txt")

# Parse the .txt files for each scenario
scenario1_file = '../../../Hydrogen/PV_agriPV_south.txt' # Path to the first scenario file
scenario2_file = '../../../Hydrogen/PV_agriPV_north_east-west.txt'   # Path to the second scenario file
scenario3_file = '../../../Hydrogen/PV_agriPV_north_south-north.txt'  # Path to the third scenario file

# Read the data from each file
scenario1_data = pd.read_csv(scenario1_file, header=None, names=['Rooftop PV + AgriPV (South E/W)'])
scenario2_data = pd.read_csv(scenario2_file, header=None, names=['Rooftop PV + AgriPV (North E/W)'])
scenario3_data = pd.read_csv(scenario3_file, header=None, names=['Rooftop PV + AgriPV (North S/N)'])

# Add the production data to each scenario
scenario1_data['Rooftop PV + AgriPV (South E/W)'] += e_pv_sto + e_pv_tip + e_pv_main_building
scenario2_data['Rooftop PV + AgriPV (North E/W)'] += e_pv_sto + e_pv_tip + e_pv_main_building
scenario3_data['Rooftop PV + AgriPV (North S/N)'] += e_pv_sto + e_pv_tip + e_pv_main_building

# Set the index based on the date range
scenario1_data.index = date_range
scenario2_data.index = date_range
scenario3_data.index = date_range

# Create separate figures and axes for each scenario
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Plot the data for each scenario
ax1.plot(consumption_df.index, consumption_df['Consumption'], color='darkblue', label='Electricity Consumption')
ax1.plot(scenario1_data.index, scenario1_data['Rooftop PV + AgriPV (South E/W)'], color='darkgreen', label='Rooftop PV + AgriPV (South E/W)', alpha = 0.7)
ax1.axhline(y=100, color='black', linestyle='--', label='Feed-in limit (100 kW)')
ax1.set_xlabel('Year-Month')
ax1.set_ylabel('Energy [kWh]')
ax1.set_title('Electricity Consumption and Rooftop PV + AgriPV (South E/W)')
ax1.legend()

ax2.plot(consumption_df.index, consumption_df['Consumption'], color='darkblue', label='Electricity Consumption')
ax2.plot(scenario2_data.index, scenario2_data['Rooftop PV + AgriPV (North E/W)'], color='darkred', label='Rooftop PV + AgriPV (North E/W)', alpha = 0.7)
ax2.axhline(y=100, color='black', linestyle='--', label='Feed-in limit (100 kW)')
ax2.set_xlabel('Year-Month')
ax2.set_ylabel('Energy [kWh]')
ax2.set_title('Electricity Consumption and Rooftop PV + AgriPV (North E/W)')
ax2.legend()

ax3.plot(consumption_df.index, consumption_df['Consumption'], color='darkblue', label='Electricity Consumption')
ax3.plot(scenario3_data.index, scenario3_data['Rooftop PV + AgriPV (North S/N)'], color='darkorange', label='Rooftop PV + AgriPV (North S/N)', alpha = 0.7)
ax3.axhline(y=100, color='black', linestyle='--', label='Feed-in limit (100 kW)')
ax3.set_xlabel('Year-Month')
ax3.set_ylabel('Energy [kWh]')
ax3.set_title('Electricity Consumption and Rooftop PV + AgriPV (North S/N)')
ax3.legend()

plt.show()