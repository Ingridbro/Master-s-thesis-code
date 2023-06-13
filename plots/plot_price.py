import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Parse the .txt file
with open('../../../Hydrogen/PV_STO.txt', 'r') as file:
    data = [float(line.strip()) for line in file]

# Check the number of hours in your data
num_hours = len(data)

# Generate a date-time index based on the length of your data
date_range = pd.date_range(start='2000-01-01', periods=num_hours, freq='H')

# Create a DataFrame
df = pd.DataFrame(data, columns=['Consumption'], index=date_range)

# Plot the data
plt.figure(figsize=(10,6))
plt.plot(df.index, df['Consumption'], color='darkorange') # Change color of the line
plt.xlabel('Month')
plt.ylabel('Production [kWh]')
plt.title('Simulated PV production from STO building (rooftop solar PV)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b')) # Format x-axis to display month names
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # Set major ticks location every month
plt.xticks(rotation=45) # Rotate x-axis labels
plt.tick_params(axis='both', which='major', labelsize=8) # Reduce the size of the ticks
plt.tight_layout() # Adjust subplot parameters to give specified padding.
plt.show()
