import pandas as pd
import numpy as np

# Define the function
def aggregate_energy_data(input_file, output_file):
    # Load the data
    data = np.loadtxt(input_file)
    df = pd.DataFrame(data, columns=['consumption'])

    # Calculate the timestamp based on the row number (which corresponds to the hour of the year)
    start_time = pd.Timestamp('2022-01-01')
    df['timestamp'] = df.index.to_series().apply(lambda x: start_time + pd.Timedelta(hours=x))

    # Set the timestamp as the index
    df.set_index('timestamp', inplace=True)

    # Resample the data to get monthly totals
    monthly_data = df.resample('M').sum()

    # Reset index to make 'timestamp' a column again
    monthly_data.reset_index(inplace=True)

    # Convert timestamp to month
    monthly_data['timestamp'] = monthly_data['timestamp'].dt.to_period('M')

    # Save the result to a new text file
    monthly_data.to_csv(output_file, sep="\t", index=False)

# Call the function with your input and output filenames
aggregate_energy_data("../../../Hydrogen/PV_agriPV_north_east-west.txt", 'monthly_pv_production_north-e-w.txt')

