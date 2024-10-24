question 9 : Distance Matrix Calculation

import pandas as pd
import numpy as np

# Step 1: Create a sample dataset and save it to a CSV file
def create_sample_csv(file_path):
    # Create a sample dataset with more IDs and distances
    data = {
        'from_id': [1001400, 1001400, 1001402, 1001402, 1001404, 1001406, 
                    1001400, 1001402, 1001404, 1001406, 1001408, 1001410, 1001412],
        'to_id': [1001402, 1001404, 1001404, 1001406, 1001406, 1001400, 
                  1001408, 1001408, 1001410, 1001412, 1001410, 1001412, 1001400],
        'distance': [9.7, 29.9, 20.2, 16.0, 21.7, 45.9, 
                     67.6, 57.9, 11.1, 15.6, 78.7, 94.3, 84.6]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
    print(f"CSV file '{file_path}' created successfully!")

# Step 2: Calculate the distance matrix from the CSV file
def calculate_distance_matrix(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Check for missing values
    if df.isnull().any().any():
        print("Warning: Missing values detected in the dataset.")
        return None

    # Ensure distance is numeric
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')

    # Extract unique IDs
    unique_ids = df['from_id'].unique().tolist() + df['to_id'].unique().tolist()
    unique_ids = list(set(unique_ids))  # Remove duplicates

    # Initialize a distance matrix with np.inf (infinity) for unknown distances
    distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)

    # Set the diagonal to 0 (distance from a point to itself)
    np.fill_diagonal(distance_matrix.values, 0)

    # Populate the initial distances from the DataFrame
    for _, row in df.iterrows():
        distance_matrix.loc[row['from_id'], row['to_id']] = row['distance']
        distance_matrix.loc[row['to_id'], row['from_id']] = row['distance']  # Ensure symmetry

    # Apply the Floyd-Warshall algorithm to calculate all-pairs shortest paths
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.loc[i, j] > distance_matrix.loc[i, k] + distance_matrix.loc[k, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]

    return distance_matrix

# Main execution
csv_file_path = 'dataset-2.csv'
create_sample_csv(csv_file_path)
result_df = calculate_distance_matrix(csv_file_path)

# Display the result
print(result_df)


question 10 : unroll distance matrix

import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Create a list to store the rows of the new DataFrame
    rows = []

    # Iterate over the distance matrix DataFrame
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            # Skip the case where id_start is the same as id_end
            if id_start != id_end:
                distance = distance_matrix.loc[id_start, id_end]
                rows.append((id_start, id_end, distance))

    # Create a new DataFrame from the list of rows
    unrolled_df = pd.DataFrame(rows, columns=['id_start', 'id_end', 'distance'])
    
    return unrolled_df

# Example usage:
# distance_matrix = pd.read_csv('distance_matrix.csv', index_col=0)  # Assuming you load your distance matrix here
# result_df = unroll_distance_matrix(distance_matrix)
# print(result_df)


Question 11: finding IDs within percentage threshold

import pandas as pd

def find_ids_within_ten_percentage_threshold(df, reference_id):
    # Calculate the average distance for the reference_id
    average_distance = df.loc[df['id_start'] == reference_id, 'distance'].mean()
    
    if pd.isna(average_distance):
        return []  # Return empty list if there's no data for the reference_id
    
    # Calculate the threshold values
    lower_threshold = average_distance * 0.90
    upper_threshold = average_distance * 1.10

    # Filter the DataFrame for IDs within the threshold
    filtered_ids = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]['id_start']

    # Return a sorted list of unique IDs
    return sorted(filtered_ids.unique())

# Example usage:
# df = pd.read_csv('dataset-10.csv')  # Assuming this is your DataFrame from Question 10
# result_ids = find_ids_within_ten_percentage_threshold(df, reference_id)
# print(result_ids)




question 12 : calculate toll rate
import pandas as pd

def calculate_toll_rate(df):
    # Define rate coefficients for each vehicle type
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates and add to the DataFrame
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate

    return df

# Example usage
if __name__ == "__main__":
    # Sample DataFrame with 10 rows
    data = {
        'id_start': [1001400, 1001400, 1001400, 1001402, 1001402, 
                     1001404, 1001404, 1001406, 1001408, 1001410],
        'id_end': [1001402, 1001404, 1001406, 1001400, 1001404, 
                   1001400, 1001402, 1001408, 1001410, 1001412],
        'distance': [9.7, 29.9, 45.0, 20.2, 16.0, 
                     37.7, 48.8, 21.7, 11.1, 15.6]
    }
    df = pd.DataFrame(data)

    # Calculate toll rates
    updated_df = calculate_toll_rate(df)

    # Display the updated DataFrame
    print(updated_df)
 


Question 13 : calculate time based toll rates

import pandas as pd
import numpy as np
from datetime import time

# Sample DataFrame creation (for demonstration purposes)
def create_sample_dataframe():
    data = {
        'id_start': [1001400, 1001402, 1001404],
        'id_end': [1001402, 1001404, 1001406],
        'moto': [5, 10, 2],
        'car': [20, 15, 25],
        'rv': [3, 5, 4],
        'bus': [2, 1, 1],
        'truck': [1, 2, 1],
    }
    df = pd.DataFrame(data)
    return df

def calculate_time_based_toll_rates(df):
    # Create mapping for weekdays
    days_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                    4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    # Prepare new columns
    start_days = []
    end_days = []
    start_times = []
    end_times = []
    toll_rates = []

    # Populate the new columns based on the discount rules
    for index, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        
        for day in range(7):  # Monday to Sunday
            start_day = days_mapping[day]
            end_day = start_day  # Assume end day is the same for this example
            start_time = time(0, 0)  # Start of the day
            end_time = time(23, 59, 59)  # End of the day

            if day < 5:  # Weekdays
                # Apply different discount rates based on the time ranges
                for hour in range(24):
                    if hour < 10:  # From 00:00 to 10:00
                        discount_factor = 0.8
                    elif hour < 18:  # From 10:00 to 18:00
                        discount_factor = 1.2
                    else:  # From 18:00 to 23:59
                        discount_factor = 0.8
                    
                    # Calculate toll rates for each vehicle type
                    toll_rate_moto = row['moto'] * discount_factor
                    toll_rate_car = row['car'] * discount_factor
                    toll_rate_rv = row['rv'] * discount_factor
                    toll_rate_bus = row['bus'] * discount_factor
                    toll_rate_truck = row['truck'] * discount_factor
                    
                    toll_rates.append({
                        'id_start': id_start,
                        'id_end': id_end,
                        'start_day': start_day,
                        'end_day': end_day,
                        'start_time': start_time,
                        'end_time': end_time,
                        'moto': toll_rate_moto,
                        'car': toll_rate_car,
                        'rv': toll_rate_rv,
                        'bus': toll_rate_bus,
                        'truck': toll_rate_truck,
                    })
            
            else:  # Weekends
                discount_factor = 0.7
                # Calculate toll rates for each vehicle type
                toll_rate_moto = row['moto'] * discount_factor
                toll_rate_car = row['car'] * discount_factor
                toll_rate_rv = row['rv'] * discount_factor
                toll_rate_bus = row['bus'] * discount_factor
                toll_rate_truck = row['truck'] * discount_factor
                
                toll_rates.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': start_day,
                    'end_day': end_day,
                    'start_time': start_time,
                    'end_time': end_time,
                    'moto': toll_rate_moto,
                    'car': toll_rate_car,
                    'rv': toll_rate_rv,
                    'bus': toll_rate_bus,
                    'truck': toll_rate_truck,
                })

    # Create a new DataFrame from the toll rates list
    toll_rates_df = pd.DataFrame(toll_rates)

    return toll_rates_df

# Example usage
df = create_sample_dataframe()
result_df = calculate_time_based_toll_rates(df)

# Display the resulting DataFrame
print(result_df)
import pandas as pd
import numpy as np
from datetime import time

# Sample DataFrame creation (for demonstration purposes)
def create_sample_dataframe():
    data = {
        'id_start': [1001400, 1001402, 1001404],
        'id_end': [1001402, 1001404, 1001406],
        'moto': [5, 10, 2],
        'car': [20, 15, 25],
        'rv': [3, 5, 4],
        'bus': [2, 1, 1],
        'truck': [1, 2, 1],
    }
    df = pd.DataFrame(data)
    return df

def calculate_time_based_toll_rates(df):
    # Create mapping for weekdays
    days_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                    4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    # Prepare new columns
    start_days = []
    end_days = []
    start_times = []
    end_times = []
    toll_rates = []

    # Populate the new columns based on the discount rules
    for index, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        
        for day in range(7):  # Monday to Sunday
            start_day = days_mapping[day]
            end_day = start_day  # Assume end day is the same for this example
            start_time = time(0, 0)  # Start of the day
            end_time = time(23, 59, 59)  # End of the day

            if day < 5:  # Weekdays
                # Apply different discount rates based on the time ranges
                for hour in range(24):
                    if hour < 10:  # From 00:00 to 10:00
                        discount_factor = 0.8
                    elif hour < 18:  # From 10:00 to 18:00
                        discount_factor = 1.2
                    else:  # From 18:00 to 23:59
                        discount_factor = 0.8
                    
                    # Calculate toll rates for each vehicle type
                    toll_rate_moto = row['moto'] * discount_factor
                    toll_rate_car = row['car'] * discount_factor
                    toll_rate_rv = row['rv'] * discount_factor
                    toll_rate_bus = row['bus'] * discount_factor
                    toll_rate_truck = row['truck'] * discount_factor
                    
                    toll_rates.append({
                        'id_start': id_start,
                        'id_end': id_end,
                        'start_day': start_day,
                        'end_day': end_day,
                        'start_time': start_time,
                        'end_time': end_time,
                        'moto': toll_rate_moto,
                        'car': toll_rate_car,
                        'rv': toll_rate_rv,
                        'bus': toll_rate_bus,
                        'truck': toll_rate_truck,
                    })
            
            else:  # Weekends
                discount_factor = 0.7
                # Calculate toll rates for each vehicle type
                toll_rate_moto = row['moto'] * discount_factor
                toll_rate_car = row['car'] * discount_factor
                toll_rate_rv = row['rv'] * discount_factor
                toll_rate_bus = row['bus'] * discount_factor
                toll_rate_truck = row['truck'] * discount_factor
                
                toll_rates.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': start_day,
                    'end_day': end_day,
                    'start_time': start_time,
                    'end_time': end_time,
                    'moto': toll_rate_moto,
                    'car': toll_rate_car,
                    'rv': toll_rate_rv,
                    'bus': toll_rate_bus,
                    'truck': toll_rate_truck,
                })

    # Create a new DataFrame from the toll rates list
    toll_rates_df = pd.DataFrame(toll_rates)

    return toll_rates_df

# Example usage
df = create_sample_dataframe()
result_df = calculate_time_based_toll_rates(df)

# Display the resulting DataFrame
print(result_df)
      




