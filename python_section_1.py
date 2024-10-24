question 1 : reverse list by N elements


def reverse_in_groups(lst, n):
    result = []
    
    # Iterate through the list in steps of n
    for i in range(0, len(lst), n):
        group = []
        
        # Create the current group of n elements
        for j in range(n):
            if i + j < len(lst):
                group.append(lst[i + j])
        
        # Reverse the current group manually and add to result
        for j in range(len(group) - 1, -1, -1):
            result.append(group[j])
    
    return result

# Example usage:
input1 = [1, 2, 3, 4, 5, 6, 7, 8]
output1 = reverse_in_groups(input1, 3)
print(output1)  # Output: [3, 2, 1, 6, 5, 4, 8, 7]

input2 = [1, 2, 3, 4, 5]
output2 = reverse_in_groups(input2, 2)
print(output2)  # Output: [2, 1, 4, 3, 5]

input3 = [10, 20, 30, 40, 50, 60, 70]
output3 = reverse_in_groups(input3, 4)
print(output3)  # Output: [40, 30, 20, 10, 70, 60, 50]


question 2: lists & Dictionaries

def group_strings_by_length(strings):
    length_dict = {}
    
    for string in strings:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    
    # Sort the dictionary by keys and return as a tuple of items
    sorted_length_dict = dict(sorted(length_dict.items()))
    return sorted_length_dict

# Example usage:
input1 = ["apple", "bat", "car", "elephant", "dog", "bear"]
output1 = group_strings_by_length(input1)
print(output1)  # Output: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}

input2 = ["one", "two", "three", "four"]
output2 = group_strings_by_length(input2)
print(output2)  # Output: {3: ['one', 'two'], 4: ['four'], 5: ['three']}


Question 3: Flatten a Nested Dictionary

def flatten_dict(nested_dict, parent_key='', sep='.'):
    items = {}

    for key, value in nested_dict.items():
        # Create the new key for the flattened dictionary
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            # Recursively flatten the nested dictionary
            items.update(flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, list):
            # Iterate through the list and handle each item
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    # Flatten the dictionary in the list
                    items.update(flatten_dict(item, f"{new_key}[{index}]", sep=sep))
                else:
                    # Handle non-dictionary items in the list
                    items[f"{new_key}[{index}]"] = item
        else:
            # Add the key-value pair to the flattened dictionary
            items[new_key] = value

    return items

# Example usage
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened = flatten_dict(nested_dict)
print(flattened)



Queation 4 : Generate Unique permutation



def unique_permutations(nums):
    def backtrack(start):
        if start == len(nums):
            # If we have a complete permutation, add it to the results
            result.append(nums[:])
            return
        
        seen = set()  # To track duplicates at this level
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue  # Skip duplicates
            seen.add(nums[i])
            # Swap the current element with the start element
            nums[start], nums[i] = nums[i], nums[start]
            # Recur with the next element
            backtrack(start + 1)
            # Backtrack: swap back
            nums[start], nums[i] = nums[i], nums[start]

    result = []
    nums.sort()  # Sort to handle duplicates
    backtrack(0)
    return result

# Example usage
input_list = [1, 1, 2]
output = unique_permutations(input_list)
[1 1 2]
[1 2 1]
[2 1 1]


Question 5 : find all dates in a text

import re

def find_all_dates(text):
    # Define regex patterns for the different date formats
    patterns = [
        r'\b(\d{2})-(\d{2})-(\d{4})\b',  # dd-mm-yyyy
        r'\b(\d{2})/(\d{2})/(\d{4})\b',  # mm/dd/yyyy
        r'\b(\d{4})\.(\d{2})\.(\d{2})\b'   # yyyy.mm.dd
    ]
    
    # Combine patterns into a single regex pattern
    combined_pattern = '|'.join(patterns)
    
    # Find all matches in the text
    matches = re.findall(combined_pattern, text)
    
    # Flatten the list of tuples and filter out empty strings
    valid_dates = []
    for match in matches:
        # Join the matched groups into a valid date string
        valid_dates.append(''.join(match))
    
    return valid_dates

# Example usage
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output = find_all_dates(text)
print(output)  # Output: ['23-08-1994', '08/23/1994', '1994.08.23']


Queation 6 : decode polyline convert to dataframe with distances

import polyline
import pandas as pd
import numpy as np

def haversine(coord1, coord2):
    # Haversine formula to calculate distance between two points on the Earth's surface
    R = 6371000  # Radius of Earth in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert degrees to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def decode_polyline_and_create_dataframe(polyline_str):
    # Decode the polyline string
    coordinates = polyline.decode(polyline_str)

    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Calculate distances
    distances = [0]  # Start with 0 for the first point
    for i in range(1, len(coordinates)):
        distance = haversine(coordinates[i - 1], coordinates[i])
        distances.append(distance)
    
    df['distance'] = distances

    return df

# Example usage
polyline_str = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
df = decode_polyline_and_create_dataframe(polyline_str)
print(df)



queation 7 : matrix rotation and transformation


def rotate_and_transform_matrix(matrix):
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    # Step 2: Create a transformed matrix based on row and column sums
    transformed_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]  # sum of the row excluding the current element
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]  # sum of the column excluding the current element
            transformed_matrix[i][j] = row_sum + col_sum
    
    return transformed_matrix

# Example usage
input_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_transform_matrix(input_matrix)
print(result)


question 8 : time  check

import pandas as pd

def check_time_completeness(df):
    # Ensure timestamp columns are in datetime format
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Create a multi-index based on id and id_2
    df.set_index(['id', 'id_2'], inplace=True)

    # Group by (id, id_2) and check each group
    completeness_results = {}
    
    for (id_val, id_2_val), group in df.groupby(level=[0, 1]):
        # Get the full range of days (Monday to Sunday)
        full_days = set(pd.date_range(start=group['start'].min(), end=group['end'].max()).date)
        expected_days = {0, 1, 2, 3, 4, 5, 6}  # 0=Monday, ..., 6=Sunday
        
        # Check if there are entries for each day of the week
        days_covered = {date.weekday() for date in full_days}
        
        # Check if the time spans a full 24 hours
        start_min = group['start'].min().time()
        end_max = group['end'].max().time()
        
        # Check conditions
        complete_time = (start_min <= pd.Timestamp('00:00:00').time() and end_max >= pd.Timestamp('23:59:59').time())
        full_week = days_covered == expected_days
        
        completeness_results[(id_val, id_2_val)] = not (complete_time and full_week)

    # Convert the results into a boolean series with a multi-index
    completeness_series = pd.Series(completeness_results)
    
    return completeness_series

# Example usage:
# df = pd.read_csv('dataset-1.csv')
# result = check_time_completeness(df)
# print(result)













