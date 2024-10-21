import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv('balanced_data.csv')
df.drop('Image Name', axis=1, inplace=True)

# Print the DataFrame head and columns to check structure
print(df.head())
print("Remaining columns:", df.columns)

# Define the function to find overlapping p-values
def find_overlaps(df, class_column='Class Identifier'):
    overlaps = {}
    # Iterate through each column except the class column
    for col in df.columns:
        if col == class_column:  # Skip the class column
            continue
        
        unique_p_values = df[col].unique()
        overlap_results = {}
        
        # Check for overlaps of each unique p-value
        for value in unique_p_values:
            # Find classes that have the same p-value
            classes_with_value = df[df[col] == value][class_column].unique()
            if len(classes_with_value) > 1:  # More than one class has the same p-value
                overlap_results[value] = classes_with_value.tolist()

        overlaps[col] = overlap_results

    return overlaps

# Call the function
overlapping_p_values = find_overlaps(df)

# Print the results
print("Overlapping p-values across classes:")
for test, overlaps in overlapping_p_values.items():
    print(f"{test}: {overlaps}")
