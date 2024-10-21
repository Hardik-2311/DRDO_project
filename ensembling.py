import pandas as pd

# Load the existing CSV file
input_csv_file = 'final.csv'  # Replace with your existing CSV file name

# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv_file)

# Display the first few rows of the DataFrame (optional)
print("Original Data:")
print(df.head())

# Assuming the last column is the target class column
target_column = 'Class Identifier'  # Change this if your target column is named differently

def check_overlapping_values(df, target_column):
    overlaps = {}
    
    # Loop through each feature (column) except the target column
    for col in df.columns:
        if col != target_column:  # Exclude target column
            # Identify overlapping values
            unique_values = df[col].unique()
            for value in unique_values:
                # Get classes for the given value
                classes = df[df[col] == value][target_column].unique()
                if len(classes) > 1:
                    overlaps[value] = classes.tolist()  # Store overlapping value and corresponding classes
    
    return overlaps

# Check for overlapping values
overlapping_values = check_overlapping_values(df, target_column)

# Display results
if overlapping_values:
    print("Overlapping values found:")
    for value, classes in overlapping_values.items():
        print(f"Value: {value} -> Classes: {classes}")
else:
    print("No overlapping values found.")
