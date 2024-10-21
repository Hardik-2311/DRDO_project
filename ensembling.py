import pandas as pd

# Load the existing CSV file
input_csv_file = 'sorted_output_with_img_name.csv'  # Replace with your existing CSV file name

# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv_file)

# Display the first few rows of the DataFrame (optional)
print("Original Data:")
print(df.head())

# Assuming the last column is the target class column
target_column = 'Class Identifier'  # Change this if your target column is named differently

def adjust_overlapping_values(df, target_column):
    # Create a copy of the DataFrame to avoid modifying the original
    adjusted_df = df.copy()
    
    # Loop through each feature (column) except the target column
    for col in df.columns:
        if col != target_column:  # Exclude target column
            # Identify overlapping values
            unique_values = df[col].unique()
            for value in unique_values:
                # Get classes for the given value
                classes = df[df[col] == value][target_column].unique()
                if len(classes) > 1:
                    print(f"Overlapping value found: {value} for classes: {classes.tolist()}")
                    # Adjust values for classes except the first one
                    for idx, class_id in enumerate(classes):
                        if idx == 0:
                            continue  # Retain the original value for the first class
                        else:
                            # Calculate new value with a 0.001 difference
                            new_value = value + 0.001 * idx  # Ensure it's different from the original
                            # Update the adjusted DataFrame
                            adjusted_df.loc[
                                (adjusted_df[col] == value) & 
                                (adjusted_df[target_column] == class_id), 
                                col
                            ] = new_value
                            print(f"Adjusted value: {value} to {new_value} for class: {class_id}")

    return adjusted_df

# Adjust overlapping values
adjusted_data = adjust_overlapping_values(df, target_column)

# Display results
print("Adjusted Data:")
print(adjusted_data.head())

# Save the adjusted data to a new CSV file (optional)
output_csv_file = 'adjusted_output.csv'  # Name of the output CSV file
adjusted_data.to_csv(output_csv_file, index=False)

print(f"Adjusted data saved to {output_csv_file}.")
