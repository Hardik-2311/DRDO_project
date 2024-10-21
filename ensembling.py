import pandas as pd

# Load the existing CSV file
input_csv_file = 'combined_output.csv'  # Replace with your existing CSV file name
output_csv_file = 'balanced_data.csv'  # Name of the output CSV file

# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv_file)

# Display the first few rows of the DataFrame (optional)
print("Original Data:")
print(df.head())

# Assuming the last column is the target class column
target_column = 'Class Identifier' 
  # Change this if your target column is named differently

# Create a new DataFrame to store the sampled data
sampled_data = pd.DataFrame()

# Loop through each class and sample 200 unique entries
for class_label in df[target_column].unique():
    class_data = df[df[target_column] == class_label]
    # Sample 200 unique entries without replacement
    sampled_class_data = class_data.sample(n=1650, random_state=42)  # Set random_state for reproducibility
    sampled_data = pd.concat([sampled_data, sampled_class_data])

# Reset index of the final DataFrame
sampled_data.reset_index(drop=True, inplace=True)

# Save the sampled data to a new CSV file
sampled_data.to_csv(output_csv_file, index=False)

print(f"Sampled data saved to {output_csv_file}.")
