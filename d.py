import pandas as pd

# Load the two CSV files
csv1 = pd.read_csv('output.csv')
csv2 = pd.read_csv('train1.csv')

# Concatenate the two DataFrames
merged_df = pd.concat([csv1, csv2])

# Remove duplicates
cleaned_df = merged_df.drop_duplicates()

# Save the cleaned DataFrame to a new CSV
cleaned_df.to_csv('merged_cleaned.csv', index=False)

print("CSV files merged and duplicates removed!")
