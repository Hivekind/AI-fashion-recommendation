import csv

# Define the expected number of columns (based on the header)
EXPECTED_COLUMNS = 10  # Change this if your CSV has a different number of columns

def find_malformed_rows(csv_file):
    malformed_rows = []
    
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for line_number, row in enumerate(reader, start=1):
            if len(row) != EXPECTED_COLUMNS:
                malformed_rows.append((line_number, row))

    return malformed_rows

csv_file = 'sql-init-scripts/dataset/products.csv'
malformed_rows = find_malformed_rows(csv_file)

# Output the malformed rows with line numbers
if malformed_rows:
    print(f"Found {len(malformed_rows)} malformed rows:")
    for line_number, row in malformed_rows:
        print(f"Line {line_number}: {row}")
else:
    print("No malformed rows found. All rows have the expected column count.")
