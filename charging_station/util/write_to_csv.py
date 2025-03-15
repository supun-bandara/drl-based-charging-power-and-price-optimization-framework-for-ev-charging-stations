# write outputs to csv

import csv

def write_to_csv(data):
    file_path = "results.csv"

    file_exists = False
    try:
        with open(file_path, 'r'):
            file_exists = True
    except FileNotFoundError:
        pass

    # Open the CSV file in append mode
    with open(file_path, 'a', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        # If the file is newly created, write the header row
        if not file_exists:
            header = data.keys()
            csv_writer.writerow(header)

        # Write the data row
        csv_writer.writerow(data.values())
    
    return