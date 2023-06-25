import os
import json
import csv
import glob


def process_trial_files(directory):
    csv_data = []
    trial_files = glob.glob(os.path.join(directory, 'trial_*/trial.json'))

    for trial_file in trial_files:
        with open(trial_file, 'r') as file:
            data = json.load(file)

            row = {}
            row['trial_id'] = data.get('trial_id')
            row.update(data.get('hyperparameters', {}).get('values', {}))
            row.update({
                'accuracy':
                    data.get('metrics', {}).get('metrics', {}).get('accuracy', {}).get('observations', [{}])[0].get(
                        'value', [0])[0],
                'auc':
                    data.get('metrics', {}).get('metrics', {}).get('auc', {}).get('observations', [{}])[0].get('value',
                                                                                                               [0])[0],
                'time': os.path.getmtime(trial_file),  # replace this with your own time data if available
                'row_label': data.get('trial_id')  # add row_label that mirrors trial_id
            })

            csv_data.append(row)

    return csv_data


def write_to_csv(csv_data, output_file):
    if not csv_data:
        print('No data to write.')
        return

    # Get all field names from the data
    fieldnames = list(set().union(*[d.keys() for d in csv_data]))

    # Write data to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)


# process the trial files
csv_data = process_trial_files('my_dir/Stock_Trading_dense')

# write the data to CSV
write_to_csv(csv_data, 'output.csv')