import requests
import time
import csv

BASE = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
OUTPUT = "data/raw/nyc_311_full.csv"

LIMIT = 50000  # reasonable chunk size
offset = 0

# prepare the output file with headers
first = True

while True:
    params = {
        "$limit": LIMIT,
        "$offset": offset
    }
    response = requests.get(BASE, params=params)

    if response.status_code != 200:
        print("Failed at offset", offset, response.status_code)
        break

    text = response.text.strip().splitlines()
    if len(text) <= 1:
        print("No more data — done.")
        break

    if first:
        with open(OUTPUT, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in csv.reader(text):
                writer.writerow(row)
        first = False
    else:
        with open(OUTPUT, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            reader = csv.reader(text)
            next(reader)  # skip header row
            for row in reader:
                writer.writerow(row)


    print(f"Downloaded rows up to offset {offset}")
    offset += LIMIT

    time.sleep(1)  # gentle API rate limit
