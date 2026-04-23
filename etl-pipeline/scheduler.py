import time
from datetime import datetime
from lab10.etl import run_etl

for i in range(6):
    print(f"Running ETL at {datetime.now()}")
    run_etl()

    if i < 5:
        time.sleep(600)