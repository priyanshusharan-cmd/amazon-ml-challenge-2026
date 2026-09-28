import sqlite3
import pandas as pd
import time

conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE catalog (entity_id TEXT, data TEXT)")
conn.execute("CREATE INDEX idx ON catalog(entity_id)")

# insert 100k
print("inserting")
conn.executemany("INSERT INTO catalog VALUES (?, ?)", [(str(i), "data") for i in range(100000)])

needed = [str(i) for i in range(50000)]
t0 = time.time()
conn.execute("CREATE TEMPORARY TABLE temp_ids (entity_id TEXT)")
conn.executemany("INSERT INTO temp_ids VALUES (?)", [(n,) for n in needed])
df = pd.read_sql("SELECT c.* FROM catalog c INNER JOIN temp_ids t ON c.entity_id = t.entity_id", conn)
conn.execute("DROP TABLE temp_ids")
t1 = time.time()

print(f"Fetched {len(df)} rows in {t1-t0:.2f} seconds")
