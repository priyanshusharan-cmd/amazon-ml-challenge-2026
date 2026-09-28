import sqlite3

conn = sqlite3.connect(':memory:')
try:
    conn.execute('CREATE VIRTUAL TABLE test_fts USING fts5(content);')
    conn.execute("INSERT INTO test_fts VALUES ('walmart store 123');")
    res = conn.execute("SELECT * FROM test_fts WHERE test_fts MATCH 'walmart';").fetchall()
    print("SQLite FTS5 IS AVAILABLE and WORKING! Result:", res)
except Exception as e:
    print("FTS5 not available:", e)
conn.close()
