import sqlite3

def check_db(name, path):
    print(f"=== {name}: {path} ===")
    conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    tables = [t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print("Tables:", tables)
    for t in tables:
        cols = cur.execute(f"PRAGMA table_info({t})").fetchall()
        count = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        sample = cur.execute(f"SELECT * FROM {t} LIMIT 1").fetchone()
        print(f"Table '{t}' ({count:,} rows):")
        for col in cols:
            print(f"   {col[1]} ({col[2]})")
        print("   Sample:", sample)
    conn.close()

if __name__ == '__main__':
    check_db("Train DB", "output/train_catalog_temp.db")
    print()
    check_db("Test DB", "output/test_catalog_temp.db")
