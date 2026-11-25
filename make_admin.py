"""Promote a registered account to admin.

1. Register normally in the app (so the account exists).
2. Run:  python make_admin.py your_email@example.com
3. Log out and log in again. You will see the 'Admin Dashboard' tab.
"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "diabetes_app.db"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python make_admin.py your_email@example.com")
        return
    email = sys.argv[1].strip().lower()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute("UPDATE users SET role = 'admin' WHERE email = ?", (email,))
        conn.commit()
        if cur.rowcount:
            print(f"Done: {email} is now an admin. Log out and log in again.")
        else:
            print(f"No account found for {email}. Register in the app first.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()