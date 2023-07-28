import sqlite3

DATABASE = './database/07_10_pilot_testing.db'  # Replace with your database file path
user_id = 16

with sqlite3.connect(DATABASE) as conn:
    cursor = conn.cursor()

    # Delete rows from the 'recommendations' table where user_id is 3
    cursor.execute("DELETE FROM recommendations WHERE user_id = {}".format(user_id))
    conn.commit()

    # Optionally, delete rows from the 'users' table where user_id is 3
    cursor.execute("DELETE FROM users WHERE user_id = {}".format(user_id))
    conn.commit()
