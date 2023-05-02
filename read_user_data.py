import sqlite3

def read_all_recommendations():
    conn = sqlite3.connect('local_users.db')
    cursor = conn.execute('SELECT * FROM recommendations')
    rows = cursor.fetchall()

    for i, row in enumerate(rows):
        # print(row)
        print('Recommendation ID: {}, User ID: {}, Label: {}, Doc ID: {}, Response Time: {}'.format(row[0], row[1], row[2], row[3], row[4]))
        # print(row[5])
        print(row[6])
        print(row[7])
        print(row[8])
        print(row[9])
        # if i == 2:
        #     break

    conn.close()


def print_users_and_modes():
    with sqlite3.connect('local_users.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, global_training_acc FROM users")
        rows = cursor.fetchall()

    print("User ID | global_training_acc")
    print("--------+------")
    for row in rows:
        user_id, mode = row
        print(f"{user_id:7} | {mode}")

# Call the function to print user and mode information
# read_all_recommendations()
# print_users_and_modes()

import sqlite3

def create_connection(database='local_users.db'):
    conn = sqlite3.connect(database)
    return conn

conn = create_connection()
cursor = conn.execute('SELECT global_training_acc FROM recommendations WHERE user_id = ?', (1,))

# Fetch all rows with user_id = 1
rows = cursor.fetchall()

# If you want to fetch only the first row with user_id = 1, use cursor.fetchone() instead
# row = cursor.fetchone()

# Close the connection
conn.close()

# Print the global_training_acc values for user_id = 1
for row in rows:
    print(row[0])


