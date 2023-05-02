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
    with sqlite3.connect('users.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, mode FROM users")
        rows = cursor.fetchall()

    print("User ID | Mode")
    print("--------+------")
    for row in rows:
        user_id, mode = row
        print(f"{user_id:7} | {mode}")

# Call the function to print user and mode information
read_all_recommendations()
# print_users_and_modes()

