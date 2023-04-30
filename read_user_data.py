import sqlite3

def read_all_recommendations():
    conn = sqlite3.connect('users.db')
    cursor = conn.execute('SELECT * FROM recommendations')
    rows = cursor.fetchall()

    for row in rows:
        # print(row)
        print('Recommendation ID: {}, User ID: {}, Label: {}, Doc ID: {}, Response Time: {}'.format(row[0], row[1], row[2], row[3], row[4]))
    
    conn.close()

read_all_recommendations()
