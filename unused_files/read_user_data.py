import sqlite3
import numpy as np
from plot_figure import plot

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


def read_accuracy_db(user_id, database, accuracy_type):
    conn = sqlite3.connect(database)
    
    cursor = conn.execute('SELECT {} FROM recommendations WHERE user_id = ?'.format(accuracy_type), (user_id,))

    # Fetch all rows with user_id = 1
    rows = cursor.fetchall()

    conn.close()

    accuracies = []
    for row in rows:
        # print(row[0])
        accuracies.append(row[0])

    return accuracies

def read_user_accuracy_np(data_path):
    array = np.load(data_path)
    return array



save_path = './plot_results/accuracy_plot.png'
# Call the function to print user and mode information
# read_all_recommendations()
# print_users_and_modes()
database_name = 'local_users.db'
acc_type = 'global_training_acc'
LA_session_acc = read_accuracy_db(1, database_name, acc_type)
logistic_acc = read_user_accuracy_np('./np_files/classifier_results.npy')

# print(len(LA_session_acc))
print(len(logistic_acc[0]), len(logistic_acc[1]), len(logistic_acc[2]), len(logistic_acc[3]), len(LA_session_acc))

'''
The following data 
'''