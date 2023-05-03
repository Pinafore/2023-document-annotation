import pandas as pd
from plotnine import ggplot, aes, geom_line, theme_minimal, ggtitle
import warnings
from plotnine.exceptions import PlotnineWarning
import sqlite3
import numpy as np


def plot(save_path, data_structure):
    # Example dataset
    key_list = list(data_structure.keys())
    x_axis = key_list[0]
    y_axis = key_list[1]
    color_group = key_list[2]

    df = pd.DataFrame(data_structure)

    # Create a line plot of training iterations vs. accuracy
    accuracy_plot = (
    ggplot(df, aes(x=x_axis, y=y_axis, color=color_group, group=color_group))
    + geom_line()
    + theme_minimal()
    + ggtitle('Numer of documents labeled vs. Training Accuracy for Multiple Models')
    )
    # Print the plot to the console
    print(accuracy_plot)

    # Save the plot to a file, suppressing the warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PlotnineWarning)
        accuracy_plot.save(save_path, dpi=300, width=6, height=4)


'''
Read all recommendations and print them on terminal
'''
def read_all_recommendations(db_name):
    conn = sqlite3.connect(db_name)
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


'''
print all the user ids and the mode used for the user
'''
def print_users_and_modes(db_name):
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, mode FROM users")
        rows = cursor.fetchall()

    print("User ID | Mode")
    print("--------+------")
    for row in rows:
        user_id, mode = row
        print(f"{user_id:7} | {mode}")


'''
Read the accuracy stores for a user
'''
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


# print_users_and_modes('local_users.db')
save_path = './plot_results/accuracy_plot.png'

database_name = 'local_users.db'
acc_type = 'global_training_acc'
LA_session_acc = read_accuracy_db(1, database_name, acc_type)
logistic_acc = read_user_accuracy_np('./np_files/classifier_results.npy')


min_len = min(len(logistic_acc[0]), len(logistic_acc[1]), len(logistic_acc[2]), len(logistic_acc[3]), len(LA_session_acc))


data_to_plot = {
                'number documents labeled': [i+1 for i in range(min_len)] * 2,

                'training acc': LA_session_acc[2:] + logistic_acc[2].tolist(),
                'model': ['LA session'] * min_len + ['ordered acc'] * min_len
                }


plot(save_path, data_to_plot)
# print(logistic_acc[2])