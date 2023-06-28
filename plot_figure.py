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
def read_all_recommendations(db_name, topic_info):
    if topic_info:
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id, topic_information FROM recommendations ')
            rows = cursor.fetchall()

            for row in rows:
                user_id = row[0]
                topic_information = row[1]

                print('user_id:', user_id)
                print('topic_information:', topic_information)
                print('---')
    else:
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id, label, doc_id, response_time FROM recommendations')
            rows = cursor.fetchall()

            for row in rows:
                user_id = row[0]
                label = row[1]
                doc_id = row[2]
                response_time = row[3]

                print('user_id:', user_id)
                print('label:', label)
                print('doc_id:', doc_id)
                print('response_time:', response_time)
                print('---')


def read_all_users(db_name):
    with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id, mode FROM users ')
            rows = cursor.fetchall()

            for row in rows:
                user_id = row[0]
                mode = row[1]

                print('user_id:', user_id)
                print('mode:', mode)
                print('---')

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
    for i, row in enumerate(rows):
        # print(row[0])
        accuracies.append(row[0])
        print('num doc: {}, acc: {}'.format(i, row[0]))
    
    print('-'* 10)


    return accuracies

def read_user_accuracy_np(data_path):
    array = np.load(data_path)
    return array


def plot_comparison(id1, id2, db_name, acc_type, id1_name, id2_name, save_path):
    session1_acc = read_accuracy_db(id1, db_name, acc_type)
    session2_acc = read_accuracy_db(id2, db_name, acc_type)
    print(session1_acc[0:20])
    print(session2_acc[0:20])
    start_index = 0

    if id1 % 4 == 0:
        session1_acc.pop()
    if id2 % 4 == 0:
        session2_acc.pop()

    for i in range(len(session1_acc)):
        if session1_acc[i] < 0 and i > start_index:
            start_index = i
    
    for j in range(len(session2_acc)):
        if session2_acc[j] < 0 and j > start_index:
            start_index = j 

    print(start_index)
    start_index += 1
    min_len = len(session1_acc) -start_index

    data_to_plot = {    
                'number documents labeled': [i+1+start_index for i in range(min_len)] * 2,
                'training acc': session1_acc[start_index:]+ session2_acc[start_index:],
                'model': [id1_name] * min_len + [id2_name] * min_len

                }

    plot(save_path, data_to_plot)



database_name = 'local_users.db'
database_name = 'server_users.db'
database_name = './database/06_26_2023_newsgroup_test.db'
database_name = './database/06_27_2023_newsgroup_test.db'
acc_type = 'local_training_acc'


# print_users_and_modes(database_name)
# read_all_recommendations(database_name, False)
# read_all_users(database_name)
# read_accuracy_db(5, database_name, acc_type)
# exit(0)
plot_comparison(1, 2, database_name, acc_type, 'LDA','SLDA', './plot_results/06_28_2023_20newsgroup/LDA_vs_SLDA.png')

'''
1: LDA concatenate features
2: SLDA concatenate features 
3: ETM concatenate features
4: Active Learning
5: LDA concatenate features
6: SLDA concatenate features fine tuining
7: ETM no concatenate features
8: Active Learning
9: LDA no concatenate features
10: SLDA no concatenate features
'''





'''
Original      Concat          min_cf       hand_pick_hyperparam        opt+min_cf
1: LA         31: LA         
2: LDA        32: LDA        
3: SLDA       33: SLDA       62: SLDA      55: SLDA                     72: SLDA
4: ETM        34: ETM        
'''

# # exit(0)
# LA_session_acc = read_accuracy_db(31, database_name, acc_type)
# LDA_session_acc = read_accuracy_db(32, database_name, acc_type)
# SLDA_session_acc = read_accuracy_db(33, database_name, acc_type)

# ETM_session_acc = read_accuracy_db(4, database_name, acc_type)
# concate_ETM_session_acc = read_accuracy_db(34, database_name, acc_type)
# logistic_acc = read_user_accuracy_np('./np_files/classifier_results.npy')


# # min_len = min(len(logistic_acc[0]), len(logistic_acc[1]), len(logistic_acc[2]), len(logistic_acc[3]), len(LA_session_acc), len(LDA_session_acc))
# # LDA_session_acc[2] = 0.05
# # LDA_session_acc[3] = 0.05
# print(ETM_session_acc[20:30])
# print(LA_session_acc[20:30])
# # data_to_plot = {
# #                 'number documents labeled': [i+1 for i in range(min_len)] * 5,

# #                 'training acc': LA_session_acc[2:] + logistic_acc[2].tolist() + LDA_session_acc[2:] + SLDA_session_acc[2:] + ETM_session_acc[2:],
# #                 'model': ['LA session'] * min_len + ['ordered acc'] * min_len + ['LDA session'] * min_len + ['SLDA session'] * min_len + ['ETM session'] * min_len
# #                 }

# min_len = min(len(LA_session_acc[2:]), len(ETM_session_acc))
# data_to_plot = {    
#                 'number documents labeled': [i+3 for i in range(min_len)] * 2,
#                 'training acc': concate_ETM_session_acc[2:]+ ETM_session_acc[2:],
#                 'model': ['concat ETM'] * min_len + ['ETM fitting'] * min_len

#                 }

# save_path = './plot_results/EMT0_vs_ETM_optimize.png'
# plot(save_path, data_to_plot)
# print(logistic_acc[2])