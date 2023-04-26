'''
What the backend sends
'''
# POST request
# /recommend_document
{ "topic_order": [['4', '0.5'], ['8', '0.3'], ['3', '0.2'], ['6', '0.1']], # The size of the list is not fixed, 
 
  "raw_text": "Appendix C: Regional  Climate Change Summaries",
  'document_id': 13,

    "topic": {
    '1': {
    "spans": [[0, 14], [[15, 25]], [29, 38]],
    "score": ['1.36e-05',  '1.36e-05', '0.002']
    }, 
    '2':  {
    "spans": [[100, 108], [110, 118], [124, 131]], 
    "scores": ['0.0002', '0.00025770828', '0.0017886487']
    } ,
    '3': {
    "spans": [[148, 159], [165, 174], [83, 187]], 
    "scores": ['2.31e-06', '0.008', '0.0006']
    }
    
    [[(148, 159), '2.31e-06'], [(165, 174), '0.008'], [(183, 187), '0.0006']]

    # The size of the topic is not exhaustive. The number of topics in this dictionary should be the number
    # of topics given the model. When doing processing, need to iterate through all the keys and values
    # in this dictionary
    },

    'topic_keywords': {
    'Topic 5': ['om', 'fm', 'okz', 'lm', 'ks'],
    'Topic 7': ['bh', 'rm', 'mt', 'mv', 'dy', 'rk', 'wa'],
    'Topic 10': ['mm', 'lj', 'sq', 'rl', 'gm', 'hj', 'om', 'fm', 'okz']
    }, # The most relevant topics for the current document with keywords
    "Model_prediction": "3" # Model's prediction on the topic for this document

}
    
# post THE FOLLOWING TO THE BACKEND first
# First time or user did not label and skip the current
# recommended document
{   
    "recommend": 1,
    "new_label": "None",
    "label": "3",
    "doc_id": -1,
    "response_time": -1
}

# Then
{   
    "recommend": -1,
    "new_label": "None",
    "label": "3",
    "doc_id": 13,
    "response_time": 3
}



