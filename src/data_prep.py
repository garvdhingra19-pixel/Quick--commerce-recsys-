import pandas as pd
import numpy as np

def load_and_split_data(data_dir='../data/raw'):
    """
    Loads Instacart data and splits it by time to prevent data leakage.
    Training set: All prior orders.
    Test set: The user's most recent order.
    """
    dtype_dict = {
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'product_id': np.int32,
        'reordered': np.int8
    }
    
    print("Loading datasets...")
    orders = pd.read_csv(f'{data_dir}/orders.csv', dtype=dtype_dict, usecols=['order_id', 'user_id', 'eval_set', 'order_number', 'order_hour_of_day'])
    prior_items = pd.read_csv(f'{data_dir}/order_products__prior.csv', dtype=dtype_dict, usecols=['order_id', 'product_id', 'reordered'])
    test_items = pd.read_csv(f'{data_dir}/order_products__train.csv', dtype=dtype_dict, usecols=['order_id', 'product_id', 'reordered'])
    
    print("Merging data to create time-based Train/Test splits...")
    
    # Train Data: All historical orders for each user
    prior_orders = orders[orders['eval_set'] == 'prior']
    train_data = prior_items.merge(prior_orders, on='order_id')
    
    # Test Data: The single most recent order for each user (to evaluate our predictions)
    test_orders = orders[orders['eval_set'] == 'train']
    test_data = test_items.merge(test_orders, on='order_id')
    
    print("Split complete!")
    return train_data, test_data, orders