import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
import implicit

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Quick-Commerce AI", layout="wide")
st.title("🛒 Quick-Commerce Recommendation Engine")

# --- 2. DATA CACHING (Loads models only once to keep the website fast) ---
@st.cache_resource
def load_model():
    return implicit.cpu.als.AlternatingLeastSquares.load('als_model.npz')

@st.cache_data
def load_data():
    products = pd.read_csv('../data/raw/products.csv', usecols=['product_id', 'product_name'])
    product_dict = dict(zip(products['product_id'], products['product_name']))
    
    orders = pd.read_csv('../data/raw/orders.csv', usecols=['order_id', 'user_id'])
    train_data = pd.read_csv('../data/raw/order_products__train.csv', usecols=['order_id', 'product_id'])
    user_purchases = orders.merge(train_data, on='order_id')
    
    return product_dict, user_purchases

with st.spinner("Booting up AI Engine..."):
    model = load_model()
    product_dict, user_purchases = load_data()

# --- 3. THE UI SIDEBAR ---
st.sidebar.header("User Session Setup")
selected_user = st.sidebar.number_input("Simulate Login (User ID):", min_value=1, max_value=500, value=1)
available_items = list(product_dict.items())[:100] # Limit to 100 for dropdown speed
selected_cart_item = st.sidebar.selectbox("Add Item to Cart:", options=[item[0] for item in available_items], format_func=lambda x: product_dict[x])

# --- 4. RECOMMENDATION LOGIC ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔄 Buy Again")
    user_history = user_purchases[user_purchases['user_id'] == selected_user]
    if user_history.empty:
        st.write("No past purchase history found.")
    else:
        top_staples = user_history['product_id'].value_counts().head(5)
        for rank, (prod_id, count) in enumerate(top_staples.items(), 1):
            st.write(f"**{rank}.** {product_dict.get(prod_id, 'Unknown')} *(Bought {count}x)*")

with col2:
    st.subheader("✨ New Discoveries (ALS AI)")
    matrix_user_id = selected_user - 1 # Offset because user IDs start at 1
    user_traits = model.user_factors[matrix_user_id]
    product_traits = model.item_factors
    match_scores = user_traits.dot(product_traits.T)
    
    top_5_indices = np.argsort(match_scores)[::-1][:5]
    for rank, idx in enumerate(top_5_indices, 1):
        st.write(f"**{rank}.** {product_dict.get(idx + 1, 'Unknown')} *(Score: {match_scores[idx]:.2f})*")

with col3:
    st.subheader("💡 frequently Bought Together")
    orders_with_item = user_purchases[user_purchases['product_id'] == selected_cart_item]['order_id']
    co_occurring = user_purchases[user_purchases['order_id'].isin(orders_with_item)]
    top_associations = co_occurring[co_occurring['product_id'] != selected_cart_item]['product_id'].value_counts().head(3)
    
    if top_associations.empty:
        st.write("No strong associations found.")
    else:
        for rank, (prod_id, count) in enumerate(top_associations.items(), 1):
            st.write(f"**{rank}.** {product_dict.get(prod_id, 'Unknown')} *(Co-occurred {count}x)*")