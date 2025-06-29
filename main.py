import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st
import os
from datetime import datetime

CSV_FILE = "Bakery.csv"

df_bakery = pd.read_csv(CSV_FILE)
df_bakery['DateTime'] = pd.to_datetime(df_bakery['DateTime'], format="%Y-%m-%d %H:%M:%S")
df_bakery['Month'] = df_bakery['DateTime'].dt.month
df_bakery['Day'] = df_bakery['DateTime'].dt.weekday

# Use short month names to match the Streamlit slider input
df_bakery['Month'].replace(
    [i for i in range(1, 13)],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
     "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"],
    inplace=True
)
df_bakery['Day'].replace(
    list(range(7)), 
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    inplace=True
)

df_bakery['Items'] = df_bakery['Items'].astype(str)

st.title("🛒 Market Basket Analysis using Apriori Algorithm")

def get_data(period_day='', weekday_weekend='', month='', day=''):
    data = df_bakery.copy()
    filtered_transactions = data.loc[
        (data['Daypart'].str.contains(period_day, case=False)) &
        (data['DayType'].str.contains(weekday_weekend, case=False)) &
        (data['Month'].str.contains(month, case=False)) &
        (data['Day'].str.contains(day, case=False))
    ]
    return filtered_transactions if not filtered_transactions.empty else None

def user_input():
    item = st.selectbox('Item', ["Bread", "Scandinavian", "Hot Chocolate", "Jam"])
    period_day = st.selectbox('Period Day', ["Morning", "Evening", "Afternoon", "Night"])
    weekday_weekend = st.selectbox('Weekday / Weekend', ["Weekday", "Weekend"])
    month = st.select_slider('Month', ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"])
    day = st.select_slider('Day', ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], value='Sat')
    return item, period_day, weekday_weekend, month, day

def encode(x):
    return 1 if x >= 1 else 0

def parse_list(x):
    x = list(x)
    return ", ".join(x) if len(x) > 1 else x[0]

def return_item_df(item_antecedents, association_df):
    data = association_df[["antecedents", "consequents"]].copy()
    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    row = data.loc[data["antecedents"] == item_antecedents]
    if not row.empty:
        return list(row.iloc[0, :])
    else:
        return [item_antecedents, "No recommendation found"]

def get_daypart(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

def get_daytype(dt):
    return "Weekend" if dt.weekday() >= 5 else "Weekday"

def get_new_transaction_no():
    if os.path.isfile(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        return df["TransactionNo"].max() + 1
    return 1



item, period_day, weekday_weekend, month, day = user_input()
data = get_data(period_day, weekday_weekend, month, day)

if data is not None:
    st.subheader("Filtered Transactions Preview")
    st.dataframe(data)

    item_count = data.groupby(["TransactionNo", "Items"])["Items"].count().reset_index(name='Count')
    item_count_pivot = item_count.pivot_table(index='TransactionNo', columns='Items', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    frequent_itemsets = apriori(item_count_pivot, min_support=0.01, use_colnames=True)
    association_rules_df = association_rules(frequent_itemsets, metric="lift", min_threshold=1)[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ]
    association_rules_df.sort_values('confidence', ascending=False, inplace=True)

    # st.subheader("Association Rules")
    # st.dataframe(association_rules_df)

    st.markdown("### 🔍 Recommendation Result")
    result = return_item_df(item, association_rules_df)
    st.success(f"If a customer buys **{result[0]}**, they may also buy **{result[1]}**.")
else:
    st.warning("⚠ No transactions matched your filter. Try different options.")


if "new_items" not in st.session_state:
    st.session_state.new_items = []

if "edit_index" not in st.session_state:
    st.session_state.edit_index = None


# ----- NEW ITEM TRANSACTION ENTRY FORM -----

st.title("📝 New Food Transaction Entry")

if st.session_state.edit_index is None:
    with st.form(key="add_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            item_name = st.selectbox('Item', ["Bread", "Scandinavian", "Hot Chocolate", "Jam"])
        with col2:
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1)

        add = st.form_submit_button("➕ Add Item")
        if add:
            if item_name.strip():
                st.session_state.new_items.append({
                    "Item": item_name.strip(),
                    "Quantity": int(quantity)
                })
            else:
                st.warning("Item name cannot be empty.")
else:
    # --- Edit Form ---
    item_to_edit = st.session_state.new_items[st.session_state.edit_index]
    with st.form(key="edit_form"):
        st.info(f"Editing item: {item_to_edit['Item']}")
        new_item = st.text_input("Food Item Name", value=item_to_edit['Item'])
        new_quantity = st.number_input("Quantity", min_value=1, value=item_to_edit['Quantity'], step=1)
        update = st.form_submit_button("💾 Save")
        cancel = st.form_submit_button("❌ Cancel")

        if update:
            st.session_state.new_items[st.session_state.edit_index] = {
                "Item": new_item.strip(),
                "Quantity": int(new_quantity)
            }
            st.session_state.edit_index = None
            st.success("Item updated.")
        elif cancel:
            st.session_state.edit_index = None

if st.session_state.new_items:
    st.subheader("🧾 Current Items in This Transaction")
    for i, item in enumerate(st.session_state.new_items):
        col1, col2, col3 = st.columns([3, 1, 2])
        col1.write(item["Item"])
        col2.write(f"x{item['Quantity']}")
        
        with col3:
            col_edit, col_delete = st.columns([1, 1])
            if col_edit.button("✏️", key=f"edit_{i}"):
                st.session_state.edit_index = i
                st.rerun()
            if col_delete.button("🗑️", key=f"delete_{i}"):
                st.session_state.new_items.pop(i)
                st.rerun()

     # --- Submit and Clear Buttons Row ---
    col_submit, col_clear = st.columns(2)
    if col_submit.button("✅ Submit Transaction"):
        now = datetime.now()
        transaction_no = get_new_transaction_no()
        date_str = now.strftime("%Y-%m-%d %H:%M:%S")
        daypart = get_daypart(now.hour)
        daytype = get_daytype(now)

        rows = []
        for item in st.session_state.new_items:
            for _ in range(item["Quantity"]):
                rows.append({
                    "TransactionNo": transaction_no,
                    "Items": item["Item"],
                    "DateTime": date_str,
                    "Daypart": daypart,
                    "DayType": daytype
                })

        df_new = pd.DataFrame(rows)
        file_exists = os.path.isfile(CSV_FILE)
        df_new.to_csv(CSV_FILE, mode="a", index=False, header=not file_exists)

        st.success(f"Transaction {transaction_no} saved with {len(rows)} item(s)")
        st.session_state.new_items = []
    if col_clear.button("❌ Clear Table"):
        st.session_state.new_items = []
        st.success("All items have been cleared.")
        st.rerun()

else:
    st.info("Add at least one item to submit.")
