import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st

# Load and preprocess the data
df_bakery = pd.read_csv('Bakery.csv')
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

# --- Helper Functions ---

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

# --- App Main Logic ---

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
