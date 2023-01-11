
import pandas as pd
#pip install openpyxl
#pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

#data = https://www.kaggle.com/datasets/vijayuv/onlineretail
df_ = pd.read_excel("\datasets\online_retail.xlsx", sheet_name="Year 2010-2011", engine= "openpyxl")

df = df_.copy()

post_index = df[df["StockCode"] == "POST"].index
#dataframe = dataframe[~dataframe.index.isin(post_index)].reset_index()
df.drop(post_index, inplace=True)


def checkid(df, id):
    print(df[df.StockCode == id]["Description"].values[0]) 


def outlier_threshold(dataframe, variable):
    Q1 = dataframe.quantity(0.01)
    Q3 = dataframe.quantity(0.95)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    return lower_limit, upper_limit

def replace_with_threshold(dataframe, variable):
    low_limit , up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit) , variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit) , variable] = up_limit

def retail_data_prep(dataframe):
    
    dataframe.dropna(inplace = True)
    dataframe = dataframe[~dataframe["invoiceId"].str.contains("C", na = False)] # for cancelled transaction
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_threshold(dataframe, "Quantity")
    replace_with_threshold(dataframe, "Price")
    return dataframe
    


def create_invoice_product_df(dataframe, id = False):
    if id:
        return dataframe.groupby(['Invoice',"StockCode"])["Quantity"].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice',"Description"])["Quantity"].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)



def create_rules(dataframe, id = True, country="Germany"):
    dataframe = dataframe[dataframe["Country"] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support= 0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric = "support", min_threshold = 0.01)
    return rules

rules = create_rules(df)

#Recommender


def arl_recommender(rules_df, product_id, rec_count):
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    return recommendation_list[0:rec_count]


#User 1's  basket id 21987
#User 2's  basket id 23235
#User 3's  basket id 22747


for i in [21987, 23235, 22747]:
    arl_recommender(rules, i, 1)

#antecedents
checkid(df, 21987) 
#PACK OF 6 SKULL PAPER CUPS

#consequents
checkid(df, 21086)
#SET/6 RED SPOTTY PAPER CUPS