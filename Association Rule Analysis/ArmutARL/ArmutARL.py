import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from mlxtend.frequent_patterns import apriori,association_rules 
pd.set_options('display.max_rows',50)

df = pd.read_csv('armut_data.csv')
df.info()

#UserId: Müşteri numarası

#ServiceId:

    #Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
    #Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
    #(Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)

#CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
#CreateDate: Hizmetin satın alındığı tarih

df.head()

df["Hizmet"] = [f"{i}_{j}" for i,j in zip(df["ServiceId"],df["CategoryId"])]
df["Tarih"] = pd.to_datetime(df["CreateDate"]).dt.strftime("%Y-%m")
df["SepetId"] = [f"{i}_{j}" for i,j in zip(df["UserId"],df["Tarih"])]

invoice_product_df = df.groupby(['SepetId',"Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

#Association Rule Learning

frequent_items = apriori(invoice_product_df, min_support = 0.01, use_colnames = True )
rules = association_rules( frequent_items, metric =  "support", min_threshold = 0.01)
rules.head()



#Sorgu

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules.sort_values("lift", ascending= False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items(): #frozenset
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]
arl_recommender(rules, "2_0",2)

arl_recommender(rules,"2_0", 4)