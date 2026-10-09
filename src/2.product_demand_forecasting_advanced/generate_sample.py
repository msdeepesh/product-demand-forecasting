import numpy as np,pandas as pd
r=np.random.default_rng(42); dates=pd.date_range("2023-01-01","2025-12-31")
o=[]
for i,sku in enumerate(["A100","B200","C300"]):
 for d in dates:
  promo=int(r.random()<.08); base=[80,140,220][i]; season=1+.2*np.sin(2*np.pi*d.dayofyear/365.25)
  sales=max(0,r.normal(base*season*(1.25 if promo else 1)*(.85 if d.dayofweek>=5 else 1),base*.12))
  o.append({"date":d,"sku":sku,"sales":sales,"inventory":base*7,"price":[100,150,220][i],
            "promo":promo,"marketing":500 if promo else 100,"market":100,"lead_time":7+i*3,"incoming_supply":0})
pd.DataFrame(o).to_csv("sample_demand_inventory.csv",index=False)
