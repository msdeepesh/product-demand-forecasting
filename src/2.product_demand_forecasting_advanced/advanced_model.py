import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

F=["lag1","lag7","lag14","lag30","rm7","rm30","rs7","dow","month","doy","week","price","promo","marketing","market"]

def prep(d):
    d=d.copy(); d.date=pd.to_datetime(d.date); d.sku=d.sku.astype(str)
    d=d.sort_values(["sku","date"])
    for c,v in {"price":0,"promo":0,"marketing":0,"market":0,"lead_time":7,"incoming_supply":0,
                "holding_cost_per_unit":1,"stockout_cost_per_unit":5}.items():
        if c not in d: d[c]=v
    return d

def features(d):
    x=d.copy(); g=x.groupby("sku")
    for n in [1,7,14,30]: x[f"lag{n}"]=g.sales.shift(n)
    x["rm7"]=g.sales.transform(lambda s:s.shift().rolling(7,min_periods=3).mean())
    x["rm30"]=g.sales.transform(lambda s:s.shift().rolling(30,min_periods=7).mean())
    x["rs7"]=g.sales.transform(lambda s:s.shift().rolling(7,min_periods=3).std())
    x["dow"]=x.date.dt.dayofweek; x["month"]=x.date.dt.month; x["doy"]=x.date.dt.dayofyear
    x["week"]=x.date.dt.isocalendar().week.astype(int)
    return x

def score(y,p):
    e=np.asarray(y)-np.asarray(p)
    return {"MAE":mean_absolute_error(y,p),"RMSE":np.sqrt(mean_squared_error(y,p)),
            "WAPE":np.abs(e).sum()/max(np.abs(y).sum(),1e-9),
            "sMAPE":np.mean(2*np.abs(e)/np.maximum(np.abs(y)+np.abs(p),1e-9))}

def compare(d,days=30):
    x=features(prep(d)).dropna(subset=F+["sales"])
    cut=x.date.max()-pd.Timedelta(days=days); tr=x[x.date<=cut]; va=x[x.date>cut]
    out={}
    for name,m in [
        ("HistGradientBoosting",HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_leaf_nodes=31,random_state=42)),
        ("RandomForest",RandomForestRegressor(n_estimators=250,max_depth=12,min_samples_leaf=2,n_jobs=-1,random_state=42))]:
        m.fit(tr[F],tr.sales); p=np.maximum(0,m.predict(va[F]))
        out[name]={"model":m,"metrics":score(va.sales,p),"validation":va[["date","sku","sales"]].assign(prediction=p)}
    return out

def sarimax(d,sku,days=30):
    z=prep(d); y=z[z.sku==str(sku)].set_index("date").sales.asfreq("D").interpolate().fillna(0)
    cut=y.index.max()-pd.Timedelta(days=days); tr=y[y.index<=cut]; va=y[y.index>cut]
    m=SARIMAX(tr,order=(1,1,1),seasonal_order=(1,1,1,7),
              enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
    p=np.maximum(0,m.forecast(len(va)))
    return {"model":m,"metrics":score(va,p),"validation":pd.DataFrame({"date":va.index,"sku":sku,"sales":va.values,"prediction":p.values})}

def future_tree(m,d,sku,n):
    z=prep(d); z=z[z.sku==str(sku)].sort_values("date"); rows=z.to_dict("records"); last=z.date.max()
    for _ in range(n):
        day=last+pd.Timedelta(days=1); s=[r["sales"] for r in rows]; q=rows[-1]
        v={"lag1":s[-1],"lag7":s[-7],"lag14":s[-14],"lag30":s[-30],"rm7":np.mean(s[-7:]),
           "rm30":np.mean(s[-30:]),"rs7":np.std(s[-7:]),"dow":day.dayofweek,"month":day.month,
           "doy":day.dayofyear,"week":int(day.isocalendar().week),"price":q["price"],
           "promo":q["promo"],"marketing":q["marketing"],"market":q["market"]}
        y=max(0,float(m.predict(pd.DataFrame([v])[F])[0])); rows.append({"date":day,"sales":y}); last=day
    return pd.DataFrame(rows[-n:]).rename(columns={"sales":"forecast_demand"})

def plan(f,current,incoming,lead,service,review,std):
    z={.90:1.282,.95:1.645,.975:1.96,.99:2.326}[service]
    ss=z*std*np.sqrt(max(lead,1)); ld=f.forecast_demand.iloc[:min(lead,len(f))].sum()
    available=current+incoming; p=f.copy(); p["projected_inventory"]=available-p.forecast_demand.cumsum()
    rp=ld+ss; target=p.forecast_demand.iloc[:min(lead+review,len(p))].sum()+ss
    stock=np.maximum(0,-p.projected_inventory)
    return p,ss,rp,max(0,target-available),int((p.projected_inventory<0).sum()),stock.sum()
