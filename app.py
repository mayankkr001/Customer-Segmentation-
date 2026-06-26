import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation Pro",layout="wide")
st.title("🛍️ Customer Segmentation Pro")
f=st.file_uploader("Upload CSV",type="csv")
if f:
    df=pd.read_csv(f)
    st.dataframe(df.head())
    cols=st.multiselect("Select numeric features",df.select_dtypes("number").columns.tolist(),default=df.select_dtypes("number").columns[:2].tolist())
    if len(cols)>=2:
        k=st.slider("Clusters",2,10,5)
        X=df[cols]
        model=KMeans(n_clusters=k,random_state=42,n_init=10)
        df["Cluster"]=model.fit_predict(X)
        st.dataframe(df)
        fig=px.scatter(df,x=cols[0],y=cols[1],color="Cluster")
        st.plotly_chart(fig,use_container_width=True)
        st.download_button("Download CSV",df.to_csv(index=False),"segmented.csv")
