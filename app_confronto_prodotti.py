
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carica i dati
df = pd.read_excel("DATABASE.xlsx", sheet_name="Foglio5")

# Pre-elaborazione
df['DESCRIZIONE'] = df['DESCRIZIONE'].astype(str)

# Titolo
st.title("Confronto Prodotti Elettrici")
st.write("Ricerca intelligente e confronto di prodotti elettrici dal database.")

# Ricerca intelligente
query = st.text_input("Inserisci una descrizione o codice prodotto:")

if query:
    vectorizer = TfidfVectorizer().fit(df['DESCRIZIONE'])
    tfidf_matrix = vectorizer.transform(df['DESCRIZIONE'])
    query_vec = vectorizer.transform([query])

    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:5]
    results = df.iloc[top_indices].copy()
    results['SIMILARITÀ'] = similarities[top_indices]
    
    st.subheader("Risultati della ricerca:")
    st.dataframe(results[['PRODUCT ID', 'AZIENDA', 'DESCRIZIONE', 'PREZZO', 'SIMILARITÀ']])

    selected = st.multiselect("Seleziona i prodotti da confrontare:", results['PRODUCT ID'].tolist())

    if selected:
        confronto = df[df['PRODUCT ID'].isin(selected)]
        st.subheader("Confronto Prodotti Selezionati")
        st.dataframe(confronto.reset_index(drop=True))

# Filtro per azienda
st.sidebar.header("Filtri")
aziende = df['AZIENDA'].unique().tolist()
aziende_sel = st.sidebar.multiselect("Filtra per azienda:", aziende)

if aziende_sel:
    st.subheader("Prodotti Filtrati per Azienda")
    st.dataframe(df[df['AZIENDA'].isin(aziende_sel)])
