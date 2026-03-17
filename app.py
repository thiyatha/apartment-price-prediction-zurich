import streamlit as st

# Seitentitel
st.set_page_config(page_title="Apartment Price Prediction Zurich", page_icon="🏠")

# Überschrift
st.title("🏠 Apartment Price Prediction Zurich")
st.write("Diese App schätzt den Mietpreis einer Wohnung im Kanton Zürich.")

# Eingaben
living_area = st.number_input("Wohnfläche in m²", min_value=10, max_value=500, value=60)
rooms = st.number_input("Anzahl Zimmer", min_value=1.0, max_value=10.0, value=2.5, step=0.5)
year_built = st.number_input("Baujahr", min_value=1800, max_value=2025, value=2005)
floor = st.number_input("Stockwerk", min_value=0, max_value=40, value=2)
zip_code = st.text_input("PLZ", value="8001")

# Neues Feature
apartment_age = 2025 - year_built

# Vorhersage
if st.button("Preis vorhersagen"):
    # Platzhalterformel bis dein echtes Modell eingebaut ist
    predicted_price = 900 + living_area * 24 + rooms * 190 + floor * 20 - apartment_age * 3

    st.success(f"Geschätzter Mietpreis: CHF {predicted_price:,.2f}")
    st.info(f"Neues Feature: Apartment Age = {apartment_age}")
