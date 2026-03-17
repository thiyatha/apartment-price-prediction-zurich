import joblib
import pandas as pd
import gradio as gr

# Gespeichertes Modell laden
# Wichtig: Die Datei "best_model.joblib" muss im gleichen Ordner liegen wie app.py
model = joblib.load("best_model.joblib")


def predict_rent(living_space, rooms, floor, year_built, zip_code, balcony, parking):
    """
    Erstellt aus den Benutzereingaben ein DataFrame
    und gibt die Mietpreis-Vorhersage zurück.
    """

    # Neue Feature berechnen
    apartment_age = 2026 - year_built

    # Eingabedaten in genau der Struktur vorbereiten,
    # die auch beim Training verwendet wurde
    input_data = pd.DataFrame([{
        "living_space": living_space,
        "rooms": rooms,
        "floor": floor,
        "year_built": year_built,
        "apartment_age": apartment_age,
        "zip_code": str(zip_code),
        "balcony": balcony,
        "parking": parking
    }])

    # Vorhersage mit dem trainierten Modell
    prediction = model.predict(input_data)[0]

    # Schöne Ausgabe formatieren
    return f"Geschätzte Monatsmiete: CHF {prediction:,.2f}"


# Gradio-Interface definieren
demo = gr.Interface(
    fn=predict_rent,
    inputs=[
        gr.Number(label="Wohnfläche in m²"),
        gr.Number(label="Zimmer"),
        gr.Number(label="Stockwerk"),
        gr.Number(label="Baujahr"),
        gr.Textbox(label="PLZ"),
        gr.Dropdown(choices=["yes", "no"], label="Balkon"),
        gr.Dropdown(choices=["yes", "no"], label="Parkplatz")
    ],
    outputs=gr.Textbox(label="Vorhersage"),
    title="Apartment Price Prediction Zurich",
    description="Diese App schätzt die Monatsmiete einer Wohnung im Kanton Zürich."
)

# App starten
if __name__ == "__main__":
    demo.launch()
