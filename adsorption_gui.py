# -*- coding: utf-8 -*-
"""
Streamlit GUI for XGBoost Adsorption Prediction with Molecule Search
"""

# 1. Import Streamlit FIRST and set page config
import streamlit as st
st.set_page_config(page_title="Adsorption Prediction", layout="centered")

# 2. Import other dependencies
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import requests
from time import sleep
import sys
import numpy as np

# 3. Load pre-trained XGBoost model (CPU-safe)
@st.cache_resource
def load_model():
    try:
        import xgboost as xgb
        st.info(f"Using XGBoost version: {xgb.__version__}")
        model = xgb.XGBRegressor(tree_method='hist', predictor='cpu_predictor')
        model.load_model("best_model_XGB_cpu.json")  # CPU-safe format
        return model
    except ImportError:
        st.error("XGBoost not installed. Add it to requirements.txt")
        st.stop()
    except FileNotFoundError:
        st.error("Model file 'best_model_XGB_cpu.json' not found in app directory.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

xgb_model = load_model()

# 4. Descriptor calculator setup
required_descriptors = ['fr_nitro', 'PEOE_VSA12', 'PEOE_VSA2', 'EState_VSA2']
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(required_descriptors)

# 5. Initialize session state
def init_session():
    st.session_state.setdefault('smiles_value', "CC(=O)NC1=CC=C(C=C1)O")  # Paracetamol
    st.session_state.setdefault('run_prediction', False)
    st.session_state.setdefault('last_searched', "")

init_session()

# 6. App Title and Intro
st.title("\U0001F9EA Adsorption Capacity Predictor")
st.markdown("Predict Qe (mg/g) using molecular descriptors and process conditions")

# 7. Sidebar Input Section
with st.sidebar:
    st.header("\u2697\ufe0f Input Parameters")

    st.subheader("\U0001F50D Search by Molecule Name")
    molecule_name = st.text_input("Enter molecule name", "Paracetamol")
    col1, col2 = st.columns(2)
    search_clicked = col1.button("Search PubChem")
    use_for_prediction = col2.button("Use for Prediction")

    if search_clicked:
        if molecule_name != st.session_state.last_searched:
            with st.spinner("Searching PubChem..."):
                try:
                    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{molecule_name}/property/CanonicalSMILES/JSON"
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                        st.session_state.smiles_value = smiles
                        st.session_state.last_searched = molecule_name
                        st.success(f"Found SMILES: {smiles}")
                    else:
                        st.error("Molecule not found in PubChem.")
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                sleep(0.5)
        else:
            st.info("Using previously found SMILES.")

    if use_for_prediction:
        st.session_state.run_prediction = True
        st.success("SMILES loaded for prediction!")

    smiles = st.text_input("SMILES Structure", st.session_state.smiles_value, key="smiles_input")
    st.session_state.smiles_value = smiles

    st.subheader("\u2699\ufe0f Process Conditions")
    temp = st.number_input("Temperature (K)", min_value=273.0, max_value=400.0, value=300.0)
    c0 = st.number_input("Initial Concentration (mg/L)", min_value=0.00, max_value=100.00, value=8.48)
    time = st.number_input("Time (min)", min_value=0.0, max_value=400.0, value=120.0)
    ce = st.number_input("Equilibrium Conc. (mg/L)", min_value=0.00, max_value=100.00, value=1.65)

    if ce >= c0:
        st.warning("Equilibrium concentration should be less than initial concentration!")

# 8. Prediction Logic
if st.button("Predict Adsorption Capacity") or st.session_state.run_prediction:
    try:
        st.session_state.run_prediction = False
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            st.error("Invalid SMILES structure.")
            st.stop()

        desc_values = calculator.CalcDescriptors(mol)
        desc_dict = dict(zip(required_descriptors, desc_values))

        input_df = pd.DataFrame([[
            desc_dict['fr_nitro'], desc_dict['PEOE_VSA12'],
            desc_dict['PEOE_VSA2'], desc_dict['EState_VSA2'],
            temp, c0, time, ce
        ]], columns=[
            'fr_nitro', 'PEOE_VSA12', 'PEOE_VSA2', 'EState_VSA2',
            'Temperature (T)', 'initial concentration (C0)',
            'Time (min)', 'ce(mg/L)'
        ])

        prediction = xgb_model.predict(input_df)[0]
        st.success(f"Predicted Adsorption Capacity: **{prediction:.2f} mg/g**")

        with st.expander("View Input Details"):
            st.write("**Molecular Descriptors:**")
            st.json(desc_dict)
            st.write("**Process Conditions:**")
            st.json({
                "Temperature": f"{temp} K",
                "Initial Concentration": f"{c0} mg/L",
                "Time": f"{time} min",
                "Equilibrium Concentration": f"{ce} mg/L"
            })

    except Exception as e:
        st.error(f"Prediction process failed: {str(e)}")

# 9. About Section
st.markdown("---")
st.subheader("About the Model")
st.write("""
This prediction system uses:
- **4 Molecular Descriptors** from chemical structure:
  - fr_nitro, PEOE_VSA12, PEOE_VSA2, EState_VSA2
- **Process Conditions**: Temperature, C0, Time, and Ce
- **XGBoost Regressor**: Trained on experimental adsorption data
""")

# 10. Instructions
st.markdown("---")
st.subheader("\U0001F50D Molecule Search Tips")
st.write("""
1. Enter names like "Paracetamol", "Caffeine", etc.
2. Click **Search PubChem** to fetch SMILES
3. Verify SMILES structure
4. Click **Use for Prediction**
5. Adjust process conditions if needed
""")

# 11. Debug Info
st.markdown("---")
st.subheader("Debug Information")
st.write(f"Python version: {sys.version}")
try:
    import xgboost
    st.write(f"XGBoost version: {xgboost.__version__}")
    st.write(f"Model type: {type(xgb_model)}")
except:
    st.write("XGBoost not available")
