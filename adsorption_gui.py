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
import joblib
import requests  # For PubChem API access
import os  # For environment variables
from time import sleep  # For API rate limiting

# 3. Load pre-trained XGBoost model - using relative path
# Create a function with caching for model loading
@st.cache_resource
def load_model():
    try:
        # Use relative path for deployment
        return joblib.load("best_model_XGB.joblib")
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'best_model_XGB.joblib' is in the app directory.")
        st.stop()

xgb_model = load_model()

# 4. Configure required descriptors and calculator
required_descriptors = ['fr_nitro', 'PEOE_VSA12', 'PEOE_VSA2', 'EState_VSA2']
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(required_descriptors)

# Initialize session state variables
if 'smiles_value' not in st.session_state:
    st.session_state.smiles_value = "CC(=O)NC1=CC=C(C=C1)O"  # Default paracetamol SMILES
if 'run_prediction' not in st.session_state:
    st.session_state.run_prediction = False
if 'last_searched' not in st.session_state:
    st.session_state.last_searched = ""

# 5. App interface
st.title("🧪 Adsorption Capacity Predictor")
st.markdown("Predict Qe(mg/g) using molecular descriptors and process conditions")

# 6. Sidebar inputs
with st.sidebar:
    st.header("⚗️ Input Parameters")
    
    # Molecule search section
    st.subheader("🔍 Search by Molecule Name")
    molecule_name = st.text_input("Enter molecule name", "Paracetamol", 
                                help="Search PubChem database by common name")
    
    col1, col2 = st.columns(2)
    with col1:
        search_clicked = st.button("Search PubChem")
    with col2:
        use_for_prediction = st.button("Use for Prediction")
    
    if search_clicked:
        # Only search if the molecule name has changed
        if molecule_name != st.session_state.last_searched:
            with st.spinner("Searching PubChem..."):
                try:
                    # PubChem API request
                    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{molecule_name}/property/CanonicalSMILES/JSON"
                    response = requests.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Handle case where multiple results are returned
                        if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                            smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                            st.session_state.smiles_value = smiles
                            st.session_state.last_searched = molecule_name
                            st.success(f"Found SMILES: {smiles}")
                        else:
                            st.error("No properties found for the molecule.")
                    else:
                        st.error(f"Molecule not found (HTTP {response.status_code}). Please try a different name.")
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                sleep(0.5)  # Avoid API rate limiting
        else:
            st.info("Using previously found SMILES for this molecule.")
    
    if use_for_prediction:
        st.session_state.run_prediction = True
        st.success("SMILES loaded for prediction!")
    
    # SMILES input (now connected to session state)
    smiles = st.text_input("SMILES Structure", st.session_state.smiles_value, 
                         key="smiles_input",
                         help="Enter chemical structure using SMILES notation")
    
    # Update session state with any manual changes
    st.session_state.smiles_value = smiles
    
    # Process conditions
    st.subheader("⚙️ Process Conditions")
    temp = st.number_input("Temperature (K)", min_value=273.0, max_value=400.0, value=300.0)
    c0 = st.number_input("Initial Concentration (mg/L)", min_value=0.00, max_value=100.00, value=8.48)
    time = st.number_input("Time (min)", min_value=0.0, max_value=400.0, value=120.0)
    ce = st.number_input("Equilibrium Conc. (mg/L)", min_value=0.00, max_value=100.00, value=1.65)
    
    # Add input validation
    if ce >= c0:
        st.warning("Equilibrium concentration should be less than initial concentration!")

# 7. Prediction logic
if st.button("Predict Adsorption Capacity") or st.session_state.run_prediction:
    try:
        # Reset prediction trigger
        st.session_state.run_prediction = False
        
        # Validate and process SMILES
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            st.error("Invalid SMILES structure. Please check your input.")
            st.stop()
            
        # Add chemical structure visualization
        try:
            from rdkit.Chem import Draw
            img = Draw.MolToImage(mol, size=(300, 200))
            st.image(img, caption="Chemical Structure")
        except:
            st.info("Could not display molecular structure")
            
        # Calculate descriptors
        desc_values = calculator.CalcDescriptors(mol)
        desc_dict = dict(zip(required_descriptors, desc_values))
        
        # Create input DataFrame
        input_df = pd.DataFrame([[ 
            desc_dict['fr_nitro'],
            desc_dict['PEOE_VSA12'],
            desc_dict['PEOE_VSA2'],
            desc_dict['EState_VSA2'],
            temp,
            c0,
            time,
            ce
        ]], columns=[
            'fr_nitro', 'PEOE_VSA12', 'PEOE_VSA2', 'EState_VSA2',
            'Temperature (T)', 'initial concentration (C0)', 
            'Time (min)', 'ce(mg/L)'
        ])
        
        # Make prediction
        prediction = xgb_model.predict(input_df)[0]
        
        # Display results
        st.success(f"Predicted Adsorption Capacity: **{prediction:.2f} mg/g**")
        st.markdown("---")
        
        # Show input details
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
        st.error(f"Prediction failed: {str(e)}")

# 8. Add explanatory section
st.markdown("---")
st.subheader("About the Model")
st.write("""
This prediction system uses:
- **4 Key Molecular Descriptors** calculated from chemical structure:
  - Number of nitro groups (fr_nitro)
  - PEOE VSA descriptors (PEOE_VSA12, PEOE_VSA2)
  - EState VSA descriptor (EState_VSA2)
- **Process Conditions**: Temperature, Initial Concentration, Time, and Equilibrium Concentration
- **XGBoost Regressor**: Optimized machine learning model trained on experimental data
""")

# 9. Add search instructions
st.markdown("---")
st.subheader("🔍 Molecule Search Tips")
st.write("""
1. Enter common chemical names (e.g., "Paracetamol", "Caffeine", "Aspirin")
2. Click **Search PubChem** to fetch the SMILES structure
3. Verify the imported SMILES structure
4. Click **Use for Prediction** to run prediction with imported SMILES
5. Adjust process conditions as needed
""")