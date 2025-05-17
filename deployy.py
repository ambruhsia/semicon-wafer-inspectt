import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from xgboost import Booster
import pandas as pd
# from uhm import preprocess_wafer_data

import pickle
import json
import numpy as np
from xgboost import XGBClassifier, Booster

# loading libraries
import skimage
from skimage import measure
from skimage.transform import radon
from skimage.transform import probabilistic_hough_line
from skimage import measure
from scipy import interpolate
from scipy import stats
 


def preprocess_wafer_data(x):
    """
    Preprocess wafer map data and return the feature matrix X.

    Parameters:
        x: List of wafer map arrays

    Returns:
        X: Numpy array of processed features
    """

    def cal_den(region):
        "Calculate density of regions with value 2."
        return 100 * (np.sum(region == 2) / np.size(region))

    def find_regions(wafer_map):
        "Divide the wafer map into regions and calculate density for each region."
        # wafer_map = np.array(eval(wafer_map))  # Convert string to array
        rows, cols = wafer_map.shape
        ind1 = np.arange(0, rows, rows // 5)
        ind2 = np.arange(0, cols, cols // 5)

        reg1 = wafer_map[ind1[0]:ind1[1], :]
        reg3 = wafer_map[ind1[4]:, :]
        reg4 = wafer_map[:, ind2[0]:ind2[1]]
        reg2 = wafer_map[:, ind2[4]:]

        reg5 = wafer_map[ind1[1]:ind1[2], ind2[1]:ind2[2]]
        reg6 = wafer_map[ind1[1]:ind1[2], ind2[2]:ind2[3]]
        reg7 = wafer_map[ind1[1]:ind1[2], ind2[3]:ind2[4]]
        reg8 = wafer_map[ind1[2]:ind1[3], ind2[1]:ind2[2]]
        reg9 = wafer_map[ind1[2]:ind1[3], ind2[2]:ind2[3]]
        reg10 = wafer_map[ind1[2]:ind1[3], ind2[3]:ind2[4]]
        reg11 = wafer_map[ind1[3]:ind1[4], ind2[1]:ind2[2]]
        reg12 = wafer_map[ind1[3]:ind1[4], ind2[2]:ind2[3]]
        reg13 = wafer_map[ind1[3]:ind1[4], ind2[3]:ind2[4]]

        fea_reg_den = [
            cal_den(reg1), cal_den(reg2), cal_den(reg3), cal_den(reg4),
            cal_den(reg5), cal_den(reg6), cal_den(reg7), cal_den(reg8),
            cal_den(reg9), cal_den(reg10), cal_den(reg11), cal_den(reg12), cal_den(reg13)
        ]
        return fea_reg_den
    

    def change_val(wafer_map):
        "Replace all values of 1 in the wafer map with 0."
        wafer_map[wafer_map == 1] = 0
        return wafer_map

    def cubic_inter_mean(wafer_map):
        "Perform cubic interpolation on the mean row of the sinogram."
        theta = np.linspace(0., 180., max(wafer_map.shape), endpoint=False)
        sinogram = radon(wafer_map, theta=theta)
        x_mean_row = np.mean(sinogram, axis=1)
        x = np.linspace(1, x_mean_row.size, x_mean_row.size)
        y = x_mean_row
        f = interpolate.interp1d(x, y, kind='cubic')
        xnew = np.linspace(1, x_mean_row.size, 20)
        ynew = f(xnew) / 100
        return ynew

    def cubic_inter_std(wafer_map):
        "Perform cubic interpolation on the standard deviation of the sinogram."
        theta = np.linspace(0., 180., max(wafer_map.shape), endpoint=False)
        sinogram = radon(wafer_map, theta=theta)
        x_std_row = np.std(sinogram, axis=1)
        x = np.linspace(1, x_std_row.size, x_std_row.size)
        y = x_std_row
        f = interpolate.interp1d(x, y, kind='cubic')
        xnew = np.linspace(1, x_std_row.size, 20)
        ynew = f(xnew) / 100
        return ynew

    def cal_dist(wafer_map, x, y):
        "Calculate the distance of a point from the center of the wafer map."
        dim0, dim1 = wafer_map.shape
        dist = np.sqrt((x - dim0 / 2) ** 2 + (y - dim1 / 2) ** 2)
        return dist

    def fea_geom(wafer_map):
        "Extract geometric features from the largest region in the wafer map."
        norm_area = wafer_map.shape[0] * wafer_map.shape[1]
        norm_perimeter = np.sqrt((wafer_map.shape[0]) ** 2 + (wafer_map.shape[1]) ** 2)

        img_labels = measure.label(wafer_map, connectivity=1, background=0)
        no_region = img_labels.max() - 1 if img_labels.max() > 0 else 0

        prop = measure.regionprops(img_labels)
        if no_region < len(prop):
            prop_area = prop[no_region].area / norm_area
            prop_perimeter = prop[no_region].perimeter / norm_perimeter
            prop_centroid = cal_dist(wafer_map, *prop[no_region].local_centroid)
            prop_majaxis = prop[no_region].major_axis_length / norm_perimeter
            prop_minaxis = prop[no_region].minor_axis_length / norm_perimeter
            prop_eccentricity = prop[no_region].eccentricity
            prop_solidity = prop[no_region].solidity
        else:
            # Fallback for edge cases
            prop_area, prop_perimeter, prop_majaxis, prop_minaxis = 0, 0, 0, 0
            prop_eccentricity, prop_solidity, prop_centroid = 0, 0, 0

        return prop_area, prop_perimeter, prop_majaxis, prop_minaxis, prop_eccentricity, prop_solidity

    # Preprocessing pipeline
    fea_reg = [find_regions(x)  ] 
    fea_cub_mean = [cubic_inter_mean(x)    ]
    fea_cub_std = [cubic_inter_std(x)  ]
    fea_geom = [fea_geom(x)  ]

    # Combine all features into a single array
    a = np.array(fea_reg)
    b = np.array(fea_cub_mean)
    c = np.array(fea_cub_std)
    d = np.array(fea_geom)

    X = np.concatenate((a, b, c, d), axis=1)
    return X


def plot_wafer_map(wafer_map):
    fig, ax = plt.subplots(figsize=(3, 3))  # Adjust figure size
    ax.imshow(wafer_map, cmap='gray', interpolation='nearest')
    ax.set_title("Wafer Map Input", fontsize=8)  # Reduce title font size
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Center the plot in a narrow container
    col1, col2, col3 = st.columns([1, 2, 1])  
    with col2:
        st.pyplot(fig)


st.set_page_config(page_title="WaferMap Defect Detection", layout="wide", page_icon="🔬")

filename = 'savemodle3.sav'
model = pickle.load(open(filename, 'rb'))

if isinstance(model, Booster):
    model.save_model('saved_model.json')
    model_loaded = Booster()
    model_loaded.load_model('saved_model.json')
else:
    st.error("Ready")

st.sidebar.header('📌 About')
st.sidebar.info("This tool helps semiconductor engineers detect defects in wafer maps using an XGBoost classifier.")

st.sidebar.header('🛠 How to Use')
st.sidebar.markdown("1️⃣ Enter wafer map data\n2️⃣ Click '🔍 Predict'\n3️⃣ View results and visualization.")

st.sidebar.header('📞 Contact')
st.sidebar.markdown("✉️ rajputamrita54@gmail.com")

st.title("🔬 WaferMap Defect Detection")
st.markdown("Predict defects in wafer maps using a pre-trained XGBoost model.")
st.header("📝 Input Wafer Map Data")

x = st.text_area("Enter Wafer Map Data", height=150, placeholder="[[0, 1, 0], [1, 0, 1], [0, 1, 0]]")


if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.wafer_shape = None

def predict():
    try:
        wafer_map = np.array(json.loads(x))
        st.session_state.wafer_shape = wafer_map.shape
        preprocessed_data = preprocess_wafer_data(wafer_map)
        st.session_state.prediction = model.predict(preprocessed_data)[0]
        st.session_state.wafer_map = wafer_map  # Store for plotting
    except (json.JSONDecodeError, ValueError) as e:
        st.session_state.prediction = f"⚠️ Error: Invalid input format. Please enter a valid JSON array. ({e})"
        st.session_state.wafer_map = None

st.button("🔍 Predict", on_click=predict)
mapping_type = {
    'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4,
    'Random': 5, 'Scratch': 6, 'Near-full': 7, 'None': 8
}

# Convert mapping dictionary to DataFrame for better visualization
mapping_df = pd.DataFrame(list(mapping_type.items()), columns=['Defect Type', 'Label'])

# Display Mapping Table
st.header("🗺️ Defect Type Mapping")
st.table(mapping_df)


if st.session_state.prediction is not None:
    st.markdown("---")
    if isinstance(st.session_state.prediction, str) and "Error" in st.session_state.prediction:
        st.error(st.session_state.prediction)
    else:
        st.success(f"✅ Input converted to array with shape: `{st.session_state.wafer_shape}`")
        st.info(f"📊 Predicted Defect Category: `{st.session_state.prediction}`")
        
        if st.session_state.wafer_map is not None:
            plot_wafer_map(st.session_state.wafer_map)

st.markdown("---")
