import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier, Pool
import plotly.graph_objects as go
import plotly.express as px
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left-color: #4caf50;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">❤️ Heart Disease Risk Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown("### Interactive tool to assess heart disease risk based on clinical and lifestyle factors")


FEATURE_DEFINITIONS = {
    # --- Demographics ---
    '_AGE80': { 
        'label': 'Age', 
        'type': 'slider', 
        'min': 18, 'max': 80, 'default': 45, 
        'help': 'What is your age? (Ages 80 and older are grouped as 80)', 
        'category': 'Demographics' 
    },
    '_SEX': { 
        'label': 'Biological Sex', 
        'type': 'radio', 
        'options': {'Male': 1, 'Female': 2}, 
        'default': 'Male', 
        'help': 'Sex assigned at birth', 
        'category': 'Demographics' 
    },
    '_IMPRACE': { 
        'label': 'Race/Ethnicity', 
        'type': 'selectbox', 
        'options': { 
            'White, Non-Hispanic': 1, 
            'Black, Non-Hispanic': 2, 
            'Asian, Non-Hispanic': 3, 
            'American Indian/Alaskan Native': 4, 
            'Hispanic': 5, 
            'Other race, Non-Hispanic': 6 
        }, 
        'default': 'White, Non-Hispanic', 
        'help': 'Which category best describes your racial or ethnic background?', 
        'category': 'Demographics' 
    },
    '_RACEG21': {
        'label': 'General Race Group',
        'type': 'selectbox',
        'options': {'White, Non-Hispanic': 1, 'Non-White or Hispanic': 2, 'Prefer not to say': 9},
        'default': 'White, Non-Hispanic',
        'help': 'Broad racial/ethnic classification',
        'category': 'Demographics'
    },
    'EDUCA': { 
        'label': 'Education Level', 
        'type': 'selectbox', 
        'options': { 
            'Never attended school or only kindergarten': 1, 
            'Grades 1 through 8 (Elementary)': 2, 
            'Grades 9 through 11 (Some high school)': 3, 
            'High school graduate (or GED)': 4, 
            'Some college or technical school': 5, 
            'College graduate': 6,
            'Prefer not to say': 9 
        }, 
        'default': 'High school graduate (or GED)', 
        'help': 'What is the highest grade or year of school you completed?', 
        'category': 'Demographics' 
    },
    'MARITAL': { 
        'label': 'Marital Status', 
        'type': 'selectbox', 
        'options': { 
            'Married': 1, 'Divorced': 2, 'Widowed': 3, 
            'Separated': 4, 'Never married': 5, 'Unmarried couple': 6, 
            'Prefer not to say': 9
        }, 
        'default': 'Married', 
        'help': 'Are you: (married, divorced, widowed, separated, never married, or a member of an unmarried couple)?', 
        'category': 'Demographics' 
    },
    'VETERAN3': { 
        'label': 'Military Service', 
        'type': 'radio', 
        'options': {'Yes': 1, 'No': 2, 'Don\'t Know': 7, 'Prefer not to say': 9}, 
        'default': 'No', 
        'help': 'Have you ever served on active duty in the United States Armed Forces?', 
        'category': 'Demographics' 
    },
    'EMPLOY1': { 
        'label': 'Employment Status', 
        'type': 'selectbox', 
        'options': { 
            'Employed for wages': 1, 'Self-employed': 2, 'Out of work for 1 year or more': 3, 
            'Out of work for less than 1 year': 4, 'A homemaker': 5, 'A student': 6, 
            'Retired': 7, 'Unable to work': 8, 'Prefer not to say': 9 
        }, 
        'default': 'Employed for wages', 
        'help': 'Which of the following best describes your current employment situation?', 
        'category': 'Demographics' 
    },
    'RENTHOM1': { 
        'label': 'Housing Status', 
        'type': 'selectbox', 
        'options': {'Own': 1, 'Rent': 2, 'Other arrangement': 3, 'Don\'t Know': 7, 'Prefer not to say': 9}, 
        'default': 'Own', 
        'help': 'Do you own or rent your home?', 
        'category': 'Demographics' 
    },
    '_INCOMG1': { 
        'label': 'Annual Household Income', 
        'type': 'selectbox', 
        'options': { 
            'Less than $15,000': 1, '$15,000 to < $25,000': 2, '$25,000 to < $35,000': 3, 
            '$35,000 to < $50,000': 4, '$50,000 to < $75,000': 5, '$75,000 or more': 6, 'Prefer not to say': 9 
        }, 
        'default': '$35,000 to < $50,000', 
        'help': 'What is your annual household income from all sources?', 
        'category': 'Demographics' 
    },
    
    # --- Clinical Features ---
    # CRITICAL FIX: Codebook says 1=Yes, 3=No. Original code had 1=No, 2=Yes.
    'DIABETE4': { 
        'label': 'Diabetes Diagnosis', 
        'type': 'selectbox', 
        'options': { 
            'Yes': 1, 'Yes (during pregnancy)': 2, 'No': 3, 'No, pre-diabetes or borderline': 4,
            'Don\'t Know': 7, 'Prefer not to say': 9 
        }, 
        'default': 'No', 
        'help': 'Have you ever been told by a doctor that you have diabetes?', 
        'category': 'Clinical' 
    },
    'PHYSHLTH': { 
        'label': 'Days of Poor Physical Health', 
        'type': 'slider', 
        'min': 1, 'max': 30, 'default': 0, # Codebook max is 30. 88=None (0).
        'help': 'For how many days during the past 30 days was your physical health not good? (Enter 0 for None)', 
        'category': 'Clinical' 
    },
    # CRITICAL FIX: Codebook says 1=Yes, 2=No. Original code had 1=No, 2=Yes.
    '_DRDXAR2': { 
        'label': 'Arthritis Diagnosis', 
        'type': 'radio', 
        'options': {'Yes': 1, 'No': 2}, 
        'default': 'No', 
        'help': 'Have you ever been told by a doctor that you have some form of arthritis, rheumatoid arthritis, gout, lupus, or fibromyalgia?', 
        'category': 'Clinical' 
    },
    'WTKG3': { 
        'label': 'Weight (Metric Code)', 
        'type': 'slider', 
        'min': 2300, 'max': 29500, 'default': 7000, 
        'help': 'Weight in kilograms with 2 implied decimal places (e.g., 7000 = 70.00 kg).', 
        'category': 'Clinical' 
    },
    '_BMI5CAT': { 
        'label': 'BMI Category', 
        'type': 'selectbox', 
        'options': { 'Underweight': 1, 'Normal Weight': 2, 'Overweight': 3, 'Obese': 4 }, 
        'default': 'Normal Weight', 
        'help': 'Body Mass Index (BMI) category derived from height and weight.', 
        'category': 'Clinical' 
    },
    '_RFBMI5': { 
        'label': 'Overweight or Obese Status', 
        'type': 'radio', 
        'options': {'No': 1, 'Yes': 2, 'Don\'t know/Prefer not to say': 9}, 
        'default': 'No', 
        'help': 'Calculated status: Do you have a BMI greater than 25.00?', 
        'category': 'Clinical' 
    },
    'DECIDE':{ 
        'label': 'Difficulty Concentrating', 
        'type': 'radio', 
        'options': { 'Yes': 1, 'No': 2, 'Don\'t Know': 7, 'Prefer not to say': 9}, 
        'default': 'No', 
        'help': 'Do you have serious difficulty concentrating, remembering, or making decisions?', 
        'category': 'Clinical' 
    },  
    
    # --- Lifestyle - Smoking ---
    'SMOKE100': { 
        'label': 'Smoked 100+ Cigarettes', 
        'type': 'radio', 
        'options': {'Yes': 1, 'No': 2, 'Don\'t Know': 7, 'Prefer not to say': 9}, 
        'default': 'No', 
        'help': 'Have you smoked at least 100 cigarettes in your entire life?', 
        'category': 'Lifestyle - Smoking' 
    },
    '_SMOKER3': { 
        'label': 'Current Smoking Status', 
        'type': 'selectbox', 
        'options': { 
            'Current smoker - every day': 1, 
            'Current smoker - some days': 2, 
            'Former smoker': 3, 
            'Never smoked': 4,
            'Prefer not to say': 9 
        }, 
        'default': 'Never smoked', 
        'help': 'Four-level smoker status', 
        'category': 'Lifestyle - Smoking' 
    },
    'SMOKDAY2': { 
        'label': 'Frequency of Smoking', 
        'type': 'selectbox', 
        'options': {
            'Every day': 1, 'Some days': 2, 'Not at all': 3,
            'Not applicable (Non-smoker)': 0, 'Don\'t Know': 7, 'Prefer not to say': 9
        },
        'default': 'Not applicable (Non-smoker)', 
        'help': 'Do you now smoke cigarettes every day, some days, or not at all?', 
        'category': 'Lifestyle - Smoking' 
    },
    'USENOW3': { 
        'label': 'Smokeless Tobacco Use', 
        'type': 'selectbox', 
        'options': {'Every day': 1, 'Some days': 2, 'Not at all': 3, 'Don\'t Know': 7, 'Prefer not to say': 9}, 
        'default': 'Not at all', 
        'help': 'Do you currently use chewing tobacco, snuff, or snus?', 
        'category': 'Lifestyle - Smoking' 
    },
    'LASTSMK2': { 
        'label': 'Time Since Last Smoked', 
        'type': 'selectbox', 
        'options': { 
            'Never smoked/Not applicable': 0, 
            'Within past month': 1, 'Within past 3 months': 2, 'Within past 6 months': 3, 
            'Within past year': 4, 'Within past 5 years': 5, 'Within past 10 years': 6, '10 years or more': 7, 
            'Never smoked regularly': 8, 'Don\'t Know': 77, 'Prefer not to say': 99 
        }, 
        'default': 'Never smoked/Not applicable', 
        'help': 'How long has it been since you last smoked a cigarette?', 
        'category': 'Lifestyle - Smoking' 
    },
    '_PACKDAY': { 
        'label': 'Daily Packs of Cigarettes', 
        'type': 'slider', 
        'min': 0, 'max': 100, 'default': 0, 
        'help': 'Number of packs of cigarettes smoked per day (calculated).', 
        'category': 'Lifestyle - Smoking' 
    },
    '_PACKYRS': { 
        'label': 'Total Pack-Years', 
        'type': 'slider', 
        'min': 0, 'max': 999, 'default': 0, 
        'help': 'Calculated: Years smoked multiplied by packs per day.', 
        'category': 'Lifestyle - Smoking' 
    },
    
    # --- Lifestyle - Alcohol ---
    'DRNKANY6': { 
        'label': 'Alcohol Consumption', 
        'type': 'radio', 
        'options': {'Yes': 1, 'No': 2, 'Don\'t Know': 7, 'Prefer not to say': 9}, 
        'default': 'No', 
        'help': 'During the past 30 days, did you have at least one drink of any alcoholic beverage?', 
        'category': 'Lifestyle - Alcohol' 
    },
    'ALCDAY4': { 
        'label': 'Alcohol Frequency Code', 
        'type': 'slider', 
        'min': 100, 'max': 999, 'default': 888, 
        'help': 'Format: 1xx (Days/Week), 2xx (Days/Month). Example: 105 = 5 days/week. 205 = 5 days/month. 888 = None.', 
        'category': 'Lifestyle - Alcohol' 
    },
    '_DRNKWK3': { 
        'label': 'Drinks Per Week', 
        'type': 'slider', 
        'min': 0, 'max': 9990, 'default': 0, 
        'help': 'Calculated total number of alcoholic beverages consumed per week.', 
        'category': 'Lifestyle - Alcohol' 
    },
    '_RFDRHV9': { 
        'label': 'Heavy Drinker Status', 
        'type': 'radio', 
        'options': {'No': 1, 'Yes': 2, 'Prefer not to say': 9}, 
        'default': 'No', 
        'help': 'Calculated: Men >14 drinks/week, Women >7 drinks/week.', 
        'category': 'Lifestyle - Alcohol' 
    },
    
    # --- Lifestyle - Physical Activity ---
    '_TOTINDA': { 
        'label': 'Physical Activity', 
        'type': 'radio', 
        'options': {'Had physical activity': 1, 'No physical activity': 2, 'Prefer not to say': 9}, 
        'default': 'Had physical activity', 
        'help': 'During the past 30 days, did you participate in any physical activities or exercises?', 
        'category': 'Lifestyle - Activity' 
    },
    '_PHYS14D': { 
        'label': 'Physical Health Status Level', 
        'type': 'selectbox', 
        'options': {
            'Zero days when physical health not good': 1,
            '1-13 days when physical health not good': 2,
            '14+ days when physical health not good': 3,
            'Don\'t know/Prefer not to say': 9
        },
        'default': 'Zero days when physical health not good', 
        'help': 'Three-level grouping of physical health status.', 
        'category': 'Lifestyle - Activity' 
    },
    
    # --- Additional flags ---
    '_AGE65YR': { 
        'label': 'Age 65+', 
        'type': 'radio', 
        'options': {'Age 18 to 64': 1, 'Age 65 or older': 2, 'Prefer not to say': 3}, 
        'default': 'Age 18 to 64', 
        'help': 'Two-level age category.', 
        'category': 'Demographics' 
    },
    '_ADULT': { 
        'label': 'Adult Respondent', 
        'type': 'radio', 
        'options': {'Yes': 1, 'No': 0}, 
        'default': 'Yes', 
        'help': 'Are you 18 years of age or older?', 
        'category': 'Demographics' 
    },
    '_RFSMOK3': { 
        'label': 'Current Smoker Flag', 
        'type': 'radio', 
        'options': {'No': 1, 'Yes': 2, 'Prefer not to say': 9}, 
        'default': 'No', 
        'help': 'Calculated variable: Adults who are current smokers.', 
        'category': 'Lifestyle - Smoking' 
    },
    'LCSLAST_': { 
        'label': 'Age Last Smoked', 
        'type': 'slider', 
        'min': 0, 'max': 100, 'default': 0, 
        'help': 'How old were you when you last smoked cigarettes regularly?', 
        'category': 'Clinical' 
    },
    'LCSNUMC_': { 
        'label': 'Average Cigarettes Per Day', 
        'type': 'slider', 
        'min': 0, 'max': 300, 'default': 0, 
        'help': 'On average, about how many cigarettes did you usually smoke each day?', 
        'category': 'Clinical' 
    },
    '_LCSYSMK': { 
        'label': 'Years Smoked', 
        'type': 'slider', 
        'min': 0, 'max': 100, 'default': 0, 
        'help': 'Total number of years you smoked cigarettes.', 
        'category': 'Lifestyle - Smoking' 
    },
    '_LCSSMKG': { 
        'label': 'Lung Cancer Screening Group', 
        'type': 'selectbox', 
        'options': { 
            'Current smoker, 20+ Pack Years': 1, 
            'Former smoker, 20+ Pack Years, quit < 15 yrs': 2, 
            'Current smoker, < 20 Pack Years': 3, 
            'Former smoker, 20+ Pack Years, quit >= 15 yrs': 4,
            'Former smoker, < 20 Pack Years': 5,
            'Never smoker': 6
        }, 
        'default': 'Never smoker', 
        'help': 'Smoking status grouping for lung cancer screening.', 
        'category': 'Lifestyle - Smoking' 
    },
    '_MRACE1': { 
        'label': 'Multiracial Identity', 
        'type': 'selectbox', 
        'options': {
            'White only': 1, 
            'Black or African American only': 2, 
            'American Indian or Alaskan Native only': 3, 
            'Asian Only': 4, 
            'Native Hawaiian or other Pacific Islander only': 5, 
            'Other race only': 6, 
            'Multiracial': 7, 
            'Don\'t Know': 77, 
            'Prefer not to say': 99
        }, 
        'default': 'White only', 
        'help': 'Calculated multiracial race categorization.', 
        'category': 'Demographics' 
    },
    '_HISPANC': { 
        'label': 'Hispanic Origin', 
        'type': 'radio', 
        'options': {'Hispanic, Latino/a, or Spanish origin': 1, 'Not of Hispanic, Latino/a, or Spanish origin': 2, 'Prefer not to say': 9}, 
        'default': 'Not of Hispanic, Latino/a, or Spanish origin', 
        'help': 'Are you of Hispanic, Latino/a, or Spanish origin?', 
        'category': 'Demographics' 
    },
    '_RACE': { 
        'label': 'Race/Ethnicity (Detailed)', 
        'type': 'selectbox', 
        'options': { 
            'White only, non-Hispanic': 1, 
            'Black only, non-Hispanic': 2, 
            'American Indian or Alaskan Native only, Non-Hispanic': 3, 
            'Asian only, non-Hispanic': 4, 
            'Native Hawaiian or other Pacific Islander only, Non-Hispanic': 5, 
            'Other race only, non-Hispanic': 6, 
            'Multiracial, non-Hispanic': 7, 
            'Hispanic': 8 
        }, 
        'default': 'White only, non-Hispanic', 
        'help': 'Detailed race/ethnicity categories.', 
        'category': 'Demographics' 
    },
    '_RACEGR3': { 
        'label': 'Race/Ethnicity (5 Groups)', 
        'type': 'selectbox', 
        'options': { 
            'White only, Non-Hispanic': 1, 
            'Black only, Non-Hispanic': 2, 
            'Other race only, Non-Hispanic': 3, 
            'Multiracial, Non-Hispanic': 4, 
            'Hispanic': 5, 
            'Prefer not to say': 9 
        }, 
        'default': 'White only, Non-Hispanic', 
        'help': 'Five-level race/ethnicity category.', 
        'category': 'Demographics' 
    },
    '_LCSYQTS': { 
        'label': 'Years Since Quitting Smoking', 
        'type': 'slider', 
        'min': 0, 'max': 100, 'default': 0, 
        'help': 'Number of years since you last smoked cigarettes.', 
        'category': 'Lifestyle - Smoking' 
    },
    '_RFBING6': { 
        'label': 'Binge Drinker Status', 
        'type': 'radio', 
        'options': {'No': 1, 'Yes': 2, 'Prefer not to say': 9}, 
        'default': 'No', 
        'help': 'Calculated: Having 5+ drinks (men) or 4+ drinks (women) on one occasion.', 
        'category': 'Lifestyle - Alcohol' 
    }
}

# Initialize session state for storing predictions
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []



# ADD THIS SECTION HERE - Before any widgets are created
# Define preset profiles
LOW_RISK_PROFILE = {
    '_AGE80': 35, '_SEX': 2, 'DIABETE4': 1, 'PHYSHLTH': 0,
    '_DRDXAR2': 1, 'WTKG3': 7000, '_BMI5CAT': 2, '_RFBMI5': 1,
    'SMOKE100': 2, '_SMOKER3': 4, 'SMOKDAY2': 0, 'USENOW3': 3, # SMOKDAY2 0=Not Applicable
    '_PACKYRS': 0, '_TOTINDA': 1, 
    '_PHYS14D': 1, # Changed from 20 to 1 (Zero days when health not good)
    'DRNKANY6': 2, 'ALCDAY4': 888, '_DRNKWK3': 0, '_RFDRHV9': 1,
    'EDUCA': 6, '_INCOMG1': 6, 'EMPLOY1': 1, 'MARITAL': 1,
    'VETERAN3': 2, 'RENTHOM1': 1, '_ADULT': 1, '_AGE65YR': 1,
    'LASTSMK2': 0, '_RFSMOK3': 1, 'LCSLAST_': 0, 'LCSNUMC_': 0,
    '_LCSYSMK': 0, '_LCSSMKG': 1, '_MRACE1': 1, '_HISPANC': 2,
    '_RACE': 1, '_RACEGR3': 1, '_LCSYQTS': 0, '_RFBING6': 1,
    '_IMPRACE': 1, 'DECIDE': 2
}

HIGH_RISK_PROFILE = {
    '_AGE80': 70, '_SEX': 1, 'DIABETE4': 2, 'PHYSHLTH': 20,
    '_DRDXAR2': 2, 'WTKG3': 11000, '_BMI5CAT': 4, '_RFBMI5': 2,
    'SMOKE100': 1, '_SMOKER3': 1, 
    'SMOKDAY2': 1, # Changed from 30 to 1 (Every day)
    'USENOW3': 1,
    '_PACKYRS': 50, '_TOTINDA': 2, 
    '_PHYS14D': 3, # Changed from 0 to 3 (14+ days when health not good)
    'DRNKANY6': 1, 'ALCDAY4': 220, '_DRNKWK3': 30, '_RFDRHV9': 2,
    'EDUCA': 3, '_INCOMG1': 2, 'EMPLOY1': 7, 'MARITAL': 3,
    'VETERAN3': 1, 'RENTHOM1': 2, '_ADULT': 1, '_AGE65YR': 2,
    'LASTSMK2': 1, '_RFSMOK3': 2, 'LCSLAST_': 55, 'LCSNUMC_': 40,
    '_LCSYSMK': 40, '_LCSSMKG': 4, '_MRACE1': 1, '_HISPANC': 2,
    '_RACE': 1, '_RACEGR3': 1, '_LCSYQTS': 10, '_RFBING6': 2,
    '_IMPRACE': 1, 'DECIDE': 1
}

# Initialize profile selection flags
if 'load_low_risk' not in st.session_state:
    st.session_state.load_low_risk = False
if 'load_high_risk' not in st.session_state:
    st.session_state.load_high_risk = False
if 'reset_defaults' not in st.session_state:
    st.session_state.reset_defaults = False

# Apply profile if flag is set (before widgets are created)
if st.session_state.load_low_risk:
    for key, value in LOW_RISK_PROFILE.items():
        st.session_state[key] = value
    st.session_state.load_low_risk = False

if st.session_state.load_high_risk:
    for key, value in HIGH_RISK_PROFILE.items():
        st.session_state[key] = value
    st.session_state.load_high_risk = False

if st.session_state.reset_defaults:
    for feature, config in FEATURE_DEFINITIONS.items():
        default = config['options'][config['default']] if config['type'] in ['radio', 'selectbox'] else config['default']
        st.session_state[feature] = default
    st.session_state.reset_defaults = False
# device =  'cuda' if torch.cuda.is_available() else 'cpu'



class NeuralNetMetaLearner(nn.Module):
    """
    Neural Network meta-learner for stacking ensemble predictions
    """
    def __init__(self, n_base_models, n_classes, hidden_dims=[64, 32]):
        super(NeuralNetMetaLearner, self).__init__()
        
        input_dim = n_base_models * n_classes
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], n_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class StackingEnsemble:
    """
    Stacking ensemble with XGBoost, CatBoost, LightGBM and Neural Network meta-learner
    """
    def __init__(self, n_folds=5, n_classes=2, random_state=42):
        self.n_folds = n_folds
        self.n_classes = n_classes
        self.random_state = random_state
        self.base_models = []
        self.meta_model = None
        self.scaler = StandardScaler()
        
    def _get_base_models(self):
        """Initialize base models"""
        return {
            'xgb': XGBClassifier(scale_pos_weight=scale_pos_weight, enable_categorical=True, tree_method='hist', random_state=self.random_state,eval_metric='logloss'),
            # xgb.XGBClassifier(
            #     n_estimators=300,
            #     max_depth=6,
            #     learning_rate=0.05,
            #     subsample=0.8,
            #     colsample_bytree=0.8,
            #     random_state=self.random_state,
            #     eval_metric='logloss',
            #     use_label_encoder=False
            # ),
            'catboost': CatBoostClassifier(auto_class_weights='Balanced', verbose=False),
            
            # CatBoostClassifier(
            #     iterations=300,
            #     depth=6,
            #     learning_rate=0.05,
            #     random_seed=self.random_state,
            #     verbose=False
            # ),
            'lgbm': LGBMClassifier(class_weight='balanced',random_state=self.random_state,verbose=-1)#LGBMClassifier(
            #     n_estimators=300,
            #     max_depth=6,
            #     learning_rate=0.05,
            #     subsample=0.8,
            #     colsample_bytree=0.8,
            #     random_state=self.random_state,
            #     verbose=-1
            # )
        }
    
    def _get_oof_predictions(self, X, y):
        """
        Generate out-of-fold predictions for meta-learner training
        """
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Initialize OOF prediction arrays
        oof_preds = {name: np.zeros((len(X), self.n_classes)) for name in ['xgb', 'catboost', 'lgbm']}
        
        # Store trained models for each fold
        fold_models = {name: [] for name in ['xgb', 'catboost', 'lgbm']}
        
        # print(f"Generating out-of-fold predictions with {self.n_folds}-fold CV...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # print(f"\nFold {fold + 1}/{self.n_folds}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            models = self._get_base_models()
            
            for name, model in models.items():
                # Train model
                if "cat" in name:
                    model.fit(X_train,y_train,cat_features = categorical_features_indices)
                else:
                    model.fit(X_train, y_train)
                
                # Get predictions
                val_preds = model.predict_proba(X_val)
                oof_preds[name][val_idx] = val_preds
                
                # Store model
                fold_models[name].append(model)
                
                # Print metrics
                acc = accuracy_score(y_val, val_preds.argmax(axis=1))
                f1 = f1_score(y_val, val_preds.argmax(axis=1), average='macro')
                print(f"  {name}: Accuracy = {acc:.4f} | F1 Score = {f1}")
        
        # Store fold models
        self.base_models = fold_models
        
        # Concatenate all base model predictions
        oof_meta_features = np.concatenate([oof_preds[name] for name in ['xgb', 'catboost', 'lgbm']], axis=1)
        
        return oof_meta_features
    
    def _train_meta_learner(self, meta_features, y, epochs=100, batch_size=64):
        """
        Train neural network meta-learner
        """
        print("\nTraining Neural Network meta-learner...")
        
        # Scale features
        meta_features_scaled = self.scaler.fit_transform(meta_features)

        counts = np.bincount(y)
        total = counts.sum()
        weights = total / (len(counts) * counts)
        class_weights = {0: weights[0], 1: weights[1]}

        
        self.meta_model = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=0.001, weight_decay=1e-5),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params=dict(mode='min', patience=5, factor=0.5)
        )#.to(device)

        self.meta_model.fit(
            X_train=meta_features_scaled,
            y_train=y,
            max_epochs=50,
            weights=np.array([class_weights[label] for label in y]),
            batch_size=1024,

        )
    
    def fit(self, X, y):
        """
        Fit the stacking ensemble
        """
        # print("=" * 60)
        # print("STACKING ENSEMBLE TRAINING")
        # print("=" * 60)
        
        # Generate out-of-fold predictions
        meta_features = self._get_oof_predictions(X, y)
        
        # Train meta-learner
        self._train_meta_learner(meta_features, y)
        
        # print("\n" + "=" * 60)
        # print("TRAINING COMPLETED")
        # print("=" * 60)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        """
        # Get predictions from all base models (averaged across folds)
        base_preds = []
        
        for name in ['xgb', 'catboost', 'lgbm']:
            fold_preds = []
            for model in self.base_models[name]:
                fold_preds.append(model.predict_proba(X))
            # Average predictions across folds
            avg_pred = np.mean(fold_preds, axis=0)
            base_preds.append(avg_pred)
        
        # Concatenate base model predictions
        meta_features = np.concatenate(base_preds, axis=1)
        
        # Scale features
        meta_features_scaled = self.scaler.transform(meta_features)
        
        proba = self.meta_model.predict_proba(meta_features_scaled)
        return proba
    
    def predict(self, X,threshold=None):
        """
        Predict class labels
        """
        
        proba = self.predict_proba(X)
        if threshold is not None:
            return (proba[:, 1] >= threshold).astype(int)
        return (proba[:, 1] >= 0.5).astype(int)


# Load or create model (placeholder - you'll need to save your trained model)
import zipfile # Add this import at the top of your script

@st.cache_resource
def load_model():
    """
    Load model directly from a zip file and force to CPU.
    Assumes the zip file is named 'model.zip' and contains 'model.pkl'.
    """
    try:
        # Open the zip file in read mode
        with zipfile.ZipFile('model.zip', 'r') as archive:
            # Open the specific pickle file inside the zip
            with archive.open('model.pkl') as model_file:
                
                # Load the model using the file object, mapping to CPU
                model = torch.load(model_file, map_location=torch.device('cpu'))
                
                # --- CRITICAL CPU FIX (From previous step) ---
                if hasattr(model, 'meta_model'):
                    if hasattr(model.meta_model, 'device'):
                        model.meta_model.device = torch.device('cpu')
                    
                    if hasattr(model.meta_model, 'network'):
                        model.meta_model.network = model.meta_model.network.to('cpu')
                    elif isinstance(model.meta_model, nn.Module):
                        model.meta_model = model.meta_model.to('cpu')

                return model

    except FileNotFoundError:
        st.error("⚠️ 'model.zip' not found. Please ensure the file is in the directory.")
        return None
    except KeyError:
        st.error("⚠️ 'model.pkl' not found inside the zip file. Check the internal filename.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

model = load_model()

# Sidebar - Feature Input
st.sidebar.title("🎛️ Adjust Risk Factors")
st.sidebar.markdown("Modify the values below to see how different factors affect heart disease risk.")

# Organize features by category
categories = {}
for feature, config in FEATURE_DEFINITIONS.items():
    category = config.get('category', 'Other')
    if category not in categories:
        categories[category] = []
    categories[category].append(feature)

# Create input dictionary
user_inputs = {}

# Create tabs for different categories
category_order = ['Demographics', 'Clinical', 'Lifestyle - Smoking', 'Lifestyle - Alcohol', 'Lifestyle - Activity']
selected_category = st.sidebar.selectbox("Select Category", category_order)
selected_features =['PHYSHLTH', 'EDUCA', 'RENTHOM1', 'VETERAN3', 'DECIDE', 'SMOKE100', 'SMOKDAY2', 'USENOW3', 'ALCDAY4', 'LASTSMK2',
 '_IMPRACE', '_PHYS14D', '_TOTINDA', '_MRACE1', '_HISPANC', '_RACEG21', '_RACEGR3', '_AGE65YR', '_AGE_G', 'WTKG3',
 '_BMI5CAT', '_RFBMI5', '_SMOKER3', '_RFSMOK3', 'LCSLAST_', 'LCSNUMC_', '_LCSYSMK', '_PACKDAY', '_PACKYRS', '_LCSYQTS',
 '_LCSSMKG', 'DRNKANY6', '_RFBING6', '_ADULT']


#  ['EDUCA', 'RENTHOM1', 'VETERAN3', 'DECIDE', 'SMOKE100', 'SMOKDAY2', 'LASTSMK2', '_IMPRACE', '_PHYS14D', '_TOTINDA', '_HISPANC', '_RACE', 
#                     '_RACEGR3', '_AGE65YR', '_AGE_G', 'WTKG3', '_BMI5CAT', '_RFBMI5', '_INCOMG1', '_SMOKER3', '_RFSMOK3', 'LCSLAST_', '_LCSYSMK', '_PACKYRS',
#                      '_LCSYQTS', '_LCSSMKG', '_RFBING6', '_RFDRHV9', '_ADULT']

st.sidebar.markdown(f"### {selected_category}")
calculated_feature = ['_AGE_G', ]
for feature in list(FEATURE_DEFINITIONS.keys()):
    if feature not in calculated_feature:
        config = FEATURE_DEFINITIONS[feature]
        
        # Initialize session state with default if not exists
        if feature not in st.session_state:
            if config['type'] in ['radio', 'selectbox']:
                st.session_state[feature] = config['options'][config['default']]
            else:
                st.session_state[feature] = config['default']
        
        if config['type'] == 'slider':
            value = st.sidebar.slider(
                config['label'],
                min_value=config['min'],
                max_value=config['max'],
                # REMOVE value=config['default'], USE key ONLY
                key=feature,
                step=config.get('step', 1),
                help=config['help']
            )
        elif config['type'] in ['radio', 'selectbox']:
            options_list = list(config['options'].keys())
            current_value = st.session_state[feature]
            
            # --- ROBUST LOOKUP START ---
            # Try to find the key (label) that matches the current numeric value
            found_keys = [k for k, v in config['options'].items() if v == current_value]
            
            if found_keys:
                default_key = found_keys[0]
            else:
                # If value is invalid/not found, fallback to the default defined in config
                default_key = config['default']
                # Auto-correct the session state so it doesn't happen again
                st.session_state[feature] = config['options'][default_key]
            # --- ROBUST LOOKUP END ---
            
            if config['type'] == 'radio':
                selected = st.sidebar.radio(
                    config['label'],
                    options_list,
                    index=options_list.index(default_key),
                    key=f"{feature}_display",
                    help=config['help']
                )
            else:  # selectbox
                selected = st.sidebar.selectbox(
                    config['label'],
                    options_list,
                    index=options_list.index(default_key),
                    key=f"{feature}_display",
                    help=config['help']
                )
                
            value = config['options'][selected]
            st.session_state[feature] = value
        
        user_inputs[feature] = st.session_state[feature]

def categorize_age(age: int):
    """
    Categorize raw age (_IMPAGE) into defined age groups.

    Parameters:
        age (int): Raw age value

    Returns:
        tuple: (value, label) where
            value = numeric code (1–6)
            label = descriptive string
    """
    if 18 <= age <= 24:
        return 1 #(1, "Age 18 to 24")
    elif 25 <= age <= 34:
        return 2#(2, "Age 25 to 34")
    elif 35 <= age <= 44:
        return 3# (3, "Age 35 to 44")
    elif 45 <= age <= 54:
        return 4#(4, "Age 45 to 54")
    elif 55 <= age <= 64:
        return 5#(5, "Age 55 to 64")
    elif age >= 65:
        return 6#(6, "Age 65 or older")
    else:
        return 0#(0, "Under 18 (not categorized)")

user_inputs['_AGE_G'] = categorize_age(user_inputs['_AGE80'])



# Add preset profiles
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎭 Quick Profiles")

if st.sidebar.button("Low Risk Profile"):
    st.session_state.load_low_risk = True
    st.rerun()

if st.sidebar.button("High Risk Profile"):
    st.session_state.load_high_risk = True
    st.rerun()

if st.sidebar.button("Reset to Default"):
    st.session_state.reset_defaults = True
    st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<p class="sub-header">📊 Current Risk Assessment</p>', unsafe_allow_html=True)
    
    # Create DataFrame for prediction
    input_df = pd.DataFrame([user_inputs])
    # print(input_df[selected_features].columns)
    
    # Make prediction (if model is loaded)
    if model is not None:
        try:
            prediction_proba = model.predict_proba(input_df[selected_features])[0]
            risk_score = prediction_proba[1] * 100  # Probability of positive class
            
            # Risk level determination
            if risk_score < 20:
                risk_level = "Low"
                risk_color = "#4caf50"
                risk_class = "risk-low"
            elif risk_score < 50:
                risk_level = "Moderate"
                risk_color = "#ff9800"
                risk_class = "metric-card"
            else:
                risk_level = "High"
                risk_color = "#f44336"
                risk_class = "risk-high"
            
            # Display risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Heart Disease Risk Score", 'font': {'size': 24}},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': risk_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 20], 'color': '#e8f5e9'},
                        {'range': [20, 50], 'color': '#fff3e0'},
                        {'range': [50, 100], 'color': '#ffebee'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor="white",
                font={'color': "darkblue", 'family': "Arial"},
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk interpretation
            st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
            st.markdown(f"### Risk Level: **{risk_level}** ({risk_score:.1f}%)")
            
            if risk_level == "Low":
                st.markdown("✅ Your current profile suggests a low risk of heart disease. Continue maintaining healthy lifestyle choices!")
            elif risk_level == "Moderate":
                st.markdown("⚠️ Your current profile suggests moderate risk. Consider discussing preventive measures with your healthcare provider.")
            else:
                st.markdown("🚨 Your current profile suggests elevated risk. Please consult with a healthcare professional for a comprehensive evaluation.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure all features are properly configured.")
    else:
        st.warning("⚠️ Model not loaded. Please load a trained model to see predictions.")
        st.info("To use this dashboard:\n1. Train your model\n2. Save it using pickle or joblib\n3. Update the `load_model()` function to load your saved model")

with col2:
    st.markdown('<p class="sub-header">📈 Key Risk Factors</p>', unsafe_allow_html=True)
    
    # Display top risk factors (example - would need actual SHAP or feature importance)
    risk_factors = []
    
    if user_inputs.get('_SMOKER3', 4) in [1, 2]:
        risk_factors.append(("🚬 Current Smoker", "High Impact"))
    if user_inputs.get('DIABETE4', 1) in [2, 3]:
        risk_factors.append(("💉 Diabetes", "High Impact"))
    if user_inputs.get('_AGE80', 45) > 60:
        risk_factors.append(("👴 Age > 60", "High Impact"))
    if user_inputs.get('_BMI5CAT', 2) == 4:
        risk_factors.append(("⚖️ Obesity", "Moderate Impact"))
    if user_inputs.get('_TOTINDA', 1) == 2:
        risk_factors.append(("🏃 No Exercise", "Moderate Impact"))
    if user_inputs.get('PHYSHLTH', 0) > 10:
        risk_factors.append(("😷 Poor Physical Health", "Moderate Impact"))
    
    if risk_factors:
        for factor, impact in risk_factors:
            impact_color = "#f44336" if impact == "High Impact" else "#ff9800"
            st.markdown(f"""
                <div style="background-color: #ff9800; padding: 0.5rem; margin: 0.5rem 0; border-radius: 0.5rem; border-left: 5px solid {impact_color}">
                    <strong>{factor}</strong><br>
                    <small style="color: {impact_color}">{impact}</small>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No major risk factors identified!")

# Feature Importance Section
st.markdown("---")
st.markdown('<p class="sub-header">🎯 Understanding the Model</p>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown("### Top Lifestyle Contributors")
    st.markdown("""
    Based on the research, these lifestyle factors showed the most consistent impact:
    
    1. **Smoking Status** - Current and former smokers show elevated risk
    2. **Physical Activity** - Regular exercise reduces risk significantly
    3. **Alcohol Consumption** - Heavy drinking increases risk
    4. **BMI/Weight** - Obesity is a major risk factor
    5. **Socioeconomic Factors** - Income and education play a role
    """)

with col4:
    st.markdown("### Model Performance")
    st.markdown("""
    **Key Findings:**
    - ✅ Adding lifestyle features improved all models
    - 📊 Improvement was consistent across 5 different algorithms
    - ⚠️ Data quality limited due to 80%+ missing values
    - 🎯 F1 Score improved by 2-5% with lifestyle features
    
    **Models Used:**
    - CatBoost
    - LightGBM
    - XGBoost
    """)

# Download section
st.markdown("---")
st.markdown('<p class="sub-header">💾 Export Results</p>', unsafe_allow_html=True)

if st.button("📥 Download Current Assessment"):
    result_df = pd.DataFrame([user_inputs])
    if model is not None:
        result_df['risk_score'] = risk_score
        result_df['risk_level'] = risk_level
    
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="heart_disease_assessment.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Disclaimer:</strong> This tool is for research and educational purposes only. 
        It should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
        <p>📚 Based on CDC BRFSS 2024 data | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)
