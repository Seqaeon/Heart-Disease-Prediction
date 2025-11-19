import os
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



# Feature definitions with proper ranges and descriptions
FEATURE_DEFINITIONS = {
    # Demographics
    '_AGE80': {
        'label': 'Age',
        'type': 'slider',
        'min': 18, 'max': 80, 'default': 45,
        'help': 'Age in years (18-80)',
        'category': 'Demographics'
    },
    '_SEX': {
        'label': 'Sex',
        'type': 'radio',
        'options': {'Male': 1, 'Female': 2},
        'default': 'Male',
        'help': 'Biological sex',
        'category': 'Demographics'
    },
    '_IMPRACE': {
        'label': 'Race/Ethnicity',
        'type': 'selectbox',
        'options': {
            'White': 1, 'Black': 2, 'Asian': 3, 
            'Native American': 4, 'Hispanic': 5, 'Other': 6
        },
        'default': 'White',
        'help': 'Race/ethnicity classification',
        'category': 'Demographics'
    },
    'EDUCA': {
        'label': 'Education Level',
        'type': 'selectbox',
        'options': {
            'Never attended school': 1,
            'Elementary': 2,
            'Some high school': 3,
            'High school graduate': 4,
            'Some college': 5,
            'College graduate': 6,
        },
        'default': 'High school graduate',
        'help': 'Highest level of education completed',
        'category': 'Demographics'
    },
    'MARITAL': {
        'label': 'Marital Status',
        'type': 'selectbox',
        'options': {
            'Married': 1, 'Divorced': 2, 'Widowed': 3,
            'Separated': 4, 'Never married': 5, 'Unmarried couple': 6,  

        },
        'default': 'Married',
        'help': 'Current marital status',
        'category': 'Demographics'
    },
    'VETERAN3': {
        'label': 'Veteran Status',
        'type': 'radio',
        'options': {'Yes': 1, 'No': 2},
        'default': 'No',
        'help': 'Have you ever served in the military?',
        'category': 'Demographics'
    },
    'EMPLOY1': {
        'label': 'Employment Status',
        'type': 'selectbox',
        'options': {
            'Employed': 1, 'Self-employed': 2, 'Unemployed <1 year': 3,
            'Unemployed >1 year': 4, 'Homemaker': 5, 'Student': 6,
            'Retired': 7, 'Unable to work': 8
        },
        'default': 'Employed',
        'help': 'Current employment status',
        'category': 'Demographics'
    },
    'RENTHOM1': {
        'label': 'Housing Status',
        'type': 'radio',
        'options': {'Own': 1, 'Rent': 2, 'Other': 3},
        'default': 'Own',
        'help': 'Do you own or rent your home?',
        'category': 'Demographics'
    },
    '_INCOMG1': {
        'label': 'Income Level',
        'type': 'selectbox',
        'options': {
            '<$15,000': 1, '$15,000-$25,000': 2, '$25,000-$35,000': 3,
            '$35,000-$50,000': 4, '$50,000-$75,000': 5, '>$75,000': 6
        },
        'default': '$35,000-$50,000',
        'help': 'Annual household income range',
        'category': 'Demographics'
    },
    
    # Clinical Features
    'DIABETE4': {
        'label': 'Diabetes Status',
        'type': 'selectbox',
        'options': {
            'No': 1, 'Yes': 2, 'Borderline/Pre-diabetes': 3, 'Yes (during pregnancy)': 4
        },
        'default': 'No',
        'help': 'Have you ever been told you have diabetes?',
        'category': 'Clinical'
    },
    'PHYSHLTH': {
        'label': 'Days Physical Health Not Good',
        'type': 'slider',
        'min': 0, 'max': 30, 'default': 0,
        'help': 'Number of days in past 30 days your physical health was not good',
        'category': 'Clinical'
    },
    '_DRDXAR2': {
        'label': 'Arthritis Diagnosis',
        'type': 'radio',
        'options': {'No': 1, 'Yes': 2},
        'default': 'No',
        'help': 'Have you been diagnosed with arthritis?',
        'category': 'Clinical'
    },
    'WTKG3': {
        'label': 'Weight (kg)',
        'type': 'slider',
        'min': 40, 'max': 200, 'default': 75,
        'help': 'Body weight in kilograms',
        'category': 'Clinical'
    },
    '_BMI5CAT': {
        'label': 'BMI Category',
        'type': 'selectbox',
        'options': {
            'Underweight': 1, 'Normal weight': 2,
            'Overweight': 3, 'Obese': 4
        },
        'default': 'Normal weight',
        'help': 'Body Mass Index category',
        'category': 'Clinical'
    },
    '_RFBMI5': {
        'label': 'Overweight/Obese',
        'type': 'radio',
        'options': {'No': 1, 'Yes': 2},
        'default': 'No',
        'help': 'BMI >= 25',
        'category': 'Clinical'
    },
    'DECIDE':{
        'label': 'Difficulty Concentrating or Remembering',
        'type': 'radio',
        'options': {
            'Yes': 1,
            'No': 2,},
        'default': 'No',
        'help': 'Because of a physical, mental, or emotional condition, do you have serious difficulty concentrating, remembering, or making decisions?',
        'category': 'Clinical'


    },  
    
    # Lifestyle - Smoking
    'SMOKE100': {
        'label': 'Smoked 100+ Cigarettes Lifetime',
        'type': 'radio',
        'options': {'No': 2, 'Yes': 1},
        'default': 'No',
        'help': 'Have you smoked at least 100 cigarettes in your lifetime?',
        'category': 'Lifestyle - Smoking'
    },
    '_SMOKER3': {
        'label': 'Smoking Status',
        'type': 'selectbox',
        'options': {
            'Current smoker - daily': 1,
            'Current smoker - some days': 2,
            'Former smoker': 3,
            'Never smoked': 4
        },
        'default': 'Never smoked',
        'help': 'Current smoking status',
        'category': 'Lifestyle - Smoking'
    },
    'SMOKDAY2': {
        'label': 'Days Smoked Per Month',
        'type': 'slider',
        'min': 0, 'max': 30, 'default': 0,
        'help': 'Number of days smoked in past 30 days',
        'category': 'Lifestyle - Smoking'
    },
    'USENOW3': {
        'label': 'Current Tobacco Use',
        'type': 'selectbox',
        'options': {'Every day': 1, 'Some days': 2, 'Not at all': 3},
        'default': 'Not at all',
        'help': 'Do you currently use tobacco?',
        'category': 'Lifestyle - Smoking'
    },
    'LASTSMK2': {
        'label': 'Time Since Last Smoked',
        'type': 'selectbox',
        'options': {
            'Within past month': 1, '1-3 months': 2, '3-6 months': 3,
            '6-12 months': 4, '1-5 years': 5, '5-10 years': 6, '>10 years': 7, 'Never': 8
        },
        'default': 'Never',
        'help': 'How long since you last smoked?',
        'category': 'Lifestyle - Smoking'
    },
    # '_PACKDAY': {
    #     'label': 'Packs Per Day',
    #     'type': 'slider',
    #     'min': 0, 'max': 5, 'default': 0, 'step': 0.1,
    #     'help': 'Average packs of cigarettes per day',
    #     'category': 'Lifestyle - Smoking'
    # },
    '_PACKYRS': {
        'label': 'Pack-Years',
        'type': 'slider',
        'min': 0, 'max': 1000, 'default': 0,
        'help': 'Packs per day × years smoked',
        'category': 'Lifestyle - Smoking'
    },
    
    # Lifestyle - Alcohol
    'DRNKANY6': {
        'label': 'Alcohol Consumption',
        'type': 'radio',
        'options': {'No': 2, 'Yes': 1},
        'default': 'No',
        'help': 'Had any alcohol in past 30 days?',
        'category': 'Lifestyle - Alcohol'
    },
    'ALCDAY4': {
        'label': 'Days Drank Per Month',
        'type': 'slider',
        'min': 0, 'max': 30, 'default': 0,
        'help': 'Number of days had alcohol in past 30 days',
        'category': 'Lifestyle - Alcohol'
    },
    '_DRNKWK3': {
        'label': 'Drinks Per Week',
        'type': 'slider',
        'min': 0, 'max': 50, 'default': 0,
        'help': 'Average number of drinks per week',
        'category': 'Lifestyle - Alcohol'
    },
    '_RFDRHV9': {
        'label': 'Heavy Drinker',
        'type': 'radio',
        'options': {'No': 1, 'Yes': 2},
        'default': 'No',
        'help': 'Heavy drinking (>14 drinks/week men, >7 women)',
        'category': 'Lifestyle - Alcohol'
    },
    
    # Lifestyle - Physical Activity
    '_TOTINDA': {
        'label': 'Physical Activity',
        'type': 'radio',
        'options': {'No': 2, 'Yes': 1},
        'default': 'Yes',
        'help': 'Engaged in physical activity in past 30 days?',
        'category': 'Lifestyle - Activity'
    },
    '_PHYS14D': {
        'label': 'Days Physically Active',
        'type': 'slider',
        'min': 0, 'max': 30, 'default': 15,
        'help': 'Days physically active in past 30 days',
        'category': 'Lifestyle - Activity'
    },
    
    # Additional flags
    '_AGE65YR': {
        'label': 'Age 65+',
        'type': 'radio',
        'options': {'No': 1, 'Yes': 2},
        'default': 'No',
        'help': 'Are you 65 years or older?',
        'category': 'Demographics'
    },
    
    '_ADULT': {
        'label': 'Adult Respondent',
        'type': 'radio',
        'options': {'No': 0, 'Yes': 1},
        'default': 'Yes',
        'help': 'Adult (18+) respondent flag',
        'category': 'Demographics'
    },

    '_RFSMOK3': {
        'label': 'Current Smoker Flag',
        'type': 'radio',
        'options': {'No': 1, 'Yes': 2},
        'default': 'No',
        'help': 'Currently smoking cigarettes',
        'category': 'Lifestyle - Smoking'
    },
    'LCSLAST_': {
        'label': 'Lung Cancer Screening',
        'type': 'slider',
        'min': 0, 'max': 10, 'default': 0,
        'help': 'Years since last lung cancer screening',
        'category': 'Clinical'
    },
    'LCSNUMC_': {
        'label': 'Number of Screenings',
        'type': 'slider',
        'min': 0, 'max': 20, 'default': 0,
        'help': 'Number of lung cancer screenings',
        'category': 'Clinical'
    },
    '_LCSYSMK': {
        'label': 'Years Smoked',
        'type': 'slider',
        'min': 0, 'max': 60, 'default': 0,
        'help': 'Total years as a smoker',
        'category': 'Lifestyle - Smoking'
    },
    '_LCSSMKG': {
        'label': 'Smoking Status for Screening',
        'type': 'selectbox',
        'options': {
            'Never': 1, 'Former (quit >15 years)': 2,
            'Former (quit <15 years)': 3, 'Current': 4
        },
        'default': 'Never',
        'help': 'Smoking status for lung cancer screening eligibility',
        'category': 'Lifestyle - Smoking'
    },
    '_MRACE1': {
        'label': 'Multiracial',
        'type': 'radio',
        'options': {'No': 1, 'Yes': 2},
        'default': 'No',
        'help': 'Identify as multiracial?',
        'category': 'Demographics'
    },
    '_HISPANC': {
        'label': 'Hispanic Origin',
        'type': 'radio',
        'options': {'No': 1, 'Yes': 2},
        'default': 'No',
        'help': 'Hispanic or Latino origin?',
        'category': 'Demographics'
    },
    '_RACE': {
        'label': 'Race (Detailed)',
        'type': 'selectbox',
        'options': {
            'White': 1, 'Black': 2, 'American Indian/Alaska Native': 3,
            'Asian': 4, 'Native Hawaiian/Pacific Islander': 5,
            'Other': 6, 'Multiracial': 7, 'Hispanic': 8
        },
        'default': 'White',
        'help': 'Detailed race classification',
        'category': 'Demographics'
    },
    '_RACEGR3': {
        'label': 'Race Group',
        'type': 'selectbox',
        'options': {
            'White': 1, 'Black': 2, 'Other': 3,
            'Multiracial': 4, 'Hispanic': 5
        },
        'default': 'White',
        'help': 'Race group classification',
        'category': 'Demographics'
    },
    '_LCSYQTS': {
        'label': 'Quit Attempts',
        'type': 'slider',
        'min': 0, 'max': 20, 'default': 0,
        'help': 'Number of times tried to quit smoking',
        'category': 'Lifestyle - Smoking'
    },
    '_RFBING6': {
        'label': 'Binge Drinker',
        'type': 'radio',
        'options': {'No': 1, 'Yes': 2},
        'default': 'No',
        'help': 'Binge drinking (5+ drinks men, 4+ women on one occasion)',
        'category': 'Lifestyle - Alcohol'
    },
}

# Initialize session state for storing predictions
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []



# ADD THIS SECTION HERE - Before any widgets are created
# Define preset profiles
LOW_RISK_PROFILE = {
    '_AGE80': 35, '_SEX': 2, 'DIABETE4': 1, 'PHYSHLTH': 0,
    '_DRDXAR2': 1, 'WTKG3': 70, '_BMI5CAT': 2, '_RFBMI5': 1,
    'SMOKE100': 2, '_SMOKER3': 4, 'SMOKDAY2': 0, 'USENOW3': 3,
    '_PACKYRS': 0, '_TOTINDA': 1, '_PHYS14D': 20,
    'DRNKANY6': 2, 'ALCDAY4': 0, '_DRNKWK3': 0, '_RFDRHV9': 1,
    'EDUCA': 6, '_INCOMG1': 6, 'EMPLOY1': 1, 'MARITAL': 1,
    'VETERAN3': 2, 'RENTHOM1': 1, '_ADULT': 1, '_AGE65YR': 1,
    'LASTSMK2': 8, '_RFSMOK3': 1, 'LCSLAST_': 0, 'LCSNUMC_': 0,
    '_LCSYSMK': 0, '_LCSSMKG': 1, '_MRACE1': 1, '_HISPANC': 1,
    '_RACE': 1, '_RACEGR3': 1, '_LCSYQTS': 0, '_RFBING6': 1,
    '_IMPRACE': 1, 'DECIDE': 2
}

HIGH_RISK_PROFILE = {
    '_AGE80': 70, '_SEX': 1, 'DIABETE4': 2, 'PHYSHLTH': 20,
    '_DRDXAR2': 2, 'WTKG3': 110, '_BMI5CAT': 4, '_RFBMI5': 2,
    'SMOKE100': 1, '_SMOKER3': 1, 'SMOKDAY2': 30, 'USENOW3': 1,
    '_PACKYRS': 50, '_TOTINDA': 2, '_PHYS14D': 0,
    'DRNKANY6': 1, 'ALCDAY4': 20, '_DRNKWK3': 30, '_RFDRHV9': 2,
    'EDUCA': 3, '_INCOMG1': 2, 'EMPLOY1': 7, 'MARITAL': 3,
    'VETERAN3': 1, 'RENTHOM1': 2, '_ADULT': 1, '_AGE65YR': 2,
    'LASTSMK2': 1, '_RFSMOK3': 2, 'LCSLAST_': 5, 'LCSNUMC_': 3,
    '_LCSYSMK': 40, '_LCSSMKG': 4, '_MRACE1': 1, '_HISPANC': 1,
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


@st.cache_resource
def load_model():
    # 1. Get the directory where THIS file (app.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Construct the absolute path to the model
    # Change 'model.pkl' to 'models/model.pkl' if it is in a subfolder
    model_path = os.path.join(current_dir, 'model.pkl')
    
    # 3. Debugging: Print the path to the Streamlit logs so you can see it
    print(f"Attempting to load model from: {model_path}")

    if not os.path.exists(model_path):
        st.error(f"File not found at: {model_path}")
        st.warning("Please ensure 'model.pkl' is committed to your GitHub repository.")
        return None

    try:
        with open(model_path, 'rb') as f:
            model = torch.load(f,map_location=torch.device('cpu'))
        return model
    except Exception as e:
        st.error(f"Error loading pickle file: {e}")
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
selected_features = ['EDUCA', 'RENTHOM1', 'VETERAN3', 'DECIDE', 'SMOKE100', 'SMOKDAY2', 'LASTSMK2', '_IMPRACE', '_PHYS14D', '_TOTINDA', '_HISPANC', '_RACE', 
                    '_RACEGR3', '_AGE65YR', '_AGE_G', 'WTKG3', '_BMI5CAT', '_RFBMI5', '_INCOMG1', '_SMOKER3', '_RFSMOK3', 'LCSLAST_', '_LCSYSMK', '_PACKYRS',
                     '_LCSYQTS', '_LCSSMKG', '_RFBING6', '_RFDRHV9', '_ADULT']

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
        elif config['type'] == 'radio':
            options_list = list(config['options'].keys())
            # Find current index from session state value
            current_value = st.session_state[feature]
            default_key = [k for k, v in config['options'].items() if v == current_value][0]
            
            selected = st.sidebar.radio(
                config['label'],
                options_list,
                index=options_list.index(default_key),
                key=f"{feature}_display",  # Different key for display
                help=config['help']
            )
            value = config['options'][selected]
            st.session_state[feature] = value
            
        elif config['type'] == 'selectbox':
            options_list = list(config['options'].keys())
            # Find current index from session state value
            current_value = st.session_state[feature]
            default_key = [k for k, v in config['options'].items() if v == current_value][0]
            
            selected = st.sidebar.selectbox(
                config['label'],
                options_list,
                index=options_list.index(default_key),
                key=f"{feature}_display",  # Different key for display
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
