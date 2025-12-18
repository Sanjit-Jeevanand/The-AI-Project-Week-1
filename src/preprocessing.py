from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(categorical_cols, numerical_cols):
    scaler = StandardScaler()
    ohe = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers= [
            ("numeric", scaler, numerical_cols),
            ("categorical", ohe, categorical_cols)
        ]
    )
    return preprocessor