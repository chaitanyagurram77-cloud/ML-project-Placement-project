from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


def normalization(df):

    # Numeric feature columns
    num_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "CodingTestScore",
        "Internships"
    ]

    # Convert PlacementStatus to numeric target
    # Placed = 1, Not Placed = 0
    df = df.copy()
    df["PlacementStatus"] = df["PlacementStatus"].map({
        "Placed": 1,
        "Not Placed": 0
    })

    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["PlacementStatus"]
    )

    # Separate features and target
    X_train = train_df[num_cols]
    X_test = test_df[num_cols]

    y_train = train_df["PlacementStatus"]
    y_test = test_df["PlacementStatus"]

    # Normalization
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Prediction
    y_pred = model.predict(X_test_scaled)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)

    # Convert normalized data back to DataFrame
    train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=num_cols,
        index=X_train.index
    )

    test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=num_cols,
        index=X_test.index
    )

    return train_scaled, test_scaled, y_train, y_test, model, scaler