import os
from flask import Flask, render_template

from models.load import load_data
from models import eda as eda_module
from models import feature_engg as fe_module
from models import linear_regression as lr_module
from models import logistic_regression as logit_module
from models.predict import predict_student, get_feature_info
from flask import Flask, render_template, request

from models.predict import (
    predict_student,
    get_feature_info
)
app = Flask(__name__)
app.secret_key = "dev-secret-key"

DATA_PATH = "placement_predict_50k Dataset1.csv"
TARGET_COLS = ["PlacementStatus", "IsAnomaly", "Salary Package"]

# Very small in-memory cache so we don't reload/recompute on every click.
# Fine for a local single-user demo; swap for a proper session/store for production use.
_cache = {}


def get_df():
    if "df" not in _cache:
        _cache["df"] = load_data(DATA_PATH)
    return _cache["df"]


@app.route("/")
def index():
    data_available = os.path.exists(DATA_PATH)
    return render_template("index.html", data_available=data_available, data_path=DATA_PATH)
@app.route("/predict", methods=["GET", "POST"])
def predict():

    feature_info = get_feature_info()

    result = None
    error = None
    values = {}

    if request.method == "POST":
        values = request.form.to_dict()

        try:
            result = predict_student(values)

        except (ValueError, TypeError) as e:
            error = f"Invalid input: {e}"

    return render_template(
        "predict.html",
        numeric_cols=feature_info["numeric_cols"],
        categorical_cols=feature_info["categorical_cols"],
        categories=feature_info["categories"],
        values=values,
        result=result,
        error=error
    )

@app.route("/load")
def load_page():
    try:
        df = get_df()
        info = {
            "shape": df.shape,
            "head": df.head().to_html(classes="table"),
            "dtypes": df.dtypes.to_frame("dtype").to_html(classes="table"),
        }
        error = None
    except Exception as e:
        info = None
        error = str(e)
    return render_template("load.html", info=info, error=error)


@app.route("/eda")
def eda_page():
    try:
        df = get_df()
        summary = eda_module.summarize(df)
        stats = eda_module.stat(df)
        images = eda_module.visualize(df)
        error = None
    except Exception as e:
        summary = stats = images = None
        error = str(e)
    return render_template("eda.html", summary=summary, stats=stats, images=images, error=error)


@app.route("/feature-engg")
def feature_engg_page():
    try:
        df = get_df()
        normalized_df, standardized_df = fe_module.normalize(df, exclude_cols=TARGET_COLS)
        standardized_df = fe_module.outlier(standardized_df)
        _cache["standardized_df"] = standardized_df  # feed downstream model pages

        result = {
            "normalized_head": normalized_df.head().to_html(classes="table"),
            "standardized_head": standardized_df.head().to_html(classes="table"),
        }
        error = None
    except Exception as e:
        result = None
        error = str(e)
    return render_template("feature_engg.html", result=result, error=error)


@app.route("/linear-regression")
def linear_regression_page():
    try:
        df = _cache.get("standardized_df", get_df())
        gd_result = lr_module.simple_linear_regression_GD(df)
        sk_result = lr_module.simple_linear_regression_sklearn(df)
        error = None
    except Exception as e:
        gd_result = sk_result = None
        error = str(e)
    return render_template(
        "linear_regression.html", gd_result=gd_result, sk_result=sk_result, error=error
    )


@app.route("/logistic-regression")
def logistic_regression_page():
    try:
        df = _cache.get("standardized_df", get_df())
        result = logit_module.simple_logistic_regression(df)
        error = None
    except Exception as e:
        result = None
        error = str(e)
    return render_template("logistic_regression.html", result=result, error=error)


if __name__ == "__main__":
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False
        )
