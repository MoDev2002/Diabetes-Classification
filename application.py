from flask import Flask, request, render_template
import predictDiabetes
import threading

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # getting patient info from Form
    cholesterol_value = request.form["cholesterol"]
    glucose_value = request.form["glucose"]
    hdl_chol_value = request.form["hdl_chol"]
    age_value = request.form["age"]
    weight_value = request.form["weight"]
    systolic_bp_value = request.form["systolic_bp"]
    diastolic_bp_value = request.form["diastolic_bp"]

    def run_knn():
        global knn_prediction
        result = predictDiabetes.predict_diabetes_KNN(
            cholesterol_value,
            glucose_value,
            hdl_chol_value,
            age_value,
            weight_value,
            systolic_bp_value,
            diastolic_bp_value,
        )
        knn_prediction = result

    def run_svm():
        global svm_prediction
        result = predictDiabetes.predict_diabetes_SVM(
            cholesterol_value,
            glucose_value,
            hdl_chol_value,
            age_value,
            weight_value,
            systolic_bp_value,
            diastolic_bp_value,
        )
        svm_prediction = result

    def run_nn():
        global nn_prediction
        result = predictDiabetes.predict_diabetes_NN(
            cholesterol_value,
            glucose_value,
            hdl_chol_value,
            age_value,
            weight_value,
            systolic_bp_value,
            diastolic_bp_value,
        )
        nn_prediction = result

    def run_dt():
        global dt_prediction
        result = predictDiabetes.predict_diabetes_DT(
            cholesterol_value,
            glucose_value,
            hdl_chol_value,
            age_value,
            weight_value,
            systolic_bp_value,
            diastolic_bp_value,
        )
        dt_prediction = result

    # Assign each a thread to each model
    thread_knn = threading.Thread(target=run_knn)
    thread_svm = threading.Thread(target=run_svm)
    thread_nn = threading.Thread(target=run_nn)
    thread_dt = threading.Thread(target=run_dt)

    # start the threads concerruntly
    thread_knn.start()
    thread_svm.start()
    thread_nn.start()
    thread_dt.start()

    # terminate the threads when they're done
    thread_knn.join()
    thread_svm.join()
    thread_nn.join()
    thread_dt.join()

    return render_template(
        "result.html",
        predictions={
            "knn_prediction": knn_prediction,
            "svm_prediction": svm_prediction,
            "nn_prediction": nn_prediction,
            "dt_prediction": dt_prediction,
        },
    )


if __name__ == "__main__":
    app.run(debug=True)
