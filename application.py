from flask import Flask, request, render_template
import predictDiabetes
import threading

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    predictionKNN = threading.Thread(
        predictDiabetes.predict_diabetes_KNN,
        (
            request.form["cholesterol"],
            request.form["glucose"],
            request.form["hdl_chol"],
            request.form["age"],
            request.form["weight"],
            request.form["systolic_bp"],
            request.form["diastolic_bp"],
        ),
    )
    predictionSVM = threading.Thread(
        predictDiabetes.predict_diabetes_SVM,
        (
            request.form["cholesterol"],
            request.form["glucose"],
            request.form["hdl_chol"],
            request.form["age"],
            request.form["weight"],
            request.form["systolic_bp"],
            request.form["diastolic_bp"],
        ),
    )
    predictionNN = threading.Thread(
        predictDiabetes.predict_diabetes_NN,
        (
            request.form["cholesterol"],
            request.form["glucose"],
            request.form["hdl_chol"],
            request.form["age"],
            request.form["weight"],
            request.form["systolic_bp"],
            request.form["diastolic_bp"],
        )
    )
    predictionDT = threading.Thread(
        predictDiabetes.predict_diabetes_DT,
        (
            request.form["cholesterol"],
            request.form["glucose"],
            request.form["hdl_chol"],
            request.form["age"],
            request.form["weight"],
            request.form["systolic_bp"],
            request.form["diastolic_bp"],
        )
    )

    # Starting the Threads
    predictionKNN.start()
    predictionSVM.start()
    predictionNN.start()
    predictionDT.start()

    # Stopping the Threads
    predictionKNN.join()
    predictionSVM.join()
    predictionNN.join()
    predictionDT.join()

    return render_template(
        "diabetic.html",
        predictions=(predictionKNN, predictionSVM, predictionNN, predictionDT),
    )


if __name__ == "__main__":
    app.run(debug=True)
