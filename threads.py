import predictDiabetes
import threading


def predict():
    predictionKNN = threading.Thread(
        predictDiabetes.predict_diabetes_KNN,
        (
            150,
            120,
            50,
            25,
            120,
            140,
            70,
        ),
    )
    predictionSVM = threading.Thread(
        predictDiabetes.predict_diabetes_SVM,
        (
            150,
            120,
            50,
            25,
            120,
            140,
            70,
        ),
    )
    predictionNN = threading.Thread(
        predictDiabetes.predict_diabetes_NN,
        (
            150,
            120,
            50,
            25,
            120,
            140,
            70,
        ),
    )
    predictionDT = threading.Thread(
        predictDiabetes.predict_diabetes_DT,
        (
            150,
            120,
            50,
            25,
            120,
            140,
            70,
        ),
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

    print(predictionSVM, predictionNN, predictionDT, predictionKNN)

predict()
