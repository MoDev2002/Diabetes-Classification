import predictDiabetes
import threading


cholesterol_value = 279
glucose_value = 270
hdl_chol_value = 40
age_value = 60
weight_value = 224 / 2.2
systolic_bp_value = 174
diastolic_bp_value = 90

print('knn : ', predictDiabetes.predict_diabetes_KNN(cholesterol_value, glucose_value, hdl_chol_value, age_value, weight_value, systolic_bp_value, diastolic_bp_value))

def run_knn():
    result = predictDiabetes.predict_diabetes_KNN(cholesterol_value, glucose_value, hdl_chol_value, age_value, weight_value, systolic_bp_value, diastolic_bp_value)
    print("KNN result:", result)

def run_svm():
    result = predictDiabetes.predict_diabetes_SVM(cholesterol_value, glucose_value, hdl_chol_value, age_value, weight_value, systolic_bp_value, diastolic_bp_value)
    print("SVM result:", result)

def run_nn():
    result = predictDiabetes.predict_diabetes_NN(cholesterol_value, glucose_value, hdl_chol_value, age_value, weight_value, systolic_bp_value, diastolic_bp_value)
    print("Neural Network result:", result)

def run_dt():
    result = predictDiabetes.predict_diabetes_DT(cholesterol_value, glucose_value, hdl_chol_value, age_value, weight_value, systolic_bp_value, diastolic_bp_value)
    print("Decision Tree result:", result)

thread_knn = threading.Thread(target=run_knn)
thread_svm = threading.Thread(target=run_svm)
thread_nn = threading.Thread(target=run_nn)
thread_dt = threading.Thread(target=run_dt)


thread_knn.start()
thread_svm.start()
thread_nn.start()
thread_dt.start()

thread_knn.join()
thread_svm.join()
thread_nn.join()
thread_dt.join()

print("All threads have finished.")