from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def evaluate_classification_report(model, x_test, y_test):
    y_pred = model.predict(x_test)  
    report = classification_report(y_test, y_pred)
    return report

def evaluate_multiple_solvers(y_test, y_pred, solver):
    
    report = classification_report(y_test, y_pred)
    result = f"Resultados para o solver {solver}:\n{report}"
    return result