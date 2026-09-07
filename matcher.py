import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def load_biometric_data(data_path='data/', feature_type='pca'):
    train_file = f'train_features_{feature_type}.npy'
    test_file = f'test_features_{feature_type}.npy'
    
    train_features = np.load(os.path.join(data_path, train_file))
    test_features = np.load(os.path.join(data_path, test_file))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    
    return train_features, test_features, y_train, y_test

def match_single_face(test_feature, test_label, training_features, training_labels):
    
    norm_test = np.linalg.norm(test_feature)
    norm_train = np.linalg.norm(training_features, axis=1)
    cos_sim = np.dot(training_features, test_feature) / (norm_train * norm_test + 1e-10)
    
    
    euclid_dist = np.linalg.norm(training_features - test_feature, axis=1)
    
    cos_pred_idx = np.argmax(cos_sim)
    euclid_pred_idx = np.argmin(euclid_dist)
    
    return {
        'cosine': {
            'predicted_id': training_labels[cos_pred_idx],
            'genuine_scores': cos_sim[training_labels == test_label].tolist(),
            'impostor_scores': cos_sim[training_labels != test_label].tolist()
        },
        'euclidean': {
            'predicted_id': training_labels[euclid_pred_idx],
            'genuine_scores': euclid_dist[training_labels == test_label].tolist(),
            'impostor_scores': euclid_dist[training_labels != test_label].tolist()
        }
    }

def match_with_svm(train_features, test_features, y_train, y_test):
    print("\n==================================================")
    print("  Method C (SVM - Support Vector Machine)")
    print("==================================================")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_features)
    X_test_scaled = scaler.transform(test_features)
    
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    y_pred = svm_model.predict(X_test_scaled)
    proba  = svm_model.predict_proba(X_test_scaled)  
    classes = svm_model.classes_

    acc = accuracy_score(y_test, y_pred) * 100
    print(f"  Rank-1 Accuracy     : {acc:.2f} %")

   
    svm_scores, svm_labels = [], []
    for i, true_label in enumerate(y_test):
        class_idx = np.where(classes == true_label)[0][0]
        svm_scores.append(proba[i, class_idx])
        svm_labels.append(1)                          
        for j, cls in enumerate(classes):
            if cls != true_label:
                svm_scores.append(proba[i, j])
                svm_labels.append(0)                 

    print("==================================================\n")
    return y_pred, svm_model, np.array(svm_scores), np.array(svm_labels)

def match_with_optimized_fused_svm(pca_train, pca_test, lbp_train, lbp_test, y_train, y_test):
    print("\n--- Starting SVM Hyperparameter Optimization ---")
    
    X_train = np.hstack((pca_train, lbp_train))
    X_test = np.hstack((pca_test, lbp_test))
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],  
        'gamma': [1, 0.1, 0.01, 0.001, 'scale'], 
        'kernel': ['rbf', 'poly', 'linear'] 
    }
    
    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, 
                               cv=3, n_jobs=-1, scoring='accuracy', verbose=0)
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Optimal Parameters Found: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"Optimized Feature-Level Fusion Accuracy: {acc:.2f}%\n")
    
    return y_pred, best_model

