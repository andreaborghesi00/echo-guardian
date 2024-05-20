import warnings
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import threading
import radiomics
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
import SimpleITK as sitk

def process_fold(model, train_index, test_index, features, labels, results = None, fold_index = None, lock = None):
    
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index].ravel(), labels[test_index].ravel()
    model.fit(X = X_train, y = y_train)
    outputs = model.predict(X_test)
    accuracy = np.sum(outputs == y_test) / len(y_test)
    if lock is not None:
        with lock:
            results[fold_index] = accuracy
            return
    else:
        return accuracy

# Kfold cross validation, no use of functions of sklearn, everything is done manually in order to have more control over the process and to be able to explain it better, this helped a lot in understanding the process
def benchmark(model, train_features, train_labels, folds=10, multiprocessing = False):
    
    results = [None] * folds 
    length_fold = len(train_features) // folds
    indexes = np.arange(0, len(train_features))
    np.random.shuffle(indexes)
    
    # implement multiprocessing with parallel, it's the same as multithreading in the else case, but harder to understand because everything had to be put in a function in just one line
    if multiprocessing:
        
        parallel = Parallel(n_jobs=-1)
        # Disrtibute the folds among the cores
        results = [r for r in parallel(delayed(process_fold)(model, 
                                    indexes[:fold_index * length_fold].tolist() + indexes[fold_index * length_fold + length_fold:].tolist(),
                                    indexes[fold_index * length_fold:fold_index * length_fold + length_fold],
                                    train_features, train_labels) for fold_index in range(folds))]
    # else multithreading, easier to understand by other people, here we use lock to avoid race conditions when accessing the results list which is a shared resource
    else:
        lock = threading.Lock()
        threads = []
        
        for fold_index in range(folds):
            start = fold_index * length_fold 
            end = start + length_fold if fold_index != folds - 1 else len(indexes)
            test_indexes = indexes[start:end] 
            train_indexes = indexes[:start].tolist() + indexes[end:].tolist() 
            
            # Spawn a new thread for each fold
            thread = threading.Thread(target=process_fold, args=(model, train_indexes, test_indexes, train_features, train_labels, results, fold_index, lock))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
    
    # Sum the results after all processes have finished
    acc = np.sum(results) / len(results)
    std = np.std(results)
    return acc, std


# training on different kernels
def grid_search(train_features, test_features, train_labels, test_labels, params, folds = 10):
    
    results = {'linear': {'accuracy_kfold' : 0}, 'rbf': {'accuracy_kfold' : 0}, 'sigmoid': {'accuracy_kfold' : 0}, 'poly': {'accuracy_kfold' : 0}, 'random_forest': {'accuracy_kfold' : 0}}
    best_accuracy = 0
    gs_params = {}
    models = {}
    
    # Testing every kernel
    for kernel in params['kernel']:
        print(f'\n\n------------------\nTesting {kernel}...\n')
        
        if kernel == 'random_forest':
            for criterion in params['criterion']:
                for i in range(params['n_estimators'][0], params['n_estimators'][1], params['n_estimators'][2]):
                    model = RandomForestClassifier(criterion=criterion, n_estimators=i)
                    accuracy_kfold, std_kfold = benchmark(model,
                                                    train_features,
                                                    train_labels,
                                                    folds = folds,
                                                    multiprocessing = 1)
                    if accuracy_kfold > results[kernel]['accuracy_kfold']:
                        results[kernel] = ({'accuracy_kfold': accuracy_kfold, 'std': std_kfold, 'criterion': criterion, 'n_estimators': i})
            model = RandomForestClassifier(criterion = results[kernel]['criterion'], n_estimators = results[kernel]['n_estimators'], n_jobs=-1)
            gs_params = {'n_estimators': [results[kernel]['n_estimators']], 'criterion': [results[kernel]['criterion']]}
        
        else:
            for c in range(1,2) if kernel == 'linear' else np.arange(params['C'][0], params['C'][1], params['step']):
                for degree in range(1) if kernel!='poly' else np.arange(params['degree'][0], params['degree'][1], 1):
                    model = svm.SVC(kernel=kernel, C=c, degree=degree, gamma='auto')
                    accuracy_kfold, std_kfold = benchmark(model,
                                                    train_features,
                                                    train_labels,
                                                    folds = folds)
                    
                    if accuracy_kfold >= results[kernel]['accuracy_kfold']:
                        results[kernel] = ({'C': c, 'degree': degree, 'accuracy_kfold': accuracy_kfold, 'std': std_kfold})
                    
            model = svm.SVC(kernel=kernel,
                            C=results[kernel]['C'],
                            degree=results[kernel].get('degree', 0))
            gs_params = {'kernel' : [kernel] , 'C': [results[kernel]['C']], 'degree': [results[kernel].get('degree', 0)]}
        
        # Training phase, save every candidate model since we need them for the ensemble, but later check if the candidate is the best of the bests
        candidate = model.fit(train_features, train_labels)
        outputs_candidate = candidate.predict(test_features)
        accuracy = np.sum(outputs_candidate == test_labels) / len(test_labels)
        results[kernel].update({'accuracy' : accuracy})
        models[kernel] = candidate
        
        # Check if candidate is the best of the bests, and save it
        if accuracy > best_accuracy:
            best_model = candidate
            best_accuracy = accuracy
            print(f'New best model: {best_accuracy}')
        
        # See how the model grid search performs compared to our search        
        clf = GridSearchCV(model, n_jobs=-1, param_grid={**gs_params})
        clf.fit(train_features, train_labels)
        outputs_GS_model = clf.predict(test_features)
        GS_accuracy = np.sum(outputs_GS_model == test_labels) / len(test_labels)
        
                
        
        # Testing phase: print results and comparison with grid search
        print (f'\n\n Results on {kernel}:\n')
        print (f'''\
                Grid search: {GS_accuracy}
                Our search:  {results[kernel]["accuracy"]}
                
                Scores on kfold:
                Our search: {results[kernel]["accuracy_kfold"]}
                Grid search: {benchmark(model = clf.best_estimator_, train_features = train_features, train_labels = train_labels, folds = folds, multiprocessing=True)}
                
                --------------------------------------
                Detailed Report:
                
                Grid search:
                {classification_report(test_labels, outputs_GS_model)}
                
                Our search:
                {classification_report(test_labels, outputs_candidate)}
                
                Parameters:
                Grid search:
                {clf.best_params_}
                Our search:
                {candidate.get_params()}
                --------------------------------------
                All results:
                {results}''')
    
    return best_model, models

# @see https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/b75721a121102cf972f942fad927751089a7cc80/Python/03_Image_Details.ipynb

def srgb2gray(image):
    # Convert sRGB image to gray scale and rescale results to [0,255]    
    channels = [sitk.VectorIndexSelectionCast(image,i, sitk.sitkFloat32) for i in range(image.GetNumberOfComponentsPerPixel())]
    #linear mapping
    I = 1/255.0*(0.2126*channels[0] + 0.7152*channels[1] + 0.0722*channels[2])
    #nonlinear gamma correction
    I = I*sitk.Cast(I<=0.0031308,sitk.sitkFloat32)*12.92 + I**(1/2.4)*sitk.Cast(I>0.0031308,sitk.sitkFloat32)*1.055-0.055
    return sitk.Cast(sitk.RescaleIntensity(I), sitk.sitkUInt8)


def radiomics_features(image, mask):
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
    sitk_image = sitk.GetImageFromArray(image)
    sitk_mask = sitk.GetImageFromArray(mask)
    features = extractor.execute(sitk_image, sitk_mask, voxelBased=False, label=1)
    features_values = [float(features[key]) for key in features if key.startswith('original_')]
    return features_values