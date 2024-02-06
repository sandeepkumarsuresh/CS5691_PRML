import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def polyregression(Data,reg_parameter,degree):

    """

        Arguments:
        Data: Dataset (passed as a pandas dataframe)
        reg_parameter: regularization parameter
        Degree: Degree of the polynomial

        output: weights

        return weights

    """

    train_X = Data.iloc[:,0]
    train_Y = Data.iloc[:,1]



    design_matrix = np.asarray([[np.power(train_X[i], j) for j in range(degree + 1)] for i in range(len(train_X))])
    Identity_matrix = np.identity(degree + 1)
    weights = np.linalg.pinv(np.transpose(design_matrix) @ design_matrix + reg_parameter * Identity_matrix) @ (np.transpose(design_matrix)) @ train_Y


    return weights , design_matrix

def kfold_indices(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))

    # print(folds[0])
    return folds

if __name__ == '__main__':
    datapath = 'bayes_variance_data.csv'
    data_pd = pd.read_csv(datapath, sep=',', dtype=float)
    data_pd_to_np = data_pd.to_numpy()

    fold_indices = kfold_indices(data_pd,5)
    error_dict = {}
    total_error_train , total_error_test = [] ,[]
    reg_parameter_value = [1e-15, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e6, 1e9, 1e15]
    for reg_parameter in reg_parameter_value:
        for train_indices, test_indices in fold_indices:
            Train_data = pd.DataFrame(data_pd_to_np[train_indices])   # Converting my data to dataframe because , this is the my required form of input
            Test_data = pd.DataFrame(data_pd_to_np[test_indices])
            
            Train_data_weights , design_matrix_train = polyregression(Train_data,reg_parameter,degree=24)

            Test_data_weights , design_matrix_test = polyregression(Test_data,reg_parameter,degree=24) 

            prediction_train = design_matrix_train @ Train_data_weights
            error_train = (Train_data.iloc[:,1] - prediction_train) ** 2
            total_error_train.append((error_train))



            prediction_test = design_matrix_test @ Train_data_weights
            error_test = (Test_data.iloc[:,1] - prediction_test) ** 2
            total_error_test.append((error_test))

        

        error_dict[reg_parameter] = {'total_error_train': np.sum(total_error_train), 
                                    'total_error_test': np.sum(total_error_test),
                                    'average_training_error':np.mean(total_error_train),
                                    'average_test_error':np.mean(total_error_test)}



    for reg_param, errors in error_dict.items():
        print(f"Reg Parameter: {reg_param},\n"
            f"Total Error Train: {errors['total_error_train']},\n"
            f"Total Error Test: {errors['total_error_test']},\n"
            f"Average Training Error: {errors['average_training_error']},\n"
            f"Average Test Error: {errors['average_test_error']}\n")
        

    reg_parameters = list(error_dict.keys())
    print(reg_parameters)
    average_training_errors = [errors['average_training_error'] for errors in error_dict.values()]
    average_test_errors = [errors['average_test_error'] for errors in error_dict.values()]

    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(reg_parameters), average_training_errors, label='Average Training Error')
    plt.plot(np.log10(reg_parameters), average_test_errors, label='Average Test Error')
    plt.xlabel('log10(λ)')
    plt.ylabel('Error')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig('./plots/Learning_curve.jpg')
    plt.show()


    train_indices, test_indices = fold_indices[0]
    Train_data_new = pd.DataFrame(data_pd_to_np[train_indices])
    Test_data_new = pd.DataFrame(data_pd_to_np[test_indices])



    x_train, y_train = Train_data_new.iloc[:, 0], Train_data_new.iloc[:, 1]
    x_test, y_test = Test_data_new.iloc[:, 0], Test_data_new.iloc[:, 1]

    reg_parameter_value = [1e-15, 0.01, 1e15]

    for reg_parameter in reg_parameter_value:
        Train_data_weights_new, design_matrix_train = polyregression(Train_data_new, reg_parameter, degree=24)
        Test_data_weights_new, design_matrix_test_new = polyregression(Test_data_new, reg_parameter, degree=24) 
        
        y_pred_test = design_matrix_test_new @ Train_data_weights_new

        plt.figure()
        
        plt.scatter(x_train, y_train, label='Training Data', color='blue')
        plt.scatter(x_test, y_test, label='Validation Data', color='red')
        plt.scatter(x_test, y_pred_test, label=f'Predicted Data (Reg Parameter: {reg_parameter})', color='black')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Fitted Curve with Reg Parameter: {reg_parameter}')
        plt.legend()
        plt.savefig(f'plots/Fitted Curve with Reg Parameter:{reg_parameter}.jpg')
        plt.show()


    
