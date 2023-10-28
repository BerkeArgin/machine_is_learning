from helpers import *
from implementations import *
from preprocess import *

variables = [
    "GENHLTH", "_RFHLTH",
    "HLTHPLN1", "_HCVU651",
    "BPHIGH4", "_RFHYPE5",
    "BLOODCHO", "CHOLCHK", "_CHOLCHK",
    "BLOODCHO", "TOLDHI2", "_RFCHOL",
    "CVDINFR4", "CVDCRHD4", "_MICHD",
    "ASTHMA3", "_LTASTH1",
    "ASTHMA3", "ASTHNOW", "_CASTHM1",
    "ASTHMA3", "ASTHNOW", "_ASTHMS1",
    "HAVARTH3", "_DRDXAR1",
    "MRACE1", "ORACE3", "MRACASC1", "_PRACE1",
    "MRACE1", "MRACORG1", "MRACASC1", "_MRACE1",
    "HISPANC3", "_HISPANC",
    "MRACE1", "HISPANC3", "MRACORG1", "MRACASC1", "_MRACE1", "_HISPANC", "_RACE",
    "HISPANC3", "MRACE1", "_RACE", "_RACEG21",
    "HISPANC3", "MRACE1", "_RACE", "_RACEGR3",
    "MRACE1", "HISPANC3", "_RACEGR3", "_RACE_G1",
    "AGE", "_AGEG5YR",
    "AGE", "_AGE65YR",
    "AGE", "_IMPAGE", "_AGE80",
    "AGE", "_IMPAGE", "_AGE_G",
    "HEIGHT3", "HTIN4",
    "HEIGHT3", "HTIN4", "HTM4",
    "WEIGHT2", "WTKG3",
    "SEX", "WEIGHT2", "HEIGHT3", "HTIN4", "HTM4", "WTKG3", "_BMI5",
    "SEX", "WEIGHT2", "HEIGHT3", "HTIN4", "HTM4", "WTKG3", "_BMI5", "_BMI5CAT",
    "SEX", "WEIGHT2", "HEIGHT3", "HTIN4", "HTM4", "WTKG3", "_BMI5", "_RFBMI5",
    "CHILDREN", "_CHLDCNT",
    "EDUCA", "_EDUCAG",
    "INCOME2", "_INCOMG",
    "SMOKE100", "SMOKDAY2", "_SMOKER3",
    "SMOKE100", "SMOKDAY2", "_SMOKER3", "_RFSMOK3",
    "ALCDAY5", "DRNKANY5",
    "ALCDAY5", "DROCDY3_",
    "ALCDAY5", "DRNK3GE5", "_RFBING5",
    "ALCDAY5", "AVEDRNK", "DROCDY3_", "_DRNKWEK",
    "SEX", "ALCDAY5", "AVEDRNK", "_DRNKWEK", "_RFDRHV5",
    "FRUITJU1", "FTJUDA1_",
    "FRUIT1", "FRUTDA1_",
    "FVBEANS", "BEANDAY_",
    "FVGREEN", "GRENDAY_",
    "FVORANG", "ORNGDAY_",
    "VEGETAB1", "VEGEDA1_",
    "FRUITJU1", "FRUIT1", "FTJUDA1_", "FRUTDA1_", "_MISFRTN",
    "VEGETAB1", "FVGREEN", "FVORANG", "FVBEANS", "GRENDAY_", "ORNGDAY_", "BEANDAY_", "VEGEDA1_", "_MISVEGN",
    "FRUITJU1", "FRUIT1", "FTJUDA1_", "FRUTDA1_", "_MISFRTN", "_FRTRESP",
    "VEGETAB1", "FVGREEN", "FVORANG", "FVBEANS", "GRENDAY_", "ORNGDAY_", "BEANDAY_", "VEGEDA1_", "_MISVEGN", "_VEGRESP",
    "FRUITJU1", "FRUIT1", "FTJUDA1_", "FRUTDA1_", "_FRUTSUM",
    "VEGETAB1", "FVGREEN", "FVORANG", "FVBEANS", "GRENDAY_", "ORNGDAY_", "BEANDAY_", "VEGEDA1_", "_VEGESUM",
    "FRUITJU1", "FRUIT1", "VEGETAB1", "FVGREEN", "FVORANG", "FVBEANS", "FTJUDA1_", "FRUTDA1_", "VEGEDA1_", "GRENDAY_", "ORNGDAY_", "BEANDAY_", "_FRUTVEG",
    "XPA1MIN_", "PA1MIN_",
    "XPA1MIN_", "_PACAT1",
    "PA1MIN_", "_METVL11",
    "_METVL11", "_METVL21",
    "PAQ650", "_RFPAVIG",
    "PAQ655", "PAQ660", "_PAQ6C",
    "PAQ660", "_PAINDX2",
    "PAQ650", "PAQ665", "_PASTRNG",
    "PAQ670", "_PAREC1"
]

variables_set= set(variables)

data_path="C://Users//Nazlican//Desktop//ml//Turco-ML//project1//data//dataset_to_release" 

x_train, x_test, y_train, train_ids, test_ids, col_names_train, col_names_test,final_columns = load_csv_data(data_path, selected_cols=variables_set)

#Shifting -1 labels to 0
y_train[y_train==-1]=0

x_train,x_test,numerical_columns_indices=apply_preprocessing(x_train,x_test)

x_train,mean,std=standardize(x_train)
x_test,_,_=standardize(x_test,mean,std)

minority_count=y_train[y_train==1].shape[0]

undersampled_majority_indices = np.random.choice(np.where(y_train == 0)[0], minority_count, replace=False)
undersampled_indices = np.concatenate([undersampled_majority_indices, np.where(y_train == 1)[0]])

X_undersampled = x_train[undersampled_indices]
y_undersampled = y_train[undersampled_indices]


initial_w=np.zeros((x_train.shape[1],))

w,loss=reg_ridge_regression(y_undersampled,X_undersampled,
                                                    lambda_=0.001,
                                                    initial_w=initial_w, 
                                                    max_iters=1000, 
                                                    gamma=0.06,
                                                    )

y_pred,scores = ridge_predict(x_test,w,threshold=0.5)
y_pred[y_pred==0]=-1
create_csv_submission(test_ids,y_pred,"submission.csv")