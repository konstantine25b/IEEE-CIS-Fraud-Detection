# IEEE-CIS-Fraud-Detection
# IEEE-CIS Fraud Detection

Dataset Description
In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target isFraud.

The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.

Categorical Features - Transaction
ProductCD
card1 - card6
addr1, addr2
P_emaildomain
R_emaildomain
M1 - M9
Categorical Features - Identity
DeviceType
DeviceInfo
id_12 - id_38
The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).

You can read more about the data from this post by the competition host.

Files
train_{transaction, identity}.csv - the training set
test_{transaction, identity}.csv - the test set (you must predict the isFraud value for these observations)
sample_submission.csv - a sample submission file in the correct format



კაი დავიწყოთ.

ვფიქრობ რომ რეპოზიტორიის სტუქტურა მექნება ასე ჯერ  პრე-პროცესინგ ცალკე ფაილი სადაც დავამუშავებ მთელ დატას.
და მერე სხვადასხვა მოდელისთვის სხვადასხვა ფაილი და 1 ცალი model_inference.ipynb საბოლოოდ.

პირობიდან გამომდინარე გვაქვს რამდენიმე კატეგორიული ცვლადი
Categorical Features - Transaction
ProductCD
card1 - card6
addr1, addr2
P_emaildomain
R_emaildomain
M1 - M9
Categorical Features - Identity
DeviceType
DeviceInfo
id_12 - id_38

ასევე გვაქვს ორი ტრეინინგ სეტი ასევე ჩანს რომ TransactionId-ზე უნდა დავაფრედიქთო ფროდია თუ არა.
ეს არის კლასიფიკაციის ამოცანა ამიტომ დავიწყებ ყველაზე მარტივით logistic regression-ის გამოყენებით მარა სანამ დავიწყებ მანამდე ჯერ კაი დიდი პრე პროცესიგი გვაქ გასაკეთებელი 

მარა ვფიქრობ აქამდე რომ ჯერ დატა დავამუშავო რო მერე ცალ ცალკე გავტესტო ყველაზე ამიტო შემქმენი IEEE-CIS Fraud Detection_PreProcessing- სადაც პრე პროცესინგი იქნება.

# Pre-Processing experiment

ფაილი : ieee-cis-fraud-detection-preprocessing.ipynb

transactions - აქვს  394 columns ხოლო identity-ს 41 columns.

ამათი პირდაპირ დაჯოინებით რაღაც დატას პრობლემები მგონია რო შეიქმენება ამიტომ ჯერ გავარკვიოთ რა აქვთ საერთო და საერთოდ როგორებია.

გამოვიკვლიე
identity
Missing values per column:
id_24            139486
id_25            139101
id_07            139078
id_08            139078
id_21            139074
id_26            139070
id_23            139064
id_27            139064
id_22            139064
id_18             99120
id_03             77909
id_04             77909
id_33             70944
id_09             69307
id_10             69307
id_30             66668
id_32             66647
id_34             66428
id_14             64189
DeviceInfo        25567
id_13             16913
id_16             14893


ახლა  ვნახოთ ვის აკლია 20% ზე მეტი
Columns with more than 20% missing values in Identity Table:
id_24    96.708798
id_25    96.441868
id_07    96.425922
id_08    96.425922
id_21    96.423149
id_26    96.420375
id_23    96.416215
id_27    96.416215
id_22    96.416215
id_18    68.722137
id_03    54.016071
id_04    54.016071
id_33    49.187079
id_09    48.052110
id_10    48.052110
id_30    46.222432
id_32    46.207872
id_34    46.056034
id_14    44.503685
dtype: float64

აღმოჩნდა რომ საკმაოდ ბევრი NA გვაქ ამიტომ კაი დამუშავება მოგვიწევს.

ახლა ტრანზაქციების ცხრილი:
Transaction Table Exploration:
Number of rows: 590540
Shape of the DataFrame: (590540, 394)

Missing values per column:
dist2            552913
D7               551623
D13              528588
D14              528353
აქაც არ გვაქ კაი სიტუაცია 

ახლა  ვნახოთ ვის აკლია 20% ზე მეტი
Columns with more than 20% missing values in Transaction Table:
dist2    93.628374
D7       93.409930
D13      89.509263
D14      89.469469
D12      89.041047
           ...    
V41      28.612626
V40      28.612626
V37      28.612626
V39      28.612626
D4       28.604667
Length: 212, dtype: float64
აქ ძაან ბევრია
ამიტო კაი გაფილტვრა მოგვიწევს
Columns with more than 20% missing values in Transaction Table:
dist2    93.628374
D7       93.409930
D13      89.509263
D14      89.469469
D12      89.041047
           ...    
V239     76.053104
V238     76.053104
V234     76.053104
V227     76.053104
V222     76.053104
Length: 168, dtype: float64
მარა ამიტომ ავირჩიოთ აქ რამე threshold da vnaxe ro 60% ზე ზევით იწყება მინიმუმ 76 % იუდან ამიტო მაგაზე მაღლებს გადავყრი.

ახლა დავიწყოთ დამუშავება

ჯერ მოდი ვცადოთ რომ დავაჯოინოთ იმიტომ რომ სიგრძეებში არ იყოს ერორი ( ვცადე ამის გარეშე და ერორი იყო) და ისე გავფილტროთ 

მერე ისევ გავყოთ და 20%NA იანები ამოვაგდოთ იდენთითიდან და 60%NA ზე მეტიანები ტრანსაცტიონიდან

ვცადე ჯერ შეერთება და მერე დაყოფა identity და transactio ად და მერე NA-ების დამუშაბვება მარა ჯოინის შემდეგ ძალიან აიწია identity-ში ამიტომ ისევ ჯერ წავშალოთდა მერე დავაჯოინოთ ჯობია.

ცალ ცალკე გამოვიდა და ასეთი შედეგი გვაქ :


--- Column Categorization ---
Identity numeric columns: 9
Identity low cardinality categorical columns: 10
Identity high cardinality categorical columns: 2
Transaction numeric columns: 211
Transaction low cardinality categorical columns: 11
Transaction high cardinality categorical columns: 2

--- Processed Data Information ---
Processed identity train data shape: (115658, 32)
Processed identity test data shape: (28575, 32)
Processed transaction train data shape: (472432, 240)
Processed transaction test data shape: (118108, 240)
Identity processing time: 1.30 seconds
Transaction processing time: 13.95 seconds

და საბოლოოდ

--- Checking for Remaining NaN Values ---
Identity train NaN count: 0
Transaction train NaN count: 0

ეხა ვცდი დამერჯვას მარა 
--- Checking for NAs in Merged Datasets ---
Merged train dataset shape: (472432, 272)
Merged test dataset shape: (118108, 272)
NAs in merged train dataset: 11416768
NAs in merged test dataset: 2865056

ოოოო ეს ძააან ბევრი na
ამიტო სხვა გზა გვინდა იმის მაგივრად რომ 0 ებით ანდაც ყველაზე ხშირებით შევავსოთ მაგრამ არა, ვფიქრობ გაუსის ან სხვა განაწილებით შევავსო.
Iterative imputation with Bayesian Ridge for numeric features
MICE (Multiple Imputation by Chained Equations) for categorical features

ნუ ვცადე მარა Iterative imputation - არის საშიშიო გამოყენება უსაფროთხო არარიო ამიტო ნუმერიქალებში საშუალო იყოს და კატეგორიულშიც ანალოგიურად.

ესე გამოვიდა და ახლა ასეთი სიტუაცია გვაქ:
Merged train dataset shape: (472432, 272)
Merged test dataset shape: (118108, 272)

ამიტომ ეხა კორელაციები უნდა მოვაშოროთ და ამის მერე RFE
კორელაციის ფილტრი 90% ზე 
ხოლო RFE 50 feature
Removing 113 highly correlated features...
Dataset shape after correlation removal - train: (472432, 160)
Dataset shape after correlation removal - test: (118108, 160)
ეს კორელაციამ კარგად მოაშოეა 113 სვეტი

ამის მერე RFE იყო გაშვებული ასე 1 საათი იჩალიჩა 50 სვეტამდე დაყვანამდე და დაიყვანა

--- Final Dataset Information ---
Final train dataset shape: (472432, 50)
Final test dataset shape: (118108, 50)
Features reduced from 273 to 50

--- Feature Selection completed successfully ---
Final dataset has 50 features after preprocessing and feature selection.

--- Detailed Analysis of Selected Features ---
Selected features from identity dataset: 10 (20.0%)
Selected features from transaction dataset: 40 (80.0%)
The identity flag 'has_identity_data' was NOT selected as an important feature.

https://dagshub.com/konstantine25b/IEEE-CIS-Fraud-Detection.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D - აქ არის პრეპროცესინგის ექსპერიმენტი

კაი ახლა გვაქ პრეპროცესინგის ექსპერიმენტი mlflow-ზე

# logistic_regression experiment

ახლა გავაკეთოთ პირველი ყველაზე მარტივი ექსპერიმენტი logistic_regression-ით.

ფაილი : ieee-cis-fraud-logistic_regression.ipynb

დავალოუდე მოდელი და შემდეგ მისი პრეპროცესინგის pipeline და ამის მერე 
უკვე logistic regression გავუშვი


Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.80      0.88    113975
           1       0.10      0.60      0.17      4133

    accuracy                           0.79    118108
   macro avg       0.54      0.70      0.53    118108
weighted avg       0.95      0.79      0.86    118108

ნუ საკმაოდ დიდი სხვაობაა 0 -ებს შედარებით უკეთ აფრედიქთებს precision მაღალი აქვს, მაგრამ ერთი რამე რაც არ გამოგვრჩა და არ გაგვითვალისწინებია არის რომ ძალიან ბევრი 0 -ია და ძალიან ცოტა 1 ანუ აქ დიდი დისბალანსია ამიტომ მოდელს უჭირს დაფრედიქთება. უნდა მოვიფიქრო სხვა რამე.
https://dagshub.com/konstantine25b/IEEE-CIS-Fraud-Detection.mlflow/#/experiments/1/runs/bc479b234c0948b59cbfab988fd279ac

# logistic_regression experiment 2

იქიდან გამომდინარე რომ logistic regression- ით უკეთესი შედეგის დადებასც მგონია შესაძლებელი ამიტომ კიდე ვცდი
ეხა regularization , scaling, kfold-ს დავამატებ და ვნახოთ რამდენად შეიცვლება შედეგი.


ფაილი : ieee-cis-fraud-logistic_regression-2.ipynb

logistic regression_2 ამ ფაილშიც იგივე ნაიურად წამოვიღებ დატას როგორც logistic regression , მაგრამ დავამუშავებ ასე:

დავამატე StandardScaler, StratifiedKFold(n_splits=5, shuffle=True, random_state=42), ასევე გვაქვს რეგულარიზაციის პარამეტრები.

ეხა დავფტოთ Fitting 5 folds for each of 6 candidates, totalling 30 fits

მეტი აღარ ვქენი იმიტორო ძაან დროში იწელება. ამასაც 15-20 წუთი მოუნდა და ისე 480 fit -ის გაკეთებას ვაპირებდი კიდე კაი არ გავაკეთე თორე 4 საათი უნდა ყოფილიყო ამაზე გაშვებული.
Best parameters: {'classifier__C': 10.0, 'classifier__class_weight': 'balanced', 'classifier__penalty': 'l2', 'classifier__solver': 'liblinear'}
Best cross-validation score (Average Precision): 0.0755

ხოლო საბოლოოდ:


Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.78      0.87    113975
           1       0.10      0.65      0.17      4133

    accuracy                           0.77    118108
   macro avg       0.54      0.71      0.52    118108
weighted avg       0.95      0.77      0.84    118108

ხოო ახლა წინა და ეს logistic_regression-ები რომ შევადაროთ:

recall-ში მეორე ჯობია და დანარჩენებში fraud detection-ში კონკრეტულად თანაბარია ამიტომ მეორე ანუ ეს მოდელი ჯობია წინას.

# decision-tree experiment 

ახლა გავტესტოთ decision tree.

ფაილი: ieee-cis-fraud-detection-decision-tree.ipynb

იგივე ნაირად გავუკეთეთ პრე პროცესინგი და ახლა მინეცით რამდენიმე ჰიპერ პარამეტრი და კროს ვალიდაციით ვიპოვეთ
საუკეთესო:
param_grid = {
    'classifier__max_depth': [5, 10],  
    'classifier__min_samples_split': [10],  
    'classifier__class_weight': ['balanced'] 
}

შედეგად 

Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.84      0.91    113975
           1       0.15      0.77      0.25      4133

    accuracy                           0.84    118108
   macro avg       0.57      0.80      0.58    118108
weighted avg       0.96      0.84      0.89    118108


logistic_regression ზე ბევრად უფრო კარგად იმუშავა 
თითოეულ კომპონენტში უფრო კარგი შედეგი აქ 
precision  0.05 ით გავაუმჯობესეთ, recall 0.12  ხოლო f1-score 0.08 ით 

ნუ decision tree-იც ცუდად მუშაობს მაგრამ ბევრად ჯობია წინებს.
https://dagshub.com/konstantine25b/IEEE-CIS-Fraud-Detection.mlflow/#/experiments/3/runs/f54a6bf3c1744e4ea742ff8fc3e647d9

# random forest experiment 

ახლა უკვე გადავიდეთ მოდელებზე რომნლებიც არადაბალანსებულ მონაცემებს ბევრად უკეთ უმკლავდებიან.
დავიწყთ random forest-ით ანუ გამოვიყენოთ bagging მიდგომა

ვაკეთებთ kfold classificationს და ასევე ჰიპერპარამეტრების მოსინჯვას 
param_grid = {
    'classifier__n_estimators': [50, 100],  # Number of trees
    'classifier__max_depth': [10, 15],  # Max depth of trees
    'classifier__class_weight': ['balanced', 'balanced_subsample']  # Class weight options
}

ხოო მოდელი დატრეინინგდა და აი შედეგი:

Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.95      0.97    113975
           1       0.32      0.72      0.45      4133

    accuracy                           0.94    118108
   macro avg       0.66      0.83      0.71    118108
weighted avg       0.97      0.94      0.95    118108

მოდელი ძალიან გაუმჯობესდა:
 1       0.32      0.72      0.45      4133


precision - გაორმაგდა, თითქმის გაორმაგდა f1-score. მოცემულ არადაბალანსებულ მონაცემებში საკმაოდ კარგი შედეგია
ნუ ჯერ გავტესტავ სხვადასხვა გზას და მერე ვნახავ სხვანაირ პრე პროცესინგებსაც.

https://dagshub.com/konstantine25b/IEEE-CIS-Fraud-Detection.mlflow/#/experiments/4/runs/90ea34d328ae4a9eaeb7eae0d00335e3

ფაილი: IEEE-CIS-Fraud-Detection_random_forest.ipynb

ახლა კიდე დროა boosting- ზე გადავიდეთ და ჯერ გავტესტოთ adaboost.- ვნახოთ bagging-ზე უკეთესი შედეგი თუ ექნება.

მარა სანამ გავტესტავ ჯერ დავფიქრეთ რომელს უფრო კარგი შედეგი უნდა ქონდეს bagging-ს თუ boosting-ს
bagging- ვარიაციის შესამცირებლად ხოლო boosting bias-ის შესამცირებლად, რადგან ჩვენ ვეძეთ froud-ებს გვინდა რომ
precission -იყოს მაღალი და ანუ რაც შეიძლება შევამციროთ bias. მაგრამ შეიძლება ბევრი outlier გვქონოდა და ამას
random forest-ის უპირატესობა გამოეწვია. მარა მაინც boosting მგონია რო უკეთ იმუშაბვებს რადგან bias-ის შემცირებაა გვინდა.


# adaboost experiment 

კაი ახლა გავტესტოთ adaboost:
იგივენ ნაირი პრე პროცესინგი, 
ეს კიდე ჰიპერ პარამეტრები:
param_grid = {
    'classifier__n_estimators': [50, 100],  # Number of boosting stages
    'classifier__learning_rate': [0.1, 0.5],  # Learning rate shrinks the contribution of each classifier
    'classifier__base_estimator__class_weight': ['balanced']  # Class weight for the base estimator
}

Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.85      0.92    113975
           1       0.16      0.77      0.26      4133

    accuracy                           0.85    118108
   macro avg       0.57      0.81      0.59    118108
weighted avg       0.96      0.85      0.89    118108

ხოო აქ გაუარესდა და random forest-მა აჯობა, ხოლო ამან ოდნავ აჯობა decision_tree-ს
ნუ როგორც ჩანს ან კომპლექსურობა აკლია ამ მოდელს ან ოვერფიტში წავიდა ან მონაცემებში დიდი ვარიაციია და მაგიტო აჯობა bagging-ის მიდგომამ. მარა ამის რეალურად გასაგებად გავტესტოთ უფრო კომპლექსური მოდელი xgboost

https://dagshub.com/konstantine25b/IEEE-CIS-Fraud-Detection.mlflow/#/experiments/7/runs/c8a726876efd40939c7008d2c993db6a

ფაილი: IEEE-CIS-Fraud-Detection_adaboost.ipynb


# xgboost experiment 1

დავიწყოთ xgboost-ის გატესტვა
იგივე ნაირად დავამუშავეთ მონაცემები
param_grid = {
    'classifier__n_estimators': [50, 100],  # Number of boosting rounds
    'classifier__max_depth': [3, 6],  # Max depth of trees
    'classifier__learning_rate': [0.1],  # Learning rate
    'classifier__scale_pos_weight': [1, 10]  # Weight of positive class (for imbalanced data)
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
ამის შემთხვევაში ისევ ჰიპერპარამეტრები პლიუს kfold ანუ 24 ცალი fit მოუწევს ამ შემთხვევაში

Classification Report:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99    113975
           1       0.88      0.37      0.52      4133

    accuracy                           0.98    118108
   macro avg       0.93      0.68      0.75    118108
weighted avg       0.97      0.98      0.97    118108 

ნუ აქ precission ძალიან კარგი აქვს, მაგრამ recall დაუვარდა შესაბამისად დაბალი გამოუვიდა f1 score- ეს მოდელი საუკეთესოა ჯერჯერობით თუ ჩვენ precission გვაინტერესებს.

# xgboost experiment 2

მგონია რომ გაუმჯობესება შეიძლება მიტომ უფრო მეტ ჰიპერპარამეტრზე გავტესტავ ახლა
ვცადოთ ასეთით 
param_grid = {
    'classifier__n_estimators': [100],  # Use more trees for better learning
    'classifier__max_depth': [4, 6],  # Try deeper trees to capture complex patterns
    'classifier__learning_rate': [0.1, 0.05],  # Try a slower learning rate
    'classifier__scale_pos_weight': [20, 30],  # Increase weight for positive class
    'classifier__min_child_weight': [1, 3],  # Control overfitting
    'classifier__subsample': [0.8],  # Use subsampling to prevent overfitting
    'classifier__colsample_bytree': [0.8]  # Use column subsampling
} 
აქ 48 fit დაჭირდება
Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.92      0.95    113975
           1       0.26      0.76      0.39      4133

    accuracy                           0.92    118108
   macro avg       0.62      0.84      0.67    118108
weighted avg       0.96      0.92      0.93    118108

ამ ცდამ უფრო გააუარესა ამიტომ უფრო უკეთესი გზა იქნებ იყოს 
ვცადოთ კიდევ:

# xgboost experiment 3

### Define hyperparameter grid focused on improving precision for minority class
param_grid = {
    'classifier__n_estimators': [200],  # Reduced options
    'classifier__max_depth': [4, 6],  # Reduced options
    'classifier__learning_rate': [0.01],  # Only one learning rate
    'classifier__scale_pos_weight': [25, 35],  # Reduced options
    'classifier__min_child_weight': [3],  # Only one option
    'classifier__subsample': [0.8],  # Only one option
    'classifier__colsample_bytree': [0.8],  # Only one option
    'classifier__gamma': [0.1],  # Only one option
    'classifier__reg_alpha': [0.1],  # Only one option
    'classifier__reg_lambda': [1.0]  # Only one option
}

გავართულოთ პარამეტრები და გამოვიდა: Fitting 5 folds for each of 128 candidates, totalling 640 fits 
კაი ხანი ლოდინი მოგვიწევს, ვნახოთ შედეგი რა იქნება:
გავიდა 30 წუთი და შედეგი გაუმჯობესდა საგრძნობლად.
Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99    113975
           1       0.68      0.60      0.64      4133

    accuracy                           0.98    118108
   macro avg       0.83      0.80      0.81    118108
weighted avg       0.98      0.98      0.98    118108


https://dagshub.com/konstantine25b/IEEE-CIS-Fraud-Detection.mlflow/#/experiments/8/runs/229c272df1ec46f9b7e97ded0942dc53

ფაილი: IEEE-CIS-Fraud-Detection_xgboost.ipynb
კაი ნუ პირველ xg boost -ს საუკეთესი precission აქვს ჯერჯერობით. მაგრამ ამას xgboost-ებში საუკეთესო f1 score აქვს და დაბალანსებულია ანუ ეს მოდელი საუკეთესოა.


აქამდე რომ შევაჯამოთ 

random forest-მა 
  1       0.32      0.72      0.45      4133

  მოგვცა კარგი recall - მაგრამ დაბალი precission ანუ ბევრ სწორი ტრანზაქციზე თქვა რომ froud იყო რაც ცუდია


პირველად გაშვებულმა xgboost-მა 
 1       0.88      0.37      0.52      4133
 მოგვცა ძალიან მაღალი precission მარა ძაან დაბალი recall რაც ნიშნავს იმას რომ ძალიან ბევრი შემთხვევა გამორჩა. ამიტომაც ესეც ცუდია

 ჯერჯერრობით საუკეთესოა ბოლო მოდელი 
   1       0.68      0.60      0.64      4133
   ამ მოდელს recall და  precission დაბალანსებულად აქვთ რაც იწვევს დანარჩენ მოდელებზე უკეთეს f1 score-ს.




# LETS TRY ONLY TRANSACTION DB

კაი ნუ დამერჯვით გავტესტეტ და კიდე გავტესტავ სხვანაირად მაქვს რაღაც იდეები და გავაკეთებ მერე მაგრამ ჯერ მინდა
ვცადრო მხოლოდ transaction -ის ცხრილის გამოყენება ოღონდ აქ პირდაპირ xgboost-ზე გავტესტავ რადგან ვნახეთ რომ 
ყველაზე კარგი შედეგი სწორედ მას ქონდა.

კაი დავიოწყოთ :
# Fraud_Detection_Transaction_Only

For numerical columns, fill with median

ხოლო კატეგორიულებში ვისაც 60% ზე მეტი მისსინგ ქონდა წავშალეთ და დანარჩნებში ყველაზე ხშირით შევავსეთ.
preprocess-ინგის შემდეგ მივიღეთ ასეთი სიტუაცია:

ამის მერე future engineering

--- Feature Selection: Correlation Filter ---
Selected 96 features with correlation > 0.05

ამის მერერ RFE და აიურჩა 50 ცალუ feature
da ბოლოს დავატრეინინგე
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [4, 6],
    'classifier__learning_rate': [0.01, 0.05],
    'classifier__scale_pos_weight': [25, 35],
    'classifier__min_child_weight': [3],
    'classifier__subsample': [0.8],
    'classifier__colsample_bytree': [0.8],
    'classifier__gamma': [0.1],
    'classifier__reg_alpha': [0.1],
    'classifier__reg_lambda': [1.0]
}

--- Model Evaluation ---
              precision    recall  f1-score   support

           0       0.98      0.99      0.98    113975
           1       0.53      0.37      0.43      4133

    accuracy                           0.97    118108
   macro avg       0.75      0.68      0.71    118108
weighted avg       0.96      0.97      0.96    118108

ეს მოდელინ საკმაოდ სუსტია სხვა კომპლექსურ მოდელებთან შედარბით.  მარა მაინც აქ ცოტა ჰიპერ პარამეტრი იყო ამიტო მოდი კიდე ვცდი უფრო მეტი ჰიპერ პატამეტრით.

# Fraud_Detection_Transaction_Only 2

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [4, 6, 8],
    'classifier__learning_rate': [0.01, 0.05],
    'classifier__scale_pos_weight': [25, 35],  # To handle class imbalance
    'classifier__min_child_weight': [1, 3],
    'classifier__subsample': [0.8],
    'classifier__colsample_bytree': [0.8],
    'classifier__gamma': [0, 0.1],
    'classifier__reg_alpha': [0, 0.1],
    'classifier__reg_lambda': [1.0]
}
  precision    recall  f1-score   support

           0       0.98      0.99      0.98    113975
           1       0.55      0.40      0.47      4133

    accuracy                           0.97    118108
   macro avg       0.77      0.70      0.72    118108
weighted avg       0.96      0.97      0.97    118108

ეს წინასთან შედარებით უკეთესია მაგრამ აშკარად identity ცხრილმა ბევრად უფრო კარგი მოდელები გამოიყვანა


https://dagshub.com/konstantine25b/IEEE-CIS-Fraud-Detection.mlflow/#/experiments/9/runs/7b4b0905b5464487a50094ef5398a509

ფაილი:IEEE-CIS Fraud Detection_Transaction_Only.ipynb

ნუ ამით დავასრულოთ მხოლოდ transactions db ის გამოყენება

კაი ახლა ისევ დავუბრუნდეთ ამ მონაცემების დამერჯვა ასეთი იდეა მაქვს, რომ ჯერ გავაერთიანო დ


--- Fraud Distribution Analysis ---
Total transactions: 590540
Fraud transactions: 20663 (3.50%)
Non-fraud transactions: 569877 (96.50%)


--- Identity Information Analysis ---
Transactions with identity information: 144233
Percentage of transactions with identity info: 24.42%

Fraud distribution for transactions WITH identity information:
Total: 144233
Fraud: 11318 (7.85%)
Non-fraud: 132915 (92.15%)

Fraud distribution for transactions WITHOUT identity information:
Total: 446307
Fraud: 9345 (2.09%)
Non-fraud: 436962 (97.91%)

როგორც ვფიქრობდით identity სვეტი მნიშვნელობვანია რადგან აქ ვისაც identity აქვს მანდ უფრო მეტია განაწილება.
ასეთი აზრი მაქვს რომ გავყო მონაცემები ანუ ერტი db სადაც იქნებიან identiry + transactions და მეორე მხოლოდ transactions ასე უფრო ლოგიკურად იქნება წესით დატრეინგება შესაძლებელი.

წინა ექსპერიმენტებიდან გამომდინარე ვიზავს ასე ჯერ high null column-ებს მოვაშორებ - 20% ზე მაღლები identityდან ხოლო 60% ზე მაღლები transactions, ამ