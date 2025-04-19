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

ვფიქრობ რომ რეპოზიტორიის სტუქტურა მექნება ასე ჯერ- პრე პროცესსინგ ცალკე ფაილი სადაც დავამუშავებ მტელ დატას.
და მერე სხვადასხვა მოდელისთვის სხვადასხვა ფაილი და 1 ცალუ model_inference.ipynb საბოლოოდ.

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

transactions - აქვს  394 columns ხოლო identity-ს 41 columns.

ამათი პირდაპირ დაჯოინებით რაღაც დატას პრობლემები მგონია რო შეიქმენაბ ამიტომ ჯერ გავარკვვიოთ რა აქვთ საერთო და საერთოდ როგორებია.

გამოვიკვიე
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

აღმოჩნდა რომ საკმაოდ ბევრი NA გვაქ ამიტო კაი დამუშავება მოგვიწევს.

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
მარა ამიტომ ავირჩიოთ აქ რამე threshoild da vnaxe ro 60% ზე ზევით იწყება მინიმუმ 76 % იუდან ამიტო მაგაზე მაღლებს გადავყრი.

ახლა დავიწყოთ დამუშავება

ჯერ მოდი ვცადოთ რომ დავაჯოინოთ იმიტომ რომ სიგრძეებში არ იყოს ერორი ( ვცადე ამის გარეშე და ერორი იყო) და ისე გავფილტროთ 

მერე ისევ გავყოთ და 20%NA იანები ამოვაგდოთ იდენთითიდან და 60%NA ზე მეტიანები ტრანსაცტიონიდან

ვცადე ჯერ შეერტება და მერე დაყოფა identity და transactio ად და მერე NA-ების დამუშაბვება მარა ჯოინის შემდეგ ძალიან აიწია identity-ში ამიტომ ისევ ჯერ წავშალოთდა მერე დავაჯოინოთ ჯობია.





