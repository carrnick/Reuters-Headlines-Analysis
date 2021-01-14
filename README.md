# Reuters-Headlines-Analysis
Sentiment analysis of Reuters news headlines using Sklearn's [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) to determine if headlines can predict if the S&P500 closed higher than the previous day.

![stock-market-thumb](https://user-images.githubusercontent.com/70597605/104617738-cb6e1780-5659-11eb-85c0-09341e743af9.jpg)

# Methodology
1) Get data of each Reuters headline and the date posted.
2) Transform dataset so each row contains a date and list of headlines for that date.
3) Use [yfinance](https://pypi.org/project/yfinance/) to get price history for the S&P 500.
4) Merge pricing history and headlines together, resulting in a dataset containing a date, list of headlines, and closing price for each date.
5) Loop through dataset and assign labels to each row. The label is 1 if the S&P 500 closed higher than the previous day, and 0 otherwise.
6) Pass list of headlines into TfidfVectorizer (this will be broken down later).
7) Use machine learning models to determine if headlines can predict if the S&P 500 closed higher than the previous day (this will be broken down later).
8) Analyze models with Confusion Matrices, ROC Curve, and Precision-Recall Curve.
# Machine Learning Models and Results
## Random Forest Classifier
*Confusion Matrix*

![confmatrf](https://user-images.githubusercontent.com/70597605/104616475-53ebb880-5658-11eb-95dd-49bbb82a7835.png)

*ROC Curve*

![rfrocauc](https://user-images.githubusercontent.com/70597605/104617213-2c492000-5659-11eb-96dd-3c5dd076fc68.png)

*Precision-Recall Curve*

![prcurve_rf](https://user-images.githubusercontent.com/70597605/104616414-446c6f80-5658-11eb-8294-7cd6b4b67fcf.png)

*Accuracy Scores*

![rfmeasures](https://user-images.githubusercontent.com/70597605/104617209-2b17f300-5659-11eb-9542-d95a68d2cc9c.png)



## XGBoost
*Confusion Matrix*

![confmatxg](https://user-images.githubusercontent.com/70597605/104616797-bc3a9a00-5658-11eb-83c1-07d4298462b3.png)

*ROC Curve*

![xgra](https://user-images.githubusercontent.com/70597605/104616796-bba20380-5658-11eb-9597-98fd73513252.png)

*Precision-Recall Curve*

![xgpr](https://user-images.githubusercontent.com/70597605/104616795-bba20380-5658-11eb-9d99-ad7caf479180.png)

*Accuracy Scores*

![xgclass](https://user-images.githubusercontent.com/70597605/104616794-bba20380-5658-11eb-9aee-37b5a60d4c3f.png)

## Linear Support Vector Machine
*Confusion Matrix*

![confmatLVC](https://user-images.githubusercontent.com/70597605/104617268-3cf99600-5659-11eb-83c7-eae5b766cdac.png)

*ROC Curve*

![lsvmra](https://user-images.githubusercontent.com/70597605/104617266-3cf99600-5659-11eb-8053-3456be72b657.png)

*Precision-Recall Curve*

![lvsmprc](https://user-images.githubusercontent.com/70597605/104617265-3c60ff80-5659-11eb-865a-35e6eeda6e0e.png)

*Accuracy Scores*

![lsvmmeasures](https://user-images.githubusercontent.com/70597605/104617263-3c60ff80-5659-11eb-90a2-8d2bd633fcfd.png)
