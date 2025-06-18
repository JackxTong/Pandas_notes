rolling r2 with mask 
should expect more stable r2

coeff change as rolling 

guess: subset rolling would be 
very similar R2 to all features

masked or un masked 

ridge lasso 

rfe: from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
selector = RFE(LinearRegression(), n_features_to_select=10)
selector.fit(X_train, y_train)
selected = X.columns[selector.support_]
