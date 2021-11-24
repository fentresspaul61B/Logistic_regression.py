import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(pd.__version__)

# datasets

tbl = pd.read_csv("covid_and_voting_data.csv")
labels = tbl["PARTY"].map(lambda x: "Democrat" if x == 1 else "Republican")
tbl["labels"] = labels


st.set_page_config(page_title="ML: LogisticRegression", page_icon = '🤖')

st.write(
"""
# Logistic Regression with sklearn
## 2020 State Presidential Vote vs. Mask Use vs. Vaccination Rates
Logistic Regression is a classifcation algorithm used to
classify two or more classes.

** About the Data: **
In order to show how to implement Logistic Regression using sklearn, lets take
a look at a simple data set about 2020 mask use and COVID vaccination rates in America by state.
- STNAME: Name of the State
- FREQ/ALWAYS: porportion of people reported to wear a mask frequently or always
- FULLY_VACCINATED: porportion of people fully vaccinated in the states
- PARTY: State voted Democratic or Republican in 2020 (1: Republican, 0: Democrat)


"""
)

st.write(tbl)



st.write(
"""
Plotting the data by the features:
"""

)

# Creating vis:

import matplotlib
x = np.array(tbl["FREQ/ALWAYS"])
y = np.array(tbl["FULLY_VACCINATED"])
label = tbl["labels"]
party = tbl["PARTY"]
colors = ['red','blue']
plt.title("Mask Use vs. Vaccination Rates and Political Party by State")
plt.xlabel("Proportion of Mask Use")
plt.ylabel("Proportion of Vaccinated People")

fig = plt.scatter(x,y, c = party, cmap=matplotlib.colors.ListedColormap(colors))
plt.legend()
st.pyplot(fig = plt)


st.write(
"""
# Model Implementation
"""
)

with st.echo():
    # Installing dependencies
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Choosing arbitrary random seed:
    seed = 14

    # Creating data / labels
    X, Y = tbl[["FREQ/ALWAYS", "FULLY_VACCINATED"]], tbl["PARTY"]

    # Splitting the data:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=seed)

    # Fitting the model:
    clf = LogisticRegression(random_state=seed).fit(X_train, y_train)

    # Making predictions:
    y_pred = clf.predict(X_test)

    # Printing Accuracy:
    accuracy = sum(y_pred == y_test) / len(y_test)

    st.write(accuracy)

print(len(y_test))
st.write(
"""
# Conclusion
Using Logistic Regression we were able to predict with 100% accuracy which states
voted Republican or Democratic in the 2020 election based on their
mask use and vaccine rates.

** Note: **
Due to the very small amount of data, this is not a necessarily a useful result.
Our training set had 43 samples while our test had 8 samples.
The more difficult / insightful problem to solve would be predicting by county or city.




"""



)
