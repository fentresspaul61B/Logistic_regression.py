cases = pd.read_csv("covid_data/cases.csv")
counties = pd.read_csv("covid_data/counties.csv")
mask_use = pd.read_csv("covid_data/mask_use.csv")
vaccinations = pd.read_csv("covid_data/vaccinations.csv")
county_data = pd.read_csv("covid_data/county_data.csv")
population = pd.DataFrame(county_data.groupby("STNAME", as_index=False).sum()[["STNAME", "POPESTIMATE2020"]])

vax_fully_by_state = vaccinations.groupby('Province_State').max().reset_index()[['Province_State','People_Fully_Vaccinated']]
vax_fully_by_state = population.merge(vax_fully_by_state, how = 'left', left_on = 'STNAME', right_on = 'Province_State')
vax_fully_by_state["FULLY_VACCINATED"] = vax_fully_by_state['People_Fully_Vaccinated'] / vax_fully_by_state['POPESTIMATE2020']
vax_fully_by_state = vax_fully_by_state.drop(columns = ['Province_State', 'POPESTIMATE2020','People_Fully_Vaccinated'])



CD = county_data["ALWAYS"] +  county_data["FREQUENTLY"]
states = vax_fully_by_state["STNAME"]
key = "STNAME"
left = pd.DataFrame({"STNAME": states, "FREQ/ALWAYS": CD })
right = vax_fully_by_state
tbl = pd.merge(left, right, on=key)

# https://cookpolitical.com/2020-national-popular-vote-tracker

votes_2020 = pd.read_csv("covid_data/Popular vote backend - Sheet1 (1).csv")
votes_2020 = votes_2020.iloc[4:,0:2].rename(columns={"state": "STNAME"})
  # sliced
left = tbl
right = votes_2020
tbl = pd.merge(left, right, on=key)

tbl = tbl.rename(columns = {"called": "PARTY"})

one_hot_encoded = tbl["PARTY"].map(lambda x: 1 if x == "D" else 0)

tbl["PARTY"] = one_hot_encoded
# st.write(one_hot_encoded)
st.write(tbl.head())


tbl.to_csv("covid_and_voting_data.csv")






st.write(
"""
Lets seperate all of our x_values into bins, in order to calculate chances
of falling within a certain bin.
"""
)

with st.echo():
    import matplotlib.pyplot as plt
    import pandas as pd

    # Creating 10 bins for our x values:

    # bins = pd.cut(tbl["FREQ/ALWAYS"], 20)
    bins = pd.cut(tbl["fully_vaccinated"], 10)

    # Creating a bins column for tbl. Here we are summing together
    tbl["bin"] = [(b.left + b.right) / 2 for b in bins]

    # Groupby "fully_vaccinated" table in order put x values
    # into bins:
    grouped_tbl = tbl.groupby("bin")["called"].mean()

    # Creating line plot
    plt.xlabel("fully vaccinated rate")
    plt.ylabel("0: Republican, 1: Democrat")
    plt.plot(grouped_tbl)

    # Creating scatter
    x = tbl["fully_vaccinated"]
    y = tbl["called"]
    plt.scatter(x, y)

    # Creating sigmoid function
    def sigmoid(t):
        return 1 / (1 + (np.exp(-t)))

    lin = np.arange(min(x), max(x), 0.00001)
    plt.plot(lin, sigmoid(lin))

    st.pyplot(fig = plt)



# x = np.array(tbl["FREQ/ALWAYS"])
# y = np.array(tbl["People_Fully_Vaccinated per capita"])
# label = tbl["called"]
# colors = ['blue','red']
# plt.title("Mask Use vs. Vaccination Rates and Political Party by State")
# plt.xlabel("Proprtion of Mask Use")
# plt.ylabel("Proprtion of Vaccinated People")
# fig = plt.scatter(x,y, c = label, cmap=matplotlib.colors.ListedColormap(colors))
# st.pyplot(fig = plt)

st.write(

"""
# Logistic Regression Model from scratch in Python
** First lets split our training and testing data: **
"""

)

with st.echo():
    import numpy as np

    # Seperating labels from training data
    X = tbl[["FREQ/ALWAYS", "People_Fully_Vaccinated per capita"]]
    Y = label

    def train_test_split(data=X, labels=Y, test_size=0.1):
        # Getting Dimensions of Table:
        rows, columns = data.shape

        # Randomly shuffling features and labels
        p = np.random.permutation(rows)
        shuffled_data = data.iloc[p]
        shuffled_labels = labels.iloc[p]

        # Splitting the data:
        split = int(1 - (rows * test_size))
        X_train = shuffled_data[0:split]
        X_test = shuffled_data[split:]
        Y_train = shuffled_labels[0:split]
        Y_test = shuffled_labels[split:]

        return X_train, X_test, Y_train, Y_test

    X_train, X_test, Y_train, Y_test = train_test_split()


with st.echo():

    # Importing scipy in order to minimize loss function:
    from scipy.optimize import minimize

    # Logistic function:
    def sigmoid(t):
        return 1 / (1 + (np.exp(-t)))

    # Here is our Model:
    def logistic_regression(x, theta_1, theta_2):
        return np.log(theta_1 + theta_2 * x)

    # Defining a loss MSE function to find optimal parameters for
    # theta_2 and theta_2:
    def log_loss(params):
        return



    # Minimize loss function:
    # initial_guess = [1,1]
    # st.write(MSE(initial_guess))
    # result = minimize(MSE, initial_guess)
    # st.write(result)
    # theta_1, theta_2 = minimize(MSE, x0 = [0,0])
    # st.write(theta)

    # Defining logistic regression function which calculates
    # the chance the data point is 1 or 0, then using a simple
    # if else case, classifies the data point.
    def predict(x, theta_1, theta_2):
        chance = logistic_regression(x, theta_1, theta_2)
        if chance >= .5:
            return 1
        else:
            return 0
