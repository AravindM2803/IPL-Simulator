import numpy as np
import pandas as pd
import pickle

with open('Data/Players.pkl', 'rb') as fp:
    players = pickle.load(fp)
with open('Data/Venue.pkl', 'rb') as fp:
    venue = pickle.load(fp)
with open('Data/Teams.pkl', 'rb') as fp:
    teams = pickle.load(fp)


def get_onehot(df_inp):
    inp_cols_set = set(df_inp.columns)
    df = df_inp.copy().reset_index(drop=True)
    if "Toss" in inp_cols_set:
        df1 = pd.get_dummies(df.Toss, prefix="Toss")
        col = "Toss_"
        df_columns = set(df[col[:-1]])
        not_there = list(set(teams)-df_columns)
        data = np.zeros((df.shape[0], len(not_there)))
        df_add = pd.DataFrame(data=data, columns=[col+i for i in not_there])
        df1 = pd.concat([df1, df_add], axis=1)

    df2 = pd.get_dummies(df.Venue, prefix="Venue")
    col = "Venue_"
    df_columns = set(df[col[:-1]])
    not_there = list(set(venue)-df_columns)
    data = np.zeros((df.shape[0], len(not_there)))
    df_add = pd.DataFrame(data=data, columns=[col+i for i in not_there])
    df2 = pd.concat([df2, df_add], axis=1)

    df3 = pd.get_dummies(df.Batting_Team, prefix="Batting_Team")
    col = "Batting_Team_"
    df_columns = set(df[col[:-1]])
    not_there = list(set(teams)-df_columns)
    data = np.zeros((df.shape[0], len(not_there)))
    df_add = pd.DataFrame(data=data, columns=[col+i for i in not_there])
    df3 = pd.concat([df3, df_add], axis=1)

    df4 = pd.get_dummies(df.Bowling_Team, prefix="Bowling_Team")
    col = "Bowling_Team_"
    df_columns = set(df[col[:-1]])
    not_there = list(set(teams)-df_columns)
    data = np.zeros((df.shape[0], len(not_there)))
    df_add = pd.DataFrame(data=data, columns=[col+i for i in not_there])
    df4 = pd.concat([df4, df_add], axis=1)

    df5 = pd.get_dummies(df.Striker, prefix="Striker")
    df5_columns = set(df["Striker"])
    never_striker = list(set(players)-df5_columns)
    data = np.zeros((df.shape[0], len(never_striker)))
    df5_add = pd.DataFrame(data=data, columns=[
                           "Striker_"+i for i in never_striker])
    df5 = pd.concat([df5, df5_add], axis=1)

    df6 = pd.get_dummies(df.Non_Striker, prefix="Non_Striker")
    df6_columns = set(df["Non_Striker"])
    never_non_striker = list(set(players)-df6_columns)
    data = np.zeros((df.shape[0], len(never_non_striker)))
    df6_add = pd.DataFrame(data=data, columns=[
                           "Non_Striker_"+i for i in never_non_striker])
    df6 = pd.concat([df6, df6_add], axis=1)

    df7 = pd.get_dummies(df.Bowler, prefix="Bowler")
    df7_columns = set(df["Bowler"])
    never_bowler = list(set(players)-df7_columns)
    data = np.zeros((df.shape[0], len(never_bowler)))
    df7_add = pd.DataFrame(data=data, columns=[
                           "Bowler_"+i for i in never_bowler])
    df7 = pd.concat([df7, df7_add], axis=1)

    df_one_hot = df.copy(deep=True)
    if "Toss" in inp_cols_set:
        df_one_hot = pd.concat([df, df1, df2, df3, df4, df5, df6, df7], axis=1)
    else:
        df_one_hot = pd.concat([df, df2, df3, df4, df5, df6, df7], axis=1)
    df_result = pd.DataFrame(df_one_hot["Result"])
    if "Toss" in inp_cols_set:
        df_one_hot = df_one_hot.drop(columns=[
            "Toss", "Venue", "Batting_Team",
            "Bowling_Team", "Striker",
            "Non_Striker", "Bowler", "Result"])
    else:
        df_one_hot = df_one_hot.drop(columns=[
            "Venue", "Batting_Team", "Bowling_Team",
            "Striker", "Non_Striker", "Bowler", "Result"])
    return df_one_hot.reindex(sorted(df_one_hot.columns), axis=1), df_result


def get_cont_ids(df):
    prev = None
    start = 0
    cont_ids = []
    for ind, row in df.iterrows():
        curr = [[row['Venue'], row['Batting_Team'], row['Bowling_Team']]]
        if curr != prev and prev is not None:
            cont_ids.append([start, ind])
            start = ind
        prev = curr
    cont_ids.append([start, df.shape[0]])
    return cont_ids


def get_df_split(df):
    df_list = []
    for start, end in get_cont_ids(df):
        df_list.append(df[start:end].reset_index(drop=True))
    return df_list
