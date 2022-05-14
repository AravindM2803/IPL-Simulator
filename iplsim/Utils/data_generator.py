import random
import numpy as np
import pandas as pd
import math
import pickle
import tensorflow as tf

with open('Data/Players', 'rb') as fp:
    players = pickle.load(fp)
with open('Data/Venue', 'rb') as fp:
    venue = pickle.load(fp)
with open('Data/Teams', 'rb') as fp:
    teams = pickle.load(fp)


class LiteDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 df: pd.DataFrame,
                 window: int = 6,
                 batch_size: int = 64,
                 shuffle: bool = True,
                 validate: bool = False,
                 overs_start: int = None,
                 overs_end: int = None,
                 ):
        self.df = df
        self.overs_start = overs_start
        self.overs_end = overs_end
        self.window = window
        self.batch_size = batch_size
        self.inp_cols = None
        self.out_cols = None
        self.validate = validate
        self.batches = self.__fill_batches()
        self.len_dataset = len(self.batches)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __shuffle(self):
        for key in self.batches:
            random.shuffle(self.batches[key])

    def __fill_getitem_batches(self):
        self.getitem_batches = []
        for key in self.batches:
            curr_size_list = self.batches[key]
            num_items = math.ceil(len(curr_size_list)/self.batch_size)
            for i in range(1, num_items + 1):
                temp = curr_size_list[(i-1)*self.batch_size: i*self.batch_size]
                self.getitem_batches.append(temp)
        if self.shuffle:
            random.shuffle(self.getitem_batches)

    def __fill_batches(self):
        df_list = get_df_split(self.df)
        if self.validate:
            if (False and self.overs_start is not None and
                    self.overs_end is not None):
                small_df_list = [df[np.logical_and(
                    df["Overs"] <= self.overs_end,
                    df["Overs"] >= self.overs_start)].reset_index(
                    drop=True)
                    for df in df_list[math.floor(0.80*len(df_list)):]]
            else:
                small_df_list = [
                    df for df in df_list[math.floor(0.80*len(df_list)):]]
        else:
            small_df_list = [
                df for df in df_list[:math.floor(0.80*len(df_list))]]
        # one_hot_lis = [(inp.reindex(sorted(inp.columns), axis=1), out)
        #                for inp, out in one_hot_lis]
        batches = {}
        for i in range(1, self.window + 1):
            batches[i] = []
        for df in small_df_list:
            for end in range(df.shape[0]):
                if self.overs_start is not None and self.overs_end is not None:
                    if (df["Overs"][end] < self.overs_start or
                            df["Overs"][end] > self.overs_end):
                        continue
                start = max(end - self.window + 1, 0)
                size = end - start + 1
                batches[size].append(
                    df[start: end + 1])
        return batches

    def __len__(self):
        return len(self.getitem_batches)

    def __getitem__(self, index):
        onehot_list = [get_onehot(df) for df in self.getitem_batches[index]]
        x = np.array([x[0].values for x in onehot_list])
        y = np.array([x[1].values[-1] for x in onehot_list])
        return x, np.reshape(y, (-1, ))

    def on_epoch_end(self):
        if self.shuffle:
            self.__shuffle()
        self.__fill_getitem_batches()


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 df: pd.DataFrame,
                 window: int = 6,
                 batch_size: int = 64,
                 shuffle: bool = True,
                 validate: bool = False,
                 overs_start: int = None,
                 overs_end: int = None,
                 ):
        self.df = df
        self.overs_start = overs_start
        self.overs_end = overs_end
        self.window = window
        self.batch_size = batch_size
        self.inp_cols = None
        self.out_cols = None
        self.validate = validate
        self.batches = self.__fill_batches()
        self.len_dataset = len(self.batches)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __shuffle(self):
        for key in self.batches:
            random.shuffle(self.batches[key])

    def __fill_getitem_batches(self):
        self.getitem_batches = []
        for key in self.batches:
            curr_size_list = self.batches[key]
            num_items = math.ceil(len(curr_size_list)/self.batch_size)
            for i in range(1, num_items + 1):
                temp = curr_size_list[(i-1)*self.batch_size: i*self.batch_size]
                inp_batch = np.array([x[0] for x in temp])
                out_batch = np.array([x[1] for x in temp])
                self.getitem_batches.append((inp_batch, out_batch))
        if self.shuffle:
            random.shuffle(self.getitem_batches)

    def __fill_batches(self):
        df_list = get_df_split(self.df)
        if self.validate:
            if self.overs_start is not None and self.overs_end is not None:
                one_hot_lis = [get_onehot(
                    df[np.logical_and(
                        df["Overs"] <= self.overs_end,
                        df["Overs"] >= self.overs_start)].reset_index(
                            drop=True))
                    for df in df_list[math.floor(0.80*len(df_list)):]]
            else:
                one_hot_lis = [get_onehot(
                    df) for df in df_list[math.floor(0.80*len(df_list)):]]
        else:
            one_hot_lis = [get_onehot(
                df) for df in df_list[:math.floor(0.80*len(df_list))]]
        one_hot_lis = [(inp.reindex(sorted(inp.columns), axis=1), out)
                       for inp, out in one_hot_lis]
        batches = {}
        for i in range(1, self.window + 1):
            batches[i] = []
        for inp, out in one_hot_lis:
            inp_numpy, out_numpy = inp.values, out.values
            for end in range(inp.shape[0]):
                start = max(end - self.window + 1, 0)
                size = end - start + 1
                batches[size].append(
                    (inp_numpy[start: end + 1], out_numpy[end]))
        self.inp_cols = inp.columns.to_list()
        self.out_cols = out.columns.to_list()
        return batches

    def __len__(self):
        return len(self.getitem_batches)

    def __getitem__(self, index):
        x, y = self.getitem_batches[index]
        return x, np.reshape(y, (-1, ))

    def on_epoch_end(self):
        if self.shuffle:
            self.__shuffle()
        self.__fill_getitem_batches()


def get_onehot(df_inp):
    df = df_inp.copy().reset_index(drop=True)
    # df1=pd.get_dummies(df.Toss, prefix="Toss")
    # col = "Toss_"
    # df_columns = set(df[col[:-1]])
    # not_there = list(set(teams)-df_columns)
    # data = np.zeros((df.shape[0], len(not_there)))
    # df_add = pd.DataFrame(data=data, columns=[col+i for i in not_there])
    # df1 = pd.concat([df1, df_add], axis=1)

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
    # df_one_hot=pd.concat([df,df1,df2,df3,df4,df5,df6,df7], axis=1)
    df_one_hot = pd.concat([df, df2, df3, df4, df5, df6, df7], axis=1)
    df_result = pd.DataFrame(df_one_hot["Result"])
    # df_one_hot=df_one_hot.drop(columns=["Toss","Venue","Batting_Team","Bowling_Team","Striker","Non_Striker","Bowler","Result"])
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
