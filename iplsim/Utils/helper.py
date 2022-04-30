import pandas as pd
import functools
import numpy as np
import random
import pickle
from IPython.display import display


with open('Data/BF_Cols.pkl', 'rb') as fp:
    BF_Cols = pickle.load(fp)
with open('Data/BS_Cols', 'rb') as fp:
    BS_Cols = pickle.load(fp)
# model_inn_1 = tf.keras.models.load_model('Models/Inn1_SimpleRNN_10.h5')
# model_inn_2 = tf.keras.models.load_model('Models/Inn2_LSTM_10.h5')


def display_batting_table(inn1, onlybatting=0):
    batting_lineup = inn1.Batting_lineup
    bowling_lineup = inn1.Bowling_lineup
    temp_df = pd.DataFrame(columns=["Batsman", "Runs", "Fours",
                                    "Sixes", "Balls Faced", "Dissmissal Type",
                                    "Dismissed By"])
    temp_stages = pd.DataFrame(columns=["Dismissed Batsman",
                                        "Team Runs", "Overs"])
    temp_bowling = pd.DataFrame(columns=["Bowler", "Runs Conceded",
                                         "Wickets Taken", "Overs",
                                         "Batsman Names"])
    temp_over_summary = pd.DataFrame(columns=["Over", "Bowler",
                                              "Runs Conceded",
                                              "Wickets Taken", "Total Score",
                                              "Total Wickets"])
    # Creating Batting Table and Runs Scored
    for i in batting_lineup:
        if (i.Entered_Match):
            row_dict = {
                "Batsman": i.Name, "Runs": i.Runs,
                "Fours": i.Fours_Hit,
                "Sixes": i.Sixes_Hit,
                "Balls Faced": i.Balls,
                "Dissmissal Type": i.Dismissal if i.Dismissal else 'Not Out',
                "Dismissed By": i.Dismissal_By if i.Dismissal_By else "-"
            }
        else:
            row_dict = {"Batsman": i.Name, "Runs": "-",
                        "Fours": "-",
                        "Sixes": "-",
                        "Balls Faced": "-",
                        "Dissmissal Type": "-",
                        "Dismissed By": "-"
                        }
        if (i.Fall_Runs):
            stages_dict = {
                "Dismissed Batsman": i.Name,
                "Team Runs": i.Fall_Runs,
                "Overs": i.Fall_Over
            }
            df_dictionary = pd.DataFrame([stages_dict])
            temp_stages = pd.concat(
                [temp_stages, df_dictionary], ignore_index=True)
        df_dictionary = pd.DataFrame([row_dict])
        temp_df = pd.concat([temp_df, df_dictionary], ignore_index=True)

    # Creating Bowling Line Up
    for i in set(bowling_lineup):
        a = [k.Name for k in i.Wickets_Taken]
        bowling_dict = {"Bowler": i.Name,
                        "Runs Conceded": i.Runs_Conceded,
                        "Wickets Taken": len(i.Wickets_Taken),
                        "Overs": str(i.Overs_Bowled)+"."+str(i.Balls_Bowled),
                        "Batsman Names": functools.reduce(
                            lambda x, y: str(
                                x)+", "+str(y), a) if len(a) else "-"
                        }

        df_dictionary = pd.DataFrame([bowling_dict])
        temp_bowling = pd.concat([temp_bowling, df_dictionary],
                                 ignore_index=True)
    # Creating Over Summary
    count = 1
    for i in inn1.Overs_Summary:
        row_summary = {"Over": count,
                       "Bowler": i[-1].Name,
                       "Runs Conceded": i[0],
                       "Wickets Taken": i[1],
                       "Total Score": i[2],
                       "Total Wickets": i[3]}
        count += 1
        df_dictionary = pd.DataFrame([row_summary])
        temp_over_summary = pd.concat([temp_over_summary, df_dictionary],
                                      ignore_index=True)

    o = inn1.Overs - 1
    b = inn1.Balls
    if (inn1.Balls == 6):
        b = 0
        o += 1
    else:
        b -= 1
    temp_stages = temp_stages.sort_values(by=['Team Runs', 'Overs'])

    print(inn1.Batting_Team, " : ", inn1.Runs,
          "/", inn1.Wickets, " in ", o, ".", b)
    print("Extras:", inn1.Extras)
    print()
    display(temp_df)
    print()
    if onlybatting == 0:
        display(temp_bowling)
        print()
        display(temp_stages)
        print()
        display(temp_over_summary)


# display_batting_table(inn1)
res_to_string = """Retired Hurt
dot ball
1 run
2 runs
3 runs
4 runs
5 runs
6 runs
Wkt Bowled
Wkt Caught
Wkt LBW
Wkt Stump
Wkt Hit Wicket
Wkt Obstructing the Field
Wkt runout non striker 0
Wkt runnout striker 0
Wkt runnout non striker 1
Wkt runnout striker 1
Wkt runnout non striker 2
Wkt runnout striker 2
Wkt runnout non striker 3
Wkt runnout striker 3
Wkt runout No ball non striker total 2 runs
Wkt runnout No ball striker total 2 runs
Wkt runnout non striker 1 Extras
Wkt runnout striker 1 Extras
Wkt runnout non striker 2 Extras
Wkt runnout striker 2 Extras
Wkt runnout non striker 3 Extras
Wkt runnout striker 3 Extras
Wkt Runnout Ball_Not_Counted non striker 1 (1 run for Wide)
Wkt Runnout Ball_Not_Counted striker 1
Wkt Runnout Ball_Not_Counted non striker 2 (1 for wide and another by running)
Wkt Runnout Ball_Not_Counted striker 2
Wkt Stump Ball_Not_Counted 1
Wkt Stump No ball 1
LB/Byes 1
LB/Byes 2
LB/Byes 3
LB/Byes 4
No ball 0
No ball 1
No ball 2
No ball 3
No ball 4
No ball 6
No ball 1 Extras
No ball 2 Extras
No ball 3 Extras
No ball 4 Extras
Wide 0
Wide 1
Wide 2
Wide 3
Wide 4
Wkt runout Noball non striker total 1 run
Wkt runout Noball striker total 1 run"""
res_to_string = res_to_string.split("\n")


class Batsman:
    def __init__(self, Name):
        self.Name = Name
        self.Runs = 0
        self.Balls = 0
        self.Dismissal = None
        self.Dismissal_By = None
        self.Fall_Over = None
        self.Fall_Runs = None
        self.Fours_Hit = 0
        self.Sixes_Hit = 0
        self.Entered_Match = 0

    def update(self, runs=0, balls=1, dismissal=None,
               dismissal_by=None, fall_over=None,
               fall_runs=None, fours_hit=0, sixes_hit=0):
        self.Runs += runs
        self.Balls += balls
        self.Dismissal = dismissal
        self.Dismissal_By = dismissal_by.Name if dismissal_by else None
        self.Fall_Over = fall_over
        self.Fall_Runs = fall_runs
        self.Fours_Hit += fours_hit
        self.Sixes_Hit += sixes_hit


class Bowler:
    def __init__(self, Name):
        self.Name = Name
        self.Runs_Conceded = 0
        self.Balls_Bowled = 0
        self.Overs_Bowled = 0
        self.Wickets_Taken = []

    def update(self, runs, balls=1, wicket=None):
        self.Runs_Conceded += runs
        if self.Balls_Bowled == 5 and balls == 1:
            self.Balls_Bowled = 0
            self.Overs_Bowled += 1
        else:
            self.Balls_Bowled += balls
        if wicket is not None:
            self.Wickets_Taken.append(wicket)


class Innings:
    def __init__(self, Batting, Bowling, toss, venue, innings, df, target=0):
        self.df = df
        self.innings = innings
        self.Toss = toss
        self.Venue = venue
        self.Batting_lineup = self.init_batsman(Batting[1:])
        self.Bowling_lineup = self.init_bowlers(Bowling[1:])
        self.Batting_Team = Batting[0]
        self.Bowling_Team = Bowling[0]
        self.Runs = 0
        self.Wickets = 0
        self.Overs = 1
        self.Balls = 1
        self.Free_Hit = 0
        self.Extras = 0
        self.Striker = self.Batting_lineup[0]
        self.Non_Striker = self.Batting_lineup[1]
        self.Striker.Entered_Match = 1
        self.Non_Striker.Entered_Match = 1
        self.Bowler = self.Bowling_lineup[0]
        self.Target = target
        if self.innings == 1:
            self.Target = 0
        self.Overs_Summary = []

    def init_batsman(self, batting):
        return [Batsman(x) for x in batting]

    def init_bowlers(self, bowlers):
        bowler_name_to_pointer = {}
        for x in set(bowlers):
            bowler_name_to_pointer[x] = Bowler(x)
        return [bowler_name_to_pointer[x] for x in bowlers]

    def get_next_batsman(self):
        if self.Wickets < 10:
            self.Batting_lineup[self.Wickets+1].Entered_Match = 1
            return self.Batting_lineup[self.Wickets+1]
        return None

    def get_next_bowler(self):
        if self.Overs == 20 and self.Balls == 6:
            return self.Bowler
        return self.Bowling_lineup[(self.Overs-1) % len(self.Bowling_lineup)]

    def get_next_ball(self):
        if self.Balls == 6:
            self.swap_batsman()
            self.Overs += 1
            self.Balls = 1
            if(self.Overs == 2):
                self.Overs_Summary.append(
                    [self.Runs, self.Wickets, self.Runs, self.Wickets,
                     self.Bowler])
            else:
                self.Overs_Summary.append([self.Runs-self.Overs_Summary[-1][2],
                                           self.Wickets -
                                           self.Overs_Summary[-1][3],
                                           self.Runs, self.Wickets, self.Bowler
                                           ])
            self.Bowler = self.get_next_bowler()
        else:
            self.Balls += 1

    def swap_batsman(self):
        self.Striker, self.Non_Striker = self.Non_Striker, self.Striker

    def get_new_row(self, score, wickets, overs, balls, free_hit, toss,
                    venue, batting_team, bowling_team, striker, striker_runs,
                    striker_balls, non_striker,
                    non_striker_runs, non_striker_balls, bowler,
                    bowler_runs, bowler_overs, bowler_balls, bowler_wickets):
        n_row = {x: 0 for x in self.df.columns}
        if 'Current_Score' in self.df.columns:
            n_row['Current_Score'] = score
        if 'Wickets' in self.df.columns:
            n_row['Wickets'] = wickets
        if 'Overs' in self.df.columns:
            n_row['Overs'] = overs
        if 'Balls' in self.df.columns:
            n_row['Balls'] = balls
        if 'Free_Hit' in self.df.columns:
            n_row['Free_Hit'] = free_hit
        if 'Striker_Runs' in self.df.columns:
            n_row['Striker_Runs'] = striker_runs
        if 'Striker_Balls' in self.df.columns:
            n_row['Striker_Balls'] = striker_balls
        if 'Non_Striker_Runs' in self.df.columns:
            n_row['Non_Striker_Runs'] = non_striker_runs
        if 'Non_Striker_Balls' in self.df.columns:
            n_row['Non_Striker_Balls'] = non_striker_balls
        if 'Bowler_Runs' in self.df.columns:
            n_row['Bowler_Runs'] = bowler_runs
        if 'Bowler_Overs' in self.df.columns:
            n_row['Bowler_Overs'] = bowler_overs
        if 'Bowler_Balls' in self.df.columns:
            n_row['Bowler_Balls'] = bowler_balls
        if 'Bowler_Wickets' in self.df.columns:
            n_row['Bowler_Wickets'] = bowler_wickets
        if 'Toss_'+toss in self.df.columns:
            n_row['Toss_'+toss] = 1
        if 'Venue_'+venue in self.df.columns:
            n_row['Venue_'+venue] = 1
        if 'Batting_Team_'+batting_team in self.df.columns:
            n_row['Batting_Team_'+batting_team] = 1
        if 'Bowling_Team_'+bowling_team in self.df.columns:
            n_row['Bowling_Team_'+bowling_team] = 1
        if 'Striker_'+striker in self.df.columns:
            n_row['Striker_'+striker] = 1
        if 'Non_Striker_'+non_striker in self.df.columns:
            n_row['Non_Striker_'+non_striker] = 1
        if 'Bowler_'+bowler in self.df.columns:
            n_row['Bowler_'+bowler] = 1

        if self.Target > 0:
            n_row['Target'] = self.Target

        progress_row = {}
        progress_row["score"] = score
        progress_row["wickets"] = wickets
        progress_row["overs"] = overs
        progress_row["balls"] = balls
        progress_row["free_hit"] = free_hit
        progress_row["striker"] = striker
        progress_row["striker_runs"] = striker_runs
        progress_row["striker_balls"] = striker_balls
        progress_row["non_striker"] = non_striker
        progress_row["non_striker_runs"] = non_striker_runs
        progress_row["non_striker_balls"] = non_striker_balls
        progress_row["bowler"] = bowler
        progress_row["bowler_runs"] = bowler_runs
        progress_row["bowler_overs"] = bowler_overs
        progress_row["bowler_balls"] = bowler_balls
        progress_row["bowler_wickets"] = bowler_wickets
        return n_row, progress_row

    def simulate_inning(self, model):
        innings_progress_dic = {}
        progress_row_count = 0
        free_hit_not_possible = [8, 9, 10, 12]
        while(True):
            if self.innings == 1:
                if (self.Overs > 20 or self.Wickets == 10):
                    self.inn_progress_df = pd.DataFrame.from_dict(
                        innings_progress_dic, orient='index')
                    return self.Runs+1
            elif self.innings == 2:
                if (self.Overs > 20 or
                        self.Wickets == 10 or self.Runs >= self.Target):
                    self.inn_progress_df = pd.DataFrame.from_dict(
                        innings_progress_dic, orient='index')
                    if self.Runs >= self.Target:
                        return (self.Batting_Team + " won by "
                                + str(10-self.Wickets)+" Wickets", 1)
                    else:
                        return (self.Bowling_Team + " won by "
                                + str(self.Target-self.Runs-1)+" Runs", 0)

            curr_row, progress_dic = self.get_new_row(
                self.Runs, self.Wickets, self.Overs,
                self.Balls, self.Free_Hit, self.Toss,
                self.Venue, self.Batting_Team,
                self.Bowling_Team, self.Striker.Name,
                self.Striker.Runs, self.Striker.Balls,
                self.Non_Striker.Name,
                self.Non_Striker.Runs,
                self.Non_Striker.Balls,
                self.Bowler.Name,
                self.Bowler.Runs_Conceded,
                self.Bowler.Overs_Bowled,
                self.Bowler.Balls_Bowled,
                len(self.Bowler.Wickets_Taken))

            self.df = self.df[0:0]
            df_dictionary = pd.DataFrame([curr_row])
            self.df = pd.concat([self.df, df_dictionary], ignore_index=True)
            self.df = self.df.astype(int)
            q = model.predict(np.expand_dims(
                np.array([self.df.iloc[-1], ]), axis=0))
            q = [i for i in q[0]]
            if self.Free_Hit == 1:
                for i in free_hit_not_possible:
                    q[i] = 0

            res = random.choices(range(0, 57), weights=q, k=1)[0]
            progress_dic["result"] = res
            innings_progress_dic[progress_row_count] = progress_dic
            progress_row_count += 1
            self.ball_prediction(res)

    def ball_prediction(self, res):
        # Setting FreeHit to 0
        free_hit_continue = [30, 31, 32, 33, 34, 50, 51, 52, 53, 54]
        if (res not in free_hit_continue):
            self.Free_Hit = 0
        # Player Retired Hurt
        if (res == 0):
            self.Wickets += 1
            self.Striker.update(0, 0, "Retired Hurt", None,
                                self.Overs, self.Runs)  # Function
            self.Striker = self.get_next_batsman()  # Next Batsman
            self.Bowler.update(0, 1)
            self.get_next_ball()

        # Player Scores Runs
        elif (1 <= res <= 7):
            self.Runs += (res-1)
            self.Striker.update(res-1)
            if (res == 5):
                self.Striker.update(balls=0, fours_hit=1)
            if res == 7:
                self.Striker.update(balls=0, sixes_hit=1)

            if (res % 2 == 0):
                self.swap_batsman()
            self.Bowler.update(res-1, 1)
            self.get_next_ball()

        # Player Gets Out
        elif (8 <= res <= 13):
            self.Wickets += 1
            if (res == 8):
                self.Striker.update(
                    0, 1, "Bowled", self.Bowler, self.Overs, self.Runs)
            elif (res == 9):
                self.Striker.update(
                    0, 1, "Caught", self.Bowler, self.Overs, self.Runs)
            elif (res == 10):
                self.Striker.update(0, 1, "LBW", self.Bowler,
                                    self.Overs, self.Runs)
            elif (res == 11):
                self.Striker.update(
                    0, 1, "Stumped", self.Bowler, self.Overs, self.Runs)
            elif (res == 12):
                self.Striker.update(0, 1, "Hit Wicket",
                                    self.Bowler, self.Overs, self.Runs)
            elif (res == 13):
                self.Striker.update(
                    0, 1, "Obstructing the Field", self.Bowler,
                    self.Overs, self.Runs)
            self.Bowler.update(0, 1, self.Striker)  # Update Bowler
            self.Striker = self.get_next_batsman()
            if (res == 9 or res == 13):
                ran = random.choices([0, 1], weights=(0.3, 0.7))[0]
                if ran:
                    self.swap_batsman()
            self.get_next_ball()

        # Wide runs maybe scored by byes
        elif (50 <= res <= 54):
            thisball = res-50
            self.Extras += (res-49)
            self.Runs += (res-49)
            self.Bowler.update(res-49, 0)  # Update Bowler

            if(thisball % 2 != 0):
                self.swap_batsman()

        # No ball runs scored by byes/leg byes
        elif (46 <= res <= 49):
            thisball = res-45
            self.Extras += (res-44)
            self.Runs += (res-44)
            self.Bowler.update(1, 0)  # Update Bowler add 1 run conceded
            self.Striker.update(0, 1)  # Increment Balls Faced by 1
            self.Free_Hit = 1  # Free Hit Set

            if (thisball % 2 != 0):
                self.swap_batsman()

        # No ball but runs scored
        elif (40 <= res <= 45):
            if(res == 44):
                self.Striker.update(balls=0, fours_hit=1)
            if(res == 45):
                self.Striker.update(balls=0, sixes_hit=1)
            thisball = res-40
            self.Extras += 1
            self.Runs += (res-39)
            self.Bowler.update(res-39, 0)  # Increment Runs by res-39
            # Increment Ball by 1 and run by res-40
            self.Striker.update(res-40, 1)
            self.Free_Hit = 1  # Free Hit set

            if (thisball % 2 != 0):
                self.swap_batsman()
        # Leg byes/byes
        elif (36 <= res <= 39):
            thisball = res-35
            self.Extras += (res-35)
            self.Runs += (res-35)
            self.Bowler.update(0, 1)  # Increment Bowler by 1 ball no runs
            self.Striker.update(0, 1)  # Increment ball by 1 no runs

            if (thisball % 2 != 0):
                self.swap_batsman()

            self.get_next_ball()

        # Normal Runouts
        elif (14 <= res <= 21):
            # 0 runs non striker out
            if (res == 14):
                self.Wickets += 1
                self.Striker.update(0, 1)  # No runs, ball Increment by 1
                self.Non_Striker.update(
                    0, 0, "Run Out", None, self.Overs, self.Runs)
                self.Bowler.update(0, 1)  # Increment 1 Ball
                self.Non_Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
                self.get_next_ball()
            # 0 runs striker out
            elif (res == 15):
                self.Wickets += 1
                self.Striker.update(0, 1, "Run Out", None,
                                    self.Overs, self.Runs)
                self.Bowler.update(0, 1)  # Increment 1 Ball
                self.Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
                self.get_next_ball()
            # 1 run non striker out
            elif (res == 16):
                self.Runs += 1
                self.Wickets += 1
                self.Striker.update(1, 1)  # 1 runs, ball Increment by 1
                self.Non_Striker.update(
                    0, 0, "Run Out", None, self.Overs, self.Runs)
                self.Bowler.update(1, 1)  # Increment 1 Ball and 1 run
                self.Non_Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
                self.get_next_ball()

            # 1 run striker out
            elif (res == 17):
                self.Runs += 1
                self.Wickets += 1
                self.Striker.update(1, 1, "Run Out", None,
                                    self.Overs, self.Runs)
                self.Bowler.update(1, 1)  # Increment 1 Ball 1 run
                self.Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
                self.get_next_ball()

            # 2 run non striker out
            elif (res == 18):
                self.Runs += 2
                self.Wickets += 1
                self.Striker.update(2, 1)  # 2 runs, ball Increment by 1
                self.Non_Striker.update(
                    0, 0, "Run Out", None, self.Overs, self.Runs)
                self.Bowler.update(2, 1)  # Increment 1 Ball and 2 run
                self.Non_Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
                self.get_next_ball()

            # 2 run striker out
            elif (res == 19):
                self.Runs += 2
                self.Wickets += 1
                self.Striker.update(2, 1, "Run Out", None,
                                    self.Overs, self.Runs)
                self.Bowler.update(2, 1)  # Increment 1 Ball 2 runs
                self.Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
                self.get_next_ball()

            # 3 run non striker out
            elif (res == 20):
                self.Runs += 3
                self.Wickets += 1
                self.Striker.update(3, 1)  # 3 runs, ball Increment by 1
                self.Non_Striker.update(
                    0, 0, "Run Out", None, self.Overs, self.Runs)
                self.Bowler.update(3, 1)  # Increment 1 Ball and 3 run
                self.Non_Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
                self.get_next_ball()

            # 2 run striker out
            elif (res == 21):
                self.Runs += 3
                self.Wickets += 1
                self.Striker.update(3, 1, "Run Out", None,
                                    self.Overs, self.Runs)
                self.Bowler.update(3, 1)  # Increment 1 Ball 3 runs
                self.Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
                self.get_next_ball()

        elif (34 <= res <= 35):
            if (res == 34):
                self.Runs += 1
                self.Wickets += 1
                self.Striker.update(
                    0, 0, "Stumped", self.Bowler, self.Overs, self.Runs)
                # Increment wicket by 1 and runs by 1
                self.Bowler.update(1, 0, self.Striker)
                self.Striker = self.get_next_batsman()

            if (res == 35):
                self.Runs += 1
                self.Wickets += 1
                self.Striker.update(
                    0, 1, "Stumped", self.Bowler, self.Overs, self.Runs)
                # Increment wicket by 1 and runs by 1
                self.Bowler.update(1, 0, self.Striker)
                self.Free_Hit = 1  # Set free hit
                self.Striker = self.get_next_batsman()

        elif (24 <= res <= 29):

            # Leg bye non striker runnout 1
            if (res == 24):
                self.Extras += 1
                self.Runs += 1
                self.Wickets += 1
                self.Striker.update(0, 1)  # 0 runs, ball Increment by 1
                self.Non_Striker.update(
                    0, 0, "Run Out", None, self.Overs, self.Runs)
                self.Bowler.update(0, 1)  # Increment 1 Ball and 0 run
                self.Non_Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
                self.get_next_ball()

            # Byes 1 run striker out
            elif (res == 25):
                self.Extras += 1
                self.Runs += 1
                self.Wickets += 1
                self.Striker.update(0, 1, "Run Out", None,
                                    self.Overs, self.Runs)
                self.Bowler.update(0, 1)  # Increment 1 Ball 0 run
                self.Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
                self.get_next_ball()

            # Leg bye 2 run non striker out
            elif (res == 26):
                self.Extras += 2
                self.Runs += 2
                self.Wickets += 1
                self.Striker.update(0, 1)  # 0 runs, ball Increment by 1
                self.Non_Striker.update(
                    0, 0, "Run Out", None, self.Overs, self.Runs)
                self.Bowler.update(0, 1)  # Increment 1 Ball and 0 run
                self.Non_Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
                self.get_next_ball()

            # Leg bye 2 run striker out
            elif (res == 27):
                self.Extras += 2
                self.Runs += 2
                self.Wickets += 1
                self.Striker.update(0, 1, "Run Out", None,
                                    self.Overs, self.Runs)
                self.Bowler.update(0, 1)  # Increment 1 Ball 0 runs
                self.Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
                self.get_next_ball()

            # Leg bye 3 run non striker out
            elif (res == 28):
                self.Extras += 3
                self.Runs += 3
                self.Wickets += 1
                self.Striker.update(0, 1)  # 0 runs, ball Increment by 1
                self.Non_Striker.update(
                    0, 0, "Run Out", None, self.Overs, self.Runs)
                self.Bowler.update(0, 1)  # Increment 1 Ball and 0 run
                self.Non_Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
                self.get_next_ball()

            # Leg bye 3 run striker out
            elif (res == 29):
                self.Extras += 3
                self.Runs += 3
                self.Wickets += 1
                self.Striker.update(0, 1, "Run Out", None,
                                    self.Overs, self.Runs)
                self.Bowler.update(0, 1)  # Increment 1 Ball 0 runs
                self.Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
                self.get_next_ball()

        elif (30 <= res <= 33):
            # Wide, 0 actual runs runout non striker
            if (res == 30):
                self.Extras += 1
                self.Runs += 1
                self.Wickets += 1
                self.Striker.update(0, 0)  # 0 runs, ball Increment by 0
                self.Non_Striker.update(
                    0, 0, "Run Out", None, self.Overs, self.Runs)
                self.Bowler.update(1, 0)  # Increment 0 Ball and 1 run
                self.Non_Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()

            # Wide, 0 run striker out
            elif (res == 31):
                self.Extras += 1
                self.Runs += 1
                self.Wickets += 1
                self.Striker.update(0, 0, "Run Out", None,
                                    self.Overs, self.Runs)
                self.Bowler.update(1, 0)  # Increment 0 Ball 1 runs
                self.Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()

            # Wide, 1 run non striker out
            elif (res == 32):
                self.Extras += 2
                self.Runs += 2
                self.Wickets += 1
                self.Striker.update(0, 0)  # 0 runs, ball Increment by 0
                self.Non_Striker.update(
                    0, 0, "Run Out", None, self.Overs, self.Runs)
                self.Bowler.update(2, 0)  # Increment 0 Ball and 2 run
                self.Non_Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()

            # Wide, 1 run striker out
            elif (res == 33):
                self.Extras += 2
                self.Runs += 2
                self.Wickets += 1
                self.Striker.update(0, 0, "Run Out", None,
                                    self.Overs, self.Runs)
                self.Bowler.update(2, 0)  # Increment 0 Ball 2 runs
                self.Striker = self.get_next_batsman()

                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()

        # Noball runout
        elif (22 <= res <= 23 or 55 <= res <= 56):
            # No ball, 0 actual runs runout non striker
            if (res == 55):
                self.Extras += 1
                self.Runs += 1
                self.Wickets += 1
                self.Striker.update(0, 1)  # 0 runs, ball Increment by 1
                self.Non_Striker.update(
                    0, 0, "Run Out", None, self.Overs, self.Runs)
                self.Bowler.update(1, 0)  # Increment 0 Ball and 1 run
                self.Non_Striker = self.get_next_batsman()
                self.Free_Hit = 1  # Set free hit
                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()

            # No ball, 0 run striker out
            elif (res == 56):
                self.Extras += 1
                self.Runs += 1
                self.Wickets += 1
                self.Striker.update(0, 1, "Run Out", None,
                                    self.Overs, self.Runs)
                self.Bowler.update(1, 0)  # Increment 0 Ball 1 runs
                self.Striker = self.get_next_batsman()
                self.Free_Hit = 1  # Set free hit
                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()
            # Noball, 1 run non striker out
            elif (res == 22):
                self.Extras += 1
                self.Runs += 2
                self.Wickets += 1
                self.Striker.update(1, 1)  # 1 runs, ball Increment by 1
                self.Non_Striker.update(
                    0, 0, "Run Out", None, self.Overs, self.Runs)
                self.Bowler.update(2, 0)  # Increment 0 Ball and 2 run
                self.Non_Striker = self.get_next_batsman()
                self.Free_Hit = 1  # Set free hit
                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()

            # Noball, 1 run striker out
            elif (res == 23):
                self.Extras += 1
                self.Runs += 2
                self.Wickets += 1
                self.Striker.update(1, 1, "Run Out", None,
                                    self.Overs, self.Runs)
                self.Bowler.update(2, 0)  # Increment 0 Ball 2 runs
                self.Striker = self.get_next_batsman()
                self.Free_Hit = 1  # Set free hit
                ran = random.choice([0, 1])
                if (ran):
                    self.swap_batsman()


class Match:
    def __init__(self, TeamA, TeamB, Venue, model_inn_1,
                 model_inn_2, Display=0, Result=1):
        self.inn1 = 0
        self.inn2 = 0
        model_inn_1.reset_states()
        model_inn_2.reset_states()
        self.TeamA = TeamA
        self.TeamB = TeamB
        self.Winner = ""
        inn1_df = pd.DataFrame(columns=BF_Cols)
        inn2_df = pd.DataFrame(columns=BS_Cols)
        toss = random.choice([0, 1])
        if(toss):
            choice = random.choice([0, 1])
            if (choice):
                self.inn1 = Innings(
                    TeamA[0], TeamB[1], TeamA[0][0], Venue, 1, inn1_df)
                target = int(self.inn1.simulate_inning(model_inn_1))
                self.inn2 = Innings(
                    TeamB[0], TeamA[1], TeamA[0][0], Venue, 2, inn2_df, target)
                result, num = self.inn2.simulate_inning(model_inn_2)
                if Display:
                    print(TeamA[0][0]+" won the toss and chose to ", end='')
                    print("Bat first")
                    display_batting_table(self.inn1)
                    display_batting_table(self.inn2)
                    print(result)
                elif Result:
                    print(result)
                if num:
                    self.Winner = TeamB[0][0]
                self.Winner = TeamA[0][0]
            else:
                self.inn1 = Innings(
                    TeamB[0], TeamA[1], TeamA[0][0], Venue, 1, inn1_df)
                target = int(self.inn1.simulate_inning(model_inn_1))
                self.inn2 = Innings(
                    TeamA[0], TeamB[1], TeamA[0][0], Venue, 2, inn2_df, target)
                result, num = self.inn2.simulate_inning(model_inn_2)
                if Display:
                    print(TeamA[0][0]+" won the toss and chose to ", end='')
                    print("Bowl first")
                    display_batting_table(self.inn1)
                    display_batting_table(self.inn2)
                    print(result)
                elif Result:
                    print(result)
                if num:
                    self.Winner = TeamA[0][0]
                self.Winner = TeamB[0][0]
        else:
            choice = random.choice([0, 1])
            if(choice):
                self.inn1 = Innings(
                    TeamB[0], TeamA[1], TeamB[0][0], Venue, 1, inn1_df)
                target = int(self.inn1.simulate_inning(model_inn_1))
                self.inn2 = Innings(
                    TeamA[0], TeamB[1], TeamB[0][0], Venue, 2, inn2_df, target)
                result, num = self.inn2.simulate_inning(model_inn_2)
                if Display:
                    print(TeamB[0][0]+" won the toss and chose to ", end='')
                    print("Bat first")
                    display_batting_table(self.inn1)
                    display_batting_table(self.inn2)
                    print(result)
                elif Result:
                    print(result)
                if num:
                    self.Winner = TeamA[0][0]
                self.Winner = TeamB[0][0]

            else:
                self.inn1 = Innings(
                    TeamA[0], TeamB[1], TeamB[0][0], Venue, 1, inn1_df)
                target = int(self.inn1.simulate_inning(model_inn_1))
                self.inn2 = Innings(
                    TeamB[0], TeamA[1], TeamB[0][0], Venue, 2, inn2_df, target)
                result, num = self.inn2.simulate_inning(model_inn_2)
                if Display:
                    print(TeamB[0][0]+" won the toss and chose to ", end='')
                    print("Bowl first")
                    display_batting_table(self.inn1)
                    display_batting_table(self.inn2)
                    print(result)
                elif Result:
                    print(result)
                if num:
                    self.Winner = TeamB[0][0]
                self.Winner = TeamA[0][0]
