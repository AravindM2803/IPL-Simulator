import pandas as pd
import pickle
from Utils.helper import Innings, display_batting_table
import itertools
import random
from IPython.display import display
from Utils.sample_squads import (
    CSK_Squad, CSK_Pitch, RCB_Squad,
    RCB_Pitch, RR_Squad, RR_Pitch,
    MI_Squad, MI_Pitch, SRH_Squad,
    SRH_Pitch, DC_Squad, DC_Pitch,
    KXIP_Squad, KXIP_Pitch, KKR_Squad,
    KKR_Pitch)


with open('Data/BF_Cols.pkl', 'rb') as fp:
    BF_Cols = pickle.load(fp)
with open('Data/BS_Cols.pkl', 'rb') as fp:
    BS_Cols = pickle.load(fp)


class EvaluationMetrics():
    def __init__(self, model_inn1, model_inn2, load_path=None):
        self.bowler_stat = {}
        self.batsmen_stat = {}
        self.step = 5
        self.models = [model_inn1, model_inn2]
        self.progression_stat = {
            "runs": [[] for _ in range(int(20/self.step))],
            "wickets": [[] for _ in range(int(20/self.step))],
        }
        self.total_stat = []
        self.progress_df_list = []

        self.teams = [[CSK_Squad, CSK_Pitch],
                      [RCB_Squad, RCB_Pitch],
                      [RR_Squad, RR_Pitch],
                      [MI_Squad, MI_Pitch],
                      [SRH_Squad, SRH_Pitch],
                      [DC_Squad, DC_Pitch],
                      [KXIP_Squad, KXIP_Pitch],
                      [KKR_Squad, KKR_Pitch],
                      ]
        table_keys = ["Played", "Wins", "Losses", "Points",
                      "ByRuns", "ByBalls", "AgRuns", "AgBalls"]
        self.season_table = {
            'Chennai Super Kings': {i: 0 for i in table_keys},
            'Royal Challengers Bangalore': {i: 0 for i in table_keys},
            'Rajasthan Royals': {i: 0 for i in table_keys},
            'Mumbai Indians': {i: 0 for i in table_keys},
            'Sunrisers Hyderabad': {i: 0 for i in table_keys},
            'Delhi Capitals': {i: 0 for i in table_keys},
            'Kings XI Punjab': {i: 0 for i in table_keys},
            'Kolkata Knight Riders': {i: 0 for i in table_keys},
        }
        self.match_count = 0
        self.form_matches()

        if load_path is not None:
            self.load_object(load_path)

    def load_object(self, load_path):
        with open(load_path, "rb") as fp:
            saved_evaluator = pickle.load(fp)
        self.bowler_stat = saved_evaluator["bowler_stat"]
        self.batsmen_stat = saved_evaluator["batsmen_stat"]
        self.progression_stat = saved_evaluator["progression_stat"]
        self.total_stat = saved_evaluator["total_stat"]
        self.progress_df_list = saved_evaluator["progress_df_list"]
        self.teams = saved_evaluator["teams"]
        self.season_table = saved_evaluator["season_table"]
        self.matches = saved_evaluator["matches"]
        self.match_count = saved_evaluator["match_count"]

    def save_object(self, save_path):
        save_evaluator = {}
        save_evaluator["bowler_stat"] = self.bowler_stat
        save_evaluator["batsmen_stat"] = self.batsmen_stat
        save_evaluator["progression_stat"] = self.progression_stat
        save_evaluator["total_stat"] = self.total_stat
        save_evaluator["progress_df_list"] = self.progress_df_list
        save_evaluator["teams"] = self.teams
        save_evaluator["season_table"] = self.season_table
        save_evaluator["matches"] = self.matches
        save_evaluator["match_count"] = self.match_count
        with open(save_path, "wb") as fp:
            pickle.dump(save_evaluator, fp)

    def simulate_innings(self, batting_lineup, bowling_lineup,
                         toss_team, venue, innings=1, target=0, verbose=0):
        if innings == 1:
            inn_df = pd.DataFrame(columns=BF_Cols)
        elif innings == 2:
            inn_df = pd.DataFrame(columns=BS_Cols)
        else:
            assert False, "innings should be '1' or '2'"
        inn = Innings(batting_lineup, bowling_lineup, toss_team,
                      venue, innings, inn_df, target)
        simulation_ret = inn.simulate_inning(self.models[innings - 1])
        btl = inn.Batting_lineup
        bwl = inn.Bowling_lineup

        for i in btl:
            if (i.Entered_Match):
                batsman_dict = {
                    "Runs": i.Runs,
                    "Fours": i.Fours_Hit,
                    "Sixes": i.Sixes_Hit,
                    "Balls Faced": i.Balls,
                    "Dismissal Type": (i.Dismissal
                                       if i.Dismissal else 'Not Out'),
                    "Dismissed By": i.Dismissal_By if i.Dismissal_By else "-"
                }
                if i.Name in self.batsmen_stat:
                    self.batsmen_stat[i.Name].append(batsman_dict)
                else:
                    self.batsmen_stat[i.Name] = [batsman_dict]
        for i in set(bwl):
            bowling_dict = {"Runs Conceded": i.Runs_Conceded,
                            "Wickets Taken": len(i.Wickets_Taken),
                            "Balls": 6*(i.Overs_Bowled)+(i.Balls_Bowled)
                            }
            if i.Name in self.bowler_stat:
                self.bowler_stat[i.Name].append(bowling_dict)
            else:
                self.bowler_stat[i.Name] = [bowling_dict]

        progression_score_lis = [0 for _ in range(int(20/self.step))]
        progression_wicket_lis = [0 for _ in range(int(20/self.step))]
        count = 0
        for i in inn.Overs_Summary:
            ind = count//self.step
            progression_score_lis[ind] += i[0]
            progression_wicket_lis[ind] += i[1]
            if (count + 1)//self.step != ind:
                self.progression_stat["runs"][ind].append(
                    progression_score_lis[ind])
                self.progression_stat["wickets"][ind].append(
                    progression_wicket_lis[ind])
            count += 1
        self.total_stat.append(inn.Runs)
        self.progress_df_list.append(inn.inn_progress_df)
        if verbose:
            display_batting_table(inn, display_level=verbose-1)
        return (inn.Runs, self.get_balls(inn), simulation_ret
                if innings == 2 else None)

    def form_matches(self):
        self.match_count = 0
        combinations = list(itertools.combinations(
            [i for i in range(len(self.teams))], 2))
        self.matches = []
        for comb in combinations:
            self.matches.append(
                [[self.teams[comb[0]][0], self.teams[comb[1]][0]],
                    self.teams[comb[0]][1]]
            )
            self.matches.append(
                [[self.teams[comb[0]][0], self.teams[comb[1]][0]],
                    self.teams[comb[1]][1]],
            )
        random.shuffle(self.matches)
        for match in self.matches:
            random.shuffle(match[0])

    def display_table(self):
        display_dic = {}
        table_cols = ["Played", "Wins", "Losses", "Points"]
        for team in self.season_table:
            team_dic = self.season_table[team]
            display_dic[team] = {col: team_dic[col] for col in table_cols}
            byrr = (team_dic["ByRuns"] / team_dic["ByBalls"] * 6
                    if team_dic["ByBalls"] != 0 else 0)
            agrr = (team_dic["AgRuns"] / team_dic["AgBalls"] * 6
                    if team_dic["AgBalls"] != 0 else 0)
            display_dic[team]["NRR"] = byrr - agrr
        points_table_df = pd.DataFrame.from_dict(
            display_dic, orient='index').sort_values(by=["Points", "NRR"],
                                                     ascending=False)
        display(points_table_df)

    def reinitialize_tournament(self):
        self.form_matches()

    def simulate_match(self, verbose=0):
        if self.match_count >= len(self.matches):
            print("All Matches in the series are over")
            return
        match = self.matches[self.match_count]
        self.match_count += 1
        toss = random.choice([0, 1])
        inn1_score, inn1_balls, _ = self.simulate_innings(
            match[0][0][0], match[0][1][1],
            match[0][toss][0][0], match[1], 1, verbose=verbose)

        (inn2_score, inn2_balls,
            (ret_str, inn2_ret)) = self.simulate_innings(
            match[0][1][0], match[0][0][1],
            match[0][toss][0][0], match[1], 2,
            inn1_score+1, verbose=verbose)
        self.season_table[match[0][0][0][0]]["ByRuns"] += inn1_score
        self.season_table[match[0][0][0][0]]["ByBalls"] += inn1_balls
        self.season_table[match[0][1][0][0]]["AgRuns"] += inn1_score
        self.season_table[match[0][1][0][0]]["AgBalls"] += inn1_balls
        self.season_table[match[0][1][0][0]]["ByRuns"] += inn2_score
        self.season_table[match[0][1][0][0]]["ByBalls"] += inn2_balls
        self.season_table[match[0][0][0][0]]["AgRuns"] += inn2_score
        self.season_table[match[0][0][0][0]]["AgBalls"] += inn2_balls
        self.season_table[match[0][0][0][0]]["Played"] += 1
        self.season_table[match[0][1][0][0]]["Played"] += 1
        if inn2_ret == 1:
            self.season_table[match[0][1][0][0]]["Points"] += 2
            self.season_table[match[0][1][0][0]]["Wins"] += 1
            self.season_table[match[0][0][0][0]]["Losses"] += 1
        elif inn2_ret == 0:
            self.season_table[match[0][0][0][0]]["Points"] += 2
            self.season_table[match[0][0][0][0]]["Wins"] += 1
            self.season_table[match[0][1][0][0]]["Losses"] += 1
        elif inn2_ret == -1:
            self.season_table[match[0][0][0][0]]["Points"] += 1
            self.season_table[match[0][1][0][0]]["Points"] += 1
        if verbose:
            print(ret_str)

    def evaluate(self):
        pass

    def get_balls(self, inn):
        o = inn.Overs - 1
        b = inn.Balls
        if (inn.Balls == 6):
            b = 0
            o += 1
        else:
            b -= 1
        return o*6 + b
