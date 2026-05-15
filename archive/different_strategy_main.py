"""
Tracksuit DS Interview — Survey Allocation Optimisation
"""

import numpy as np
import pandas as pd
import pulp
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import combinations
import random

from main import NZ_AGE, NZ_GENDER # for random allocation/ selection

def load_survey_data(file_path: str) -> Dict[int, Dict[str, float]]:
    """
    Load the survey data from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing the survey data.

    Returns:
        Dict[int, Dict[str, float]]: A dictionary mapping category IDs 
        to their incidence rates and seconds per category
    """
    df_targets = pd.read_csv(file_path)
    survey_data = {}
    for _, row in df_targets.iterrows():
        category_id = int(row['category_id'])
        incidence_rate = float(row['incidence_rate'])
        seconds_per_survey = float(row['category_length_seconds'])
        survey_data[category_id] = {
            'incidence_rate': incidence_rate,
            'seconds_per_survey': seconds_per_survey
        }
    return survey_data

def generate_synthetic_respondent():
    """
    Pull from an existing pool of respondents to generate a synthetic 
    respondent with demographic distribution corresponding to NZ census data.
    Returns:
        Dict[str, str]: A dictionary containing the demographic attributes of the respondent.
    """
    # Gender: Stats NZ, 31 Dec 2025
    NZ_GENDER = {
        "Female": 0.4965,
        "Male":   0.5035,
    }

    # Age brackets: Infometrics NZ, 30 Jun 2025.
    # Two census 5-year bands straddle the bracket boundaries, so they are
    # split proportionally by the number of single years that fall in each bracket:
    #   10-14 band  →  10-13 (4 years) goes to "<14",   14 (1 year) goes to "14-17"
    #   15-19 band  →  15-17 (3 years) goes to "14-17", 18-19 (2 years) goes to "18-24"
    _NZ_AGE_COUNTS: Dict[str, int] = {
        # "<14":   297_580 + 325_440 + int(4 / 5 * 348_240),   # 0–13
        "14-17": int(1 / 5 * 348_240) + int(3 / 5 * 354_350),  # 14–17
        "18-24": int(2 / 5 * 354_350) + 327_760,               # 18–24
        "25-39": 344_080 + 401_820 + 401_290,                   # 25–39
        "40-54": 357_420 + 316_820 + 326_640,                   # 40–54
        "55-64": 313_260 + 309_870,                              # 55–64
        "65+":   272_530 + 225_340 + 185_530 + 117_040 + 65_300 + 34_400,
    }
    _total_pop = sum(_NZ_AGE_COUNTS.values())
    NZ_AGE: Dict[str, float] = {
        bracket: count / _total_pop for bracket, count in _NZ_AGE_COUNTS.items()
    }

    gender = random.choices(list(NZ_GENDER.keys()), weights=list(NZ_GENDER.values()), k=1)[0]
    age = random.choices(list(NZ_AGE.keys()), weights=list(NZ_AGE.values()), k=1)[0]

    return gender + "|" + age

# def generate_respondents_pool( generate_synthetic_respondent: function, num_respondents: int) -> List[Dict[str]]


def algorithm_greedy_allocation(survey_data: Dict[int, Dict[str, float]], respondent: Dict[str, str]) -> pd.DataFrame:
    """
    Implement a greedy/ hill climbing algorithm to optimize the survey allocation.
    This is a baseline implementation for multiple (<=3) categories in a bucket to compare
    with the LP-Optimized solution.

    Args:
        survey_data (Dict[int, Dict[str, float]]): The survey data containing category information.
        respondent (Dict[str, str]): The demographic attributes of the respondent.
    """
    

    pass
    ## decided to vibe code remainder in main.py to stay within 5-10hr time limit of project.



if __name__ == "__main__":
    file_path = "data/fake_category_data.csv"
    survey_data = load_survey_data(file_path)
    synth_resp = generate_synthetic_respondent()
    print(synth_resp)
    # print(survey_data)
