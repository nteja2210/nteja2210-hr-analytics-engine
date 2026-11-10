"""Generate a synthetic HR dataset that mirrors the IBM HR Analytics
Employee Attrition & Performance dataset (same column names).

If you have the real Kaggle file, just save it as data/hr_data.csv and this
script will not be needed.

Run:  python -m src.generate_data
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "hr_data.csv"


def generate(n: int = 1470, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(37, 9, n), 18, 60).astype(int)
    department = rng.choice(
        ["Research & Development", "Sales", "Human Resources"], n, p=[0.65, 0.30, 0.05]
    )

    roles_by_dept = {
        "Research & Development": [
            "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Research Director",
        ],
        "Sales": ["Sales Executive", "Sales Representative", "Manager"],
        "Human Resources": ["Human Resources", "Manager"],
    }
    job_role = np.array([rng.choice(roles_by_dept[d]) for d in department])

    education = rng.choice([1, 2, 3, 4, 5], n, p=[0.11, 0.19, 0.39, 0.27, 0.04])
    education_field = rng.choice(
        ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"],
        n, p=[0.41, 0.32, 0.11, 0.09, 0.05, 0.02],
    )
    gender = rng.choice(["Male", "Female"], n, p=[0.6, 0.4])
    marital = rng.choice(["Married", "Single", "Divorced"], n, p=[0.46, 0.32, 0.22])
    travel = rng.choice(
        ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], n, p=[0.71, 0.19, 0.10]
    )
    overtime = rng.choice(["No", "Yes"], n, p=[0.72, 0.28])

    total_years = np.clip(age - 18 - rng.integers(0, 8, n) + rng.integers(-2, 3, n), 0, 40)
    job_level = np.clip(
        1 + (total_years / 8).astype(int) + rng.integers(-1, 2, n), 1, 5
    )
    monthly_income = np.clip(
        2000 + job_level * 3200 + total_years * 90 + rng.normal(0, 900, n), 1009, 19999
    ).astype(int)

    years_at_company = np.clip(
        (total_years * rng.uniform(0.15, 0.8, n)).astype(int), 0, 40
    )
    years_in_role = np.clip((years_at_company * rng.uniform(0.3, 1.0, n)).astype(int), 0, 18)
    years_since_promo = np.clip((years_at_company * rng.uniform(0.0, 0.8, n)).astype(int), 0, 15)
    years_with_mgr = np.clip((years_at_company * rng.uniform(0.3, 1.0, n)).astype(int), 0, 17)

    distance = np.clip(rng.exponential(8, n) + 1, 1, 29).astype(int)
    num_companies = rng.choice(range(0, 10), n, p=[0.13, 0.35, 0.10, 0.11, 0.10, 0.05, 0.05, 0.05, 0.03, 0.03])
    stock = rng.choice([0, 1, 2, 3], n, p=[0.43, 0.40, 0.11, 0.06])
    training = rng.choice(range(0, 7), n, p=[0.04, 0.04, 0.37, 0.33, 0.09, 0.08, 0.05])
    hike = np.clip(rng.normal(15, 3.6, n), 11, 25).astype(int)
    perf = np.where(rng.random(n) < 0.85, 3, 4)

    env_sat = rng.integers(1, 5, n)
    job_sat = rng.integers(1, 5, n)
    rel_sat = rng.integers(1, 5, n)
    involvement = rng.choice([1, 2, 3, 4], n, p=[0.05, 0.25, 0.59, 0.11])
    wlb = rng.choice([1, 2, 3, 4], n, p=[0.05, 0.23, 0.61, 0.11])

    # Attrition drivers (logistic model) -> realistic ~16% attrition
    z = (
        -1.55
        + 1.35 * (overtime == "Yes")
        - 0.30 * (job_sat - 2.5)
        - 0.25 * (env_sat - 2.5)
        - 0.30 * (wlb - 2.5)
        - 0.045 * (age - 37)
        - 0.00013 * (monthly_income - 6500)
        + 0.060 * (distance - 9)
        + 0.18 * (num_companies - 2.5)
        + 0.55 * (travel == "Travel_Frequently")
        + 0.45 * (marital == "Single")
        - 0.40 * stock
        + 0.12 * years_since_promo
        - 0.08 * years_with_mgr
        + 0.30 * (job_role == "Sales Representative")
        + 0.25 * (job_role == "Laboratory Technician")
    )
    prob = 1 / (1 + np.exp(-z))
    attrition = np.where(rng.random(n) < prob, "Yes", "No")

    df = pd.DataFrame({
        "Age": age,
        "Attrition": attrition,
        "BusinessTravel": travel,
        "DailyRate": rng.integers(102, 1500, n),
        "Department": department,
        "DistanceFromHome": distance,
        "Education": education,
        "EducationField": education_field,
        "EmployeeCount": 1,
        "EmployeeNumber": np.arange(1, n + 1),
        "EnvironmentSatisfaction": env_sat,
        "Gender": gender,
        "HourlyRate": rng.integers(30, 100, n),
        "JobInvolvement": involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_sat,
        "MaritalStatus": marital,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": rng.integers(2094, 26999, n),
        "NumCompaniesWorked": num_companies,
        "Over18": "Y",
        "OverTime": overtime,
        "PercentSalaryHike": hike,
        "PerformanceRating": perf,
        "RelationshipSatisfaction": rel_sat,
        "StandardHours": 80,
        "StockOptionLevel": stock,
        "TotalWorkingYears": total_years,
        "TrainingTimesLastYear": training,
        "WorkLifeBalance": wlb,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_role,
        "YearsSinceLastPromotion": years_since_promo,
        "YearsWithCurrManager": years_with_mgr,
    })
    return df


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = generate()
    data.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(data)} rows -> {OUT_PATH}")
    print("Attrition rate:", round((data['Attrition'] == 'Yes').mean() * 100, 1), "%")
