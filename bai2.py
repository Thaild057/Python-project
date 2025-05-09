import pandas as pd
import matplotlib.pyplot as plt
import os

# Base directory where results are saved
base_dir = r"C:\Users\luong\Desktop\baitaplon trr"

# Load the CSV file from Bài 1
result_path = os.path.join(base_dir, "result.csv")
df = pd.read_csv(result_path, na_values=["N/A"])

# Create a copy of the DataFrame for calculations, converting NaN to 0 in numeric columns
df_calc = df.copy()

# Define columns to exclude (non-numeric)
exclude_columns = ["Player", "Nation", "Team", "Position"]

# Convert NaN to 0 in numeric columns for calculations
numeric_columns = [col for col in df_calc.columns if col not in exclude_columns]
for col in numeric_columns:
    df_calc[col] = pd.to_numeric(df_calc[col], errors="coerce").fillna(0)

# 1. Generate top_3.txt
rankings = {}
for col in numeric_columns:
    top_3_high = df_calc[["Player", "Team", col]].sort_values(by=col, ascending=False).head(3)
    top_3_high = top_3_high.rename(columns={col: "Value"})
    top_3_high["Rank"] = ["1st", "2nd", "3rd"]

    if df_calc[col].eq(0).all():
        top_3_low = df_calc[["Player", "Team", col]].sort_values(by=col, ascending=True).head(3)
    else:
        non_zero_df = df_calc[df_calc[col] > 0]
        top_3_low = non_zero_df[["Player", "Team", col]].sort_values(by=col, ascending=True).head(3)

    top_3_low = top_3_low.rename(columns={col: "Value"})
    top_3_low["Rank"] = ["1st", "2nd", "3rd"]

    rankings[col] = {
        "Highest": top_3_high,
        "Lowest": top_3_low
    }

# Save results to top_3.txt
top_3_path = os.path.join(base_dir, "top_3.txt")
with open(top_3_path, "w", encoding="utf-8") as f:
    for stat, data in rankings.items():
        f.write(f"\nStatistic: {stat}\n")
        f.write("\nTop 3 Highest:\n")
        f.write(data["Highest"][["Rank", "Player", "Team", "Value"]].to_string(index=False))
        f.write("\n\nTop 3 Lowest:\n")
        f.write(data["Lowest"][["Rank", "Player", "Team", "Value"]].to_string(index=False))
        f.write("\n" + "-" * 50 + "\n")
print(f"✅ Saved top 3 rankings to {top_3_path}")

# 2. Calculate median, mean, and standard deviation for results2.csv
rows = []
all_stats = {"": "all"}
for col in numeric_columns:
    all_stats[f"Median of {col}"] = df_calc[col].median()
    all_stats[f"Mean of {col}"] = df_calc[col].mean()
    all_stats[f"Std of {col}"] = df_calc[col].std()
rows.append(all_stats)

teams = sorted(df_calc["Team"].unique())
for team in teams:
    team_df = df_calc[df_calc["Team"] == team]
    team_stats = {"": team}
    for col in numeric_columns:
        team_stats[f"Median of {col}"] = team_df[col].median()
        team_stats[f"Mean of {col}"] = team_df[col].mean()
        team_stats[f"Std of {col}"] = team_df[col].std()
    rows.append(team_stats)

results_df = pd.DataFrame(rows)
results_df = results_df.rename(columns={"": ""})
for col in results_df.columns:
    if col != "":
        results_df[col] = results_df[col].round(2)

results2_path = os.path.join(base_dir, "results2.csv")
results_df.to_csv(results2_path, index=False, encoding="utf-8-sig")
print(f"✅ Successfully saved statistics to {results2_path} with {results_df.shape[0]} rows and {results_df.shape[1]} columns.")

# 3. Plot histograms for 3 attacking and 3 defensive statistics
selected_stats = ["Gls per 90", "xG per 90", "SCA90", "TklW", "Blocks", "Int"]
histograms_dir = os.path.join(base_dir, "histograms")
league_dir = os.path.join(histograms_dir, "league")
teams_dir = os.path.join(histograms_dir, "teams")

os.makedirs(league_dir, exist_ok=True)
os.makedirs(teams_dir, exist_ok=True)

for stat in selected_stats:
    if stat not in df_calc.columns:
        print(f"⚠️ Statistic {stat} not found in DataFrame. Skipping...")
        continue

    # League-wide histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df_calc[stat], bins=20, color="skyblue" if stat in ["Gls per 90", "xG per 90", "SCA90"] else "lightgreen", edgecolor="black")
    plt.title(f"League-Wide Distribution of {stat}")
    plt.xlabel(stat)
    plt.ylabel("Number of Players")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(league_dir, f"{stat}_league.png"), bbox_inches="tight")
    plt.close()
    print(f"📊 Saved league-wide histogram for {stat}")

    # Team-specific histograms
    for team in teams:
        team_data = df_calc[df_calc["Team"] == team]
        plt.figure(figsize=(8, 6))
        plt.hist(team_data[stat], bins=10, color="skyblue" if stat in ["Gls per 90", "xG per 90", "SCA90"] else "lightgreen", edgecolor="black", alpha=0.7)
        plt.title(f"{team} - Distribution of {stat}")
        plt.xlabel(stat)
        plt.ylabel("Number of Players")
        plt.grid(True, alpha=0.3)
        stat_filename = stat.replace(" ", "_")
        plt.savefig(os.path.join(teams_dir, f"{team}_{stat_filename}.png"), bbox_inches="tight")
        plt.close()
        print(f"📊 Saved histogram for {team} - {stat}")

print("✅ All histograms for selected attacking and defensive statistics have been generated and saved under 'histograms'.")

# 4. Identify the team with the highest mean for each statistic
team_means = df_calc.groupby("Team")[numeric_columns].mean().reset_index()

highest_teams = []
for stat in numeric_columns:
    if stat not in df_calc.columns:
        print(f"⚠️ Statistic {stat} not found in DataFrame. Skipping...")
        continue
    max_row = team_means.loc[team_means[stat].idxmax()]
    highest_teams.append({
        "Statistic": stat,
        "Team": max_row["Team"],
        "Mean Value": round(max_row[stat], 2)
    })

highest_teams_df = pd.DataFrame(highest_teams)
highest_team_stats_path = os.path.join(base_dir, "highest_team_stats.csv")
highest_teams_df.to_csv(highest_team_stats_path, index=False, encoding="utf-8-sig")
print(f"✅ Saved highest team stats to {highest_team_stats_path} with {highest_teams_df.shape[0]} rows.")

# 5. Determine the best-performing team
negative_stats = [
    "GA90", "crdY", "crdR", "Lost", "Mis", "Dis", "Fls", "Off", "Aerl Lost"
]

positive_stats_df = highest_teams_df[~highest_teams_df["Statistic"].isin(negative_stats)]
team_wins = positive_stats_df["Team"].value_counts()
best_team = team_wins.idxmax()
win_count = team_wins.max()

print(f"The best-performing team in the 2024-2025 Premier League season is: {best_team}")
print(f"They lead in {win_count} out of {len(positive_stats_df)} positive statistics.")