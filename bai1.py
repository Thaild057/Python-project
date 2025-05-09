import time
import pandas as pd
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from io import StringIO
import os
import uuid

# Base directory where results.csv should be saved
base_dir = r"C:\Users\luong\Desktop\baitaplon trr"

# Ensure base directory exists
os.makedirs(base_dir, exist_ok=True)

# Function to convert age to decimal format
def convert_age_to_decimal(age_str):
    try:
        if pd.isna(age_str) or age_str == "N/A":
            return "N/A"
        age_str = str(age_str).strip()
        if "-" in age_str:
            years, days = map(int, age_str.split("-"))
            decimal_age = years + (days / 365)
            return round(decimal_age, 2)
        if "." in age_str:
            return round(float(age_str), 2)
        if age_str.isdigit():
            return round(float(age_str), 2)
        return "N/A"
    except (ValueError, AttributeError):
        return "N/A"

# Function to extract country code from "Nation" column
def extract_country_code(nation_str):
    try:
        if pd.isna(nation_str) or nation_str == "N/A":
            return "N/A"
        return nation_str.split()[-1]
    except (AttributeError, IndexError):
        return "N/A"

# Function to clean player names
def clean_player_name(name):
    try:
        if pd.isna(name) or name == "N/A":
            return "N/A"
        if "," in name:
            parts = [part.strip() for part in name.split(",")]
            if len(parts) >= 2:
                return " ".join(parts[::-1])
        return " ".join(name.split()).strip()
    except (AttributeError, TypeError):
        return "N/A"

# Set up Selenium WebDriver
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Define URLs and table IDs
urls = [
    "https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/keepers/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/shooting/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/passing/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/gca/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/defense/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/possession/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/misc/2024-2025-Premier-League-Stats",
]

table_ids = [
    "stats_standard",
    "stats_keeper",
    "stats_shooting",
    "stats_passing",
    "stats_gca",
    "stats_defense",
    "stats_possession",
    "stats_misc",
]

# Define the required columns in the exact order
required_columns = [
    "Player", "Nation", "Team", "Position", "Age",
    "Matches Played", "Starts", "Minutes",
    "Gls", "Ast", "crdY", "crdR",
    "xG", "xAG",
    "PrgC", "PrgP", "PrgR",
    "Gls per 90", "Ast per 90", "xG per 90", "xAG per 90",
    "GA90", "Save%", "CS%", "PK Save%",
    "SoT%", "SoT per 90", "G per Sh", "Dist",
    "Cmp", "Cmp%", "TotDist", "ShortCmp%", "MedCmp%", "LongCmp%", "KP", "Pass into 1_3", "PPA", "CrsPA",
    "SCA", "SCA90", "GCA", "GCA90",
    "Tkl", "TklW",
    "Deff Att", "Lost",
    "Blocks", "Sh", "Pass", "Int",
    "Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen",
    "Take-Ons Att", "Succ%", "Tkld%",
    "Carries", "ProDist", "Carries 1_3", "CPA", "Mis", "Dis",
    "Rec", "Rec PrgR",
    "Fls", "Fld", "Off", "Crs", "Recov",
    "Aerl Won", "Aerl Lost", "Aerl Won%"
]

# Define column renaming dictionaries for each table
column_rename_dict = {
    "stats_standard": {
        "Unnamed: 1": "Player",
        "Unnamed: 2": "Nation",
        "Unnamed: 3": "Position",
        "Unnamed: 4": "Team",
        "Unnamed: 5": "Age",
        "Playing Time": "Matches Played",
        "Playing Time.1": "Starts",
        "Playing Time.2": "Minutes",
        "Performance": "Gls",
        "Performance.1": "Ast",
        "Performance.6": "crdY",
        "Performance.7": "crdR",
        "Expected": "xG",
        "Expected.2": "xAG",
        "Progression": "PrgC",
        "Progression.1": "PrgP",
        "Progression.2": "PrgR",
        "Per 90 Minutes": "Gls per 90",
        "Per 90 Minutes.1": "Ast per 90",
        "Per 90 Minutes.5": "xG per 90",
        "Per 90 Minutes.6": "xAG per 90"
    },
    "stats_keeper": {
        "Unnamed: 1": "Player",
        "Performance.1": "GA90",
        "Performance.4": "Save%",
        "Performance.9": "CS%",
        "Penalty Kicks.4": "PK Save%"
    },
    "stats_shooting": {
        "Unnamed: 1": "Player",
        "Standard.3": "SoT%",
        "Standard.5": "SoT per 90",
        "Standard.6": "G per Sh",
        "Standard.8": "Dist"
    },
    "stats_passing": {
        "Unnamed: 1": "Player",
        "Total": "Cmp",
        "Total.2": "Cmp%",
        "Total.3": "TotDist",
        "Short.2": "ShortCmp%",
        "Medium.2": "MedCmp%",
        "Long.2": "LongCmp%",
        "Unnamed: 26": "KP",
        "Unnamed: 27": "Pass into 1_3",
        "Unnamed: 28": "PPA",
        "Unnamed: 29": "CrsPA",
    },
    "stats_gca": {
        "Unnamed: 1": "Player",
        "SCA.1": "SCA90",
        "GCA.1": "GCA90",
    },
    "stats_defense": {
        "Unnamed: 1": "Player",
        "Tackles": "Tkl", "Tackles.1": "TklW",
        "Challenges.1": "Deff Att",
        "Challenges.3": "Lost",
        "Blocks": "Blocks",
        "Blocks.1": "Sh",
        "Blocks.2": "Pass",
        "Unnamed: 20": "Int",
    },
    "stats_possession": {
        "Unnamed: 1": "Player",
        "Touches": "Touches",
        "Touches.1": "Def Pen",
        "Touches.2": "Def 3rd",
        "Touches.3": "Mid 3rd",
        "Touches.4": "Att 3rd",
        "Touches.5": "Att Pen",
        "Take-Ons": "Take-Ons Att",
        "Take-Ons.2": "Succ%",
        "Take-Ons.4": "Tkld%",
        "Carries": "Carries",
        "Carries.2": "ProDist",
        "Carries.4": "Carries 1_3",
        "Carries.5": "CPA",
        "Carries.6": "Mis",
        "Carries.7": "Dis",
        "Receiving": "Rec",
        "Receiving.1": "Rec PrgR",
    },
    "stats_misc": {
        "Unnamed: 1": "Player",
        "Performance.3": "Fls",
        "Performance.4": "Fld",
        "Performance.5": "Off",
        "Performance.6": "Crs",
        "Performance.12": "Recov",
        "Aerial Duels": "Aerl Won",
        "Aerial Duels.1": "Aerl Lost",
        "Aerial Duels.2": "Aerl Won%"
    }
}

# Initialize dictionary to store all tables
all_tables = {}

# Crawl and process each table
for url, table_id in zip(urls, table_ids):
    print(f"🔍 Processing {table_id} from {url}")
    driver.get(url)
    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    table = None
    for comment in comments:
        if table_id in comment:
            comment_soup = BeautifulSoup(comment, "html.parser")
            table = comment_soup.find("table", {"id": table_id})
            if table:
                break

    if not table:
        print(f"⚠️ Table {table_id} not found!")
        continue

    try:
        df = pd.read_html(StringIO(str(table)), header=0)[0]
    except Exception as e:
        print(f"❌ Error reading table {table_id}: {e}")
        continue

    df = df.rename(columns=column_rename_dict.get(table_id, {}))
    df = df.loc[:, ~df.columns.duplicated()]

    if "Player" in df.columns:
        df["Player"] = df["Player"].apply(clean_player_name)

    if "Age" in df.columns:
        df["Age"] = df["Age"].apply(convert_age_to_decimal)

    all_tables[table_id] = df

# Merge all dataframes on Player column
merged_df = None

for table_id, df in all_tables.items():
    df = df[[col for col in df.columns if col in required_columns]]
    df = df.drop_duplicates(subset=["Player"], keep="first")

    if merged_df is None:
        merged_df = df
    else:
        try:
            merged_df = pd.merge(merged_df, df, on="Player", how="outer", validate="1:1")
        except Exception as e:
            print(f"❌ Merge error for {table_id}: {e}")
            continue

# Reorder columns based on required_columns
merged_df = merged_df.loc[:, [col for col in required_columns if col in merged_df.columns]]

# Convert "Minutes" column to numeric, handling invalid values
merged_df["Minutes"] = pd.to_numeric(merged_df["Minutes"], errors="coerce")

# Define columns by type
int_columns = ["Matches Played", "Starts", "Minutes", "Gls", "Ast", "crdY", "crdR", "PrgC", "PrgP", "PrgR",
               "Cmp", "TotDist", "Tkl", "TklW", "Deff Att", "Lost", "Blocks", "Sh", "Pass", "Int",
               "Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen", "Take-Ons Att",
               "Carries", "Carries 1_3", "CPA", "Mis", "Dis", "Rec", "Rec PrgR",
               "Fls", "Fld", "Off", "Crs", "Recov", "Aerl Won", "Aerl Lost"]
float_columns = ["Age", "xG", "xAG", "Gls per 90", "Ast per 90", "xG per 90", "xAG per 90", "GA90", "Save%", "CS%", "PK Save%",
                 "SoT%", "SoT per 90", "G per Sh", "Dist", "Cmp%", "ShortCmp%", "MedCmp%", "LongCmp%", "KP", "Pass into 1_3", "PPA",
                 "CrsPA", "SCA", "SCA90", "GCA", "GCA90", "Succ%", "Tkld%", "ProDist", "Aerl Won%"]
string_columns = ["Player", "Nation", "Team", "Position"]

for col in int_columns:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").astype("Int64")

for col in float_columns:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").round(2)

# Filter players with more than 90 minutes played
merged_df = merged_df[merged_df["Minutes"].notna() & (merged_df["Minutes"] > 90)]

# Convert "Nation" column to country code only
if "Nation" in merged_df.columns:
    merged_df["Nation"] = merged_df["Nation"].apply(extract_country_code)

# Clean the "Player" column again after merging
if "Player" in merged_df.columns:
    merged_df["Player"] = merged_df["Player"].apply(clean_player_name)

# Fill NaN in string columns with "N/A"
for col in string_columns:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna("N/A")

# Sort players alphabetically by first name
merged_df["First Name"] = merged_df["Player"].apply(lambda x: x.split()[0] if x != "N/A" else "")
merged_df = merged_df.sort_values(by="First Name").drop(columns=["First Name"])

# Save the merged DataFrame to CSV in base_dir, preserving NaN values
result_path = os.path.join(base_dir, "result.csv")
merged_df.to_csv(result_path, index=False, encoding="utf-8-sig", na_rep="N/A")
print(f"✅ Successfully saved merged data to {result_path} with {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")

# Close the WebDriver
driver.quit()