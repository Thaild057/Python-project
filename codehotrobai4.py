import pandas as pd
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from fuzzywuzzy import fuzz, process
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Verify By import to avoid NameError
try:
    By.TAG_NAME
except AttributeError:
    print("Error: 'By.TAG_NAME' is not available. Ensure 'selenium.webdriver.common.by' is correctly imported.")
    exit(1)

class BrowserManager:
    def __init__(self):
        self.driver = None
        self.options = Options()
        self.options.add_argument("--headless")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    def start_driver(self):
        try:
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
            return True
        except Exception as e:
            print(f"Error initializing ChromeDriver: {e}")
            return False

    def close_driver(self):
        if self.driver:
            self.driver.quit()
            print("WebDriver closed.")

    def get_page(self, url, max_attempts=3):
        for attempt in range(max_attempts):
            try:
                self.driver.get(url)
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "similar-players-table"))
                )
                return True
            except TimeoutException:
                print(f"Timeout on {url}, attempt {attempt + 1}/{max_attempts}")
                time.sleep(2)
            except Exception as e:
                print(f"Error processing {url}, attempt {attempt + 1}/{max_attempts}: {e}")
                time.sleep(2)
        print(f"Failed to process {url} after {max_attempts} attempts")
        return False

    def detect_max_pages(self, base_url):
        max_pages = 1
        try:
            self.driver.get(base_url + "1")
            pagination = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "pagination"))
            )
            page_links = pagination.find_elements(By.TAG_NAME, "a")
            for link in page_links:
                try:
                    page_num = int(link.text)
                    max_pages = max(max_pages, page_num)
                except ValueError:
                    continue
            print(f"Detected {max_pages} pages to crawl")
        except Exception as e:
            print(f"Error detecting pagination: {e}. Defaulting to 22 pages.")
            max_pages = 22
        return max_pages

class PlayerNameProcessor:
    special_cases = {
        "Manuel Ugarte Ribeiro": "Manuel Ugarte",
        "Igor Júlio": "Igor",
        "Igor Thiago": "Thiago",
        "Felipe Morato": "Morato",
        "Nathan Wood-Gordon": "Nathan Wood",
        "Bobby Reid": "Bobby Cordova-Reid",
        "J. Philogene": "Jaden Philogene Bidace"
    }

    @staticmethod
    def shorten_name(name):
        if name in PlayerNameProcessor.special_cases:
            return PlayerNameProcessor.special_cases[name]
        parts = name.strip().split()
        return f"{parts[0]} {parts[-1]}" if len(parts) >= 3 else name

    @staticmethod
    def match_player_name(shortened_name, player_names, min_score=70):
        if not player_names:
            return None, None
        best_match = process.extractOne(shortened_name, player_names, scorer=fuzz.token_sort_ratio)
        if best_match and best_match[1] >= min_score:
            return best_match[0], best_match[1]
        return None, best_match[1] if best_match else None

class FileHandler:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.csv_dir = os.path.join(base_dir, "csv")
        self.result_path = os.path.join(self.csv_dir, "result.csv")
        self.output_path = os.path.join(self.csv_dir, "all_estimate_transfer_fee.csv")
        os.makedirs(self.csv_dir, exist_ok=True)

    def load_player_data(self):
        player_positions = {}
        player_original_names = {}
        player_names = []
        if os.path.exists(self.result_path):
            try:
                df_players = pd.read_csv(self.result_path)
                player_positions = dict(zip(df_players['Player'].str.strip().apply(PlayerNameProcessor.shorten_name), df_players['Position']))
                player_original_names = dict(zip(df_players['Player'].str.strip().apply(PlayerNameProcessor.shorten_name), df_players['Player'].str.strip()))
                player_names = list(player_positions.keys())
                print(f"Loaded {len(player_names)} players from result.csv")
            except Exception as e:
                print(f"Error reading result.csv: {e}")
                print("Proceeding without player matching...")
        else:
            print("result.csv not found. Crawling all players without matching.")
        return player_positions, player_original_names, player_names

    def save_data(self, all_data, data_gk, data_df, data_mf, data_fw, all_data_unmatched, player_names):
        if all_data:
            df_all = pd.DataFrame(all_data, columns=['Player', 'Position', 'Price'])
            df_all = df_all.drop_duplicates(subset=['Player'])
            df_all.to_csv(self.output_path, index=False, encoding='utf-8-sig')
            print(f"File 'all_estimate_transfer_fee.csv' saved to: {self.output_path}")
            print(f"Total players scraped: {len(df_all)} (GK: {len(data_gk)}, DF: {len(data_df)}, MF: {len(data_mf)}, FW: {len(data_fw)})")
            if all_data_unmatched and player_names:
                print(f"Unmatched players: {len(all_data_unmatched)}")
        else:
            print("No players found. Check URL, table structure, or internet connection.")

class TransferScraper:
    def __init__(self, base_dir):
        self.file_handler = FileHandler(base_dir)
        self.browser_manager = BrowserManager()
        self.base_url = "https://www.footballtransfers.com/us/players/uk-premier-league/"

    def scrape(self):
        player_positions, player_original_names, player_names = self.file_handler.load_player_data()
        if not self.browser_manager.start_driver():
            return

        try:
            max_pages = self.browser_manager.detect_max_pages(self.base_url)
            urls = [f"{self.base_url}{i}" for i in range(1, max_pages + 1)]
            data_gk, data_df, data_mf, data_fw, all_data_unmatched = [], [], [], [], []

            for url in urls:
                print(f"Scraping: {url}")
                if not self.browser_manager.get_page(url):
                    continue

                table = self.browser_manager.driver.find_element(By.CLASS_NAME, "similar-players-table")
                rows = table.find_elements(By.TAG_NAME, "tr")

                for row in rows:
                    cols = row.find_elements(By.TAG_NAME, "td")
                    if cols and len(cols) >= 2:
                        player_name = cols[1].text.strip().split("\n")[0].strip()
                        shortened_player_name = PlayerNameProcessor.shorten_name(player_name)
                        etv = cols[-1].text.strip() if len(cols) >= 3 else "N/A"

                        if player_names:
                            matched_name, score = PlayerNameProcessor.match_player_name(shortened_player_name, player_names)
                            if matched_name:
                                original_name = player_original_names.get(matched_name, matched_name)
                                position = player_positions.get(matched_name, "Unknown")
                                print(f"Match found: {player_name} -> {original_name} (score: {score}, Position: {position})")
                                if "GK" in position:
                                    data_gk.append([original_name, position, etv])
                                elif position.startswith("DF"):
                                    data_df.append([original_name, position, etv])
                                elif position.startswith("MF"):
                                    data_mf.append([original_name, position, etv])
                                elif position.startswith("FW"):
                                    data_fw.append([original_name, position, etv])
                            else:
                                print(f"No match for: {player_name} (best match: {matched_name if matched_name else 'None'}, score: {score if score else 'N/A'})")
                                all_data_unmatched.append([player_name, "Unknown", etv])
                        else:
                            all_data_unmatched.append([player_name, "Unknown", etv])

            all_data = data_gk + data_df + data_mf + data_fw
            if not player_names:
                all_data += all_data_unmatched

            self.file_handler.save_data(all_data, data_gk, data_df, data_mf, data_fw, all_data_unmatched, player_names)

        finally:
            self.browser_manager.close_driver()

if __name__ == "__main__":
    base_dir = r"C:\Users\luong\Desktop\baitaplon trr"
    scraper = TransferScraper(base_dir)
    scraper.scrape()