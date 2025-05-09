from fuzzywuzzy import fuzz, process
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import os

class PlayerTransferScraper:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.csv_folder = os.path.join(root_dir, "csv")
        self.result_csv_path = os.path.join(self.csv_folder, "result.csv")
        self.filtered_csv_path = os.path.join(self.csv_folder, "players_over_900_minutes.csv")
        self.transfer_csv_path = os.path.join(self.csv_folder, "player_transfer_fee.csv")
        os.makedirs(self.csv_folder, exist_ok=True)
        self.browser_driver = None

    def setup_browser(self):
        """Cấu hình và khởi tạo trình duyệt Selenium."""
        browser_options = Options()
        browser_options.add_argument("--headless")
        browser_options.add_argument("--no-sandbox")
        browser_options.add_argument("--disable-dev-shm-usage")
        browser_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        self.browser_driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=browser_options)

    def close_browser(self):
        """Đóng trình duyệt."""
        if self.browser_driver:
            self.browser_driver.quit()

    def read_csv_file(self, file_path):
        """Đọc file CSV và xử lý lỗi."""
        try:
            return pd.read_csv(file_path, na_values=["N/A"])
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return None

    def filter_players(self):
        """Lọc cầu thủ có thời gian thi đấu trên 900 phút."""
        if not os.path.exists(self.result_csv_path):
            print(f"Error: File {self.result_csv_path} does not exist.")
            return None

        data_frame = self.read_csv_file(self.result_csv_path)
        if data_frame is None:
            return None

        filtered_data_frame = data_frame[data_frame['Minutes'] > 900].copy()
        print(f"Number of players with more than 900 minutes: {len(filtered_data_frame)}")

        filtered_data_frame.to_csv(self.filtered_csv_path, index=False, encoding='utf-8-sig')
        print(f"Saved filtered players to {self.filtered_csv_path} with {filtered_data_frame.shape[0]} rows and {filtered_data_frame.shape[1]} columns.")
        return filtered_data_frame

    @staticmethod
    def truncate_name(name):
        """Rút gọn tên cầu thủ."""
        parts = name.strip().split()
        return " ".join(parts[:2]) if len(parts) >= 2 else name

    def prepare_player_data(self):
        """Chuẩn bị danh sách cầu thủ và thời gian thi đấu."""
        players_data_frame = self.read_csv_file(self.filtered_csv_path)
        if players_data_frame is None:
            return None, None

        short_player_names = [self.truncate_name(name) for name in players_data_frame['Player'].str.strip()]
        minutes_by_player = dict(zip(players_data_frame['Player'].str.strip(), players_data_frame['Minutes']))
        return short_player_names, minutes_by_player

    def scrape_transfer_data(self, short_player_names):
        """Thu thập dữ liệu chuyển nhượng từ website."""
        transfer_base_url = "https://www.footballtransfers.com/us/transfers/confirmed/2024-2025/uk-premier-league/"
        transfer_urls = [f"{transfer_base_url}{i}" for i in range(1, 15)]
        transfer_data = []

        self.setup_browser()
        try:
            for url in transfer_urls:
                print(f"Scraping: {url}")
                self.browser_driver.get(url)
                try:
                    transfer_table = WebDriverWait(self.browser_driver, 20).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "transfer-table"))
                    )
                    table_rows = transfer_table.find_elements(By.TAG_NAME, "tr")
                    print(f"Found {len(table_rows)} rows in table at {url}")
                    for row in table_rows:
                        table_columns = row.find_elements(By.TAG_NAME, "td")
                        if table_columns and len(table_columns) >= 2:
                            full_player_name = table_columns[0].text.strip().split("\n")[0].strip()
                            short_player_name = self.truncate_name(full_player_name)
                            transfer_fee = table_columns[-1].text.strip() if len(table_columns) >= 3 else "N/A"
                            print(f"Processing player: {full_player_name}, Short name: {short_player_name}, Fee: {transfer_fee}")
                            top_match = process.extractOne(short_player_name, short_player_names, scorer=fuzz.token_sort_ratio)
                            if top_match and top_match[1] >= 80:
                                matched_player_name = top_match[0]
                                print(f"Matched: {full_player_name} -> {matched_player_name} (Score: {top_match[1]})")
                                transfer_data.append([full_player_name, transfer_fee])
                        else:
                            print(f"Skipping row with insufficient columns: {len(table_columns)}")
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
        finally:
            self.close_browser()

        return transfer_data

    def save_transfer_data(self, transfer_data):
        """Lưu dữ liệu chuyển nhượng vào file CSV."""
        if transfer_data:
            transfer_data_frame = pd.DataFrame(transfer_data, columns=['Player', 'Price'])
            transfer_data_frame.to_csv(self.transfer_csv_path, index=False)
            print(f"Results saved to '{self.transfer_csv_path}' with {len(transfer_data)} records")
        else:
            print("No matching players found.")

    def run(self):
        """Chạy toàn bộ quy trình."""
        filtered_data_frame = self.filter_players()
        if filtered_data_frame is None:
            return

        short_player_names, minutes_by_player = self.prepare_player_data()
        if short_player_names is None or minutes_by_player is None:
            return

        transfer_data = self.scrape_transfer_data(short_player_names)
        self.save_transfer_data(transfer_data)

# Chạy chương trình
if __name__ == "__main__":
    root_dir = r"C:\Users\luong\Desktop\baitaplon trr"
    scraper = PlayerTransferScraper(root_dir)
    scraper.run()