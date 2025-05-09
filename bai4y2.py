import pandas as pd
import numpy as np
import os
import re
from fuzzywuzzy import process, fuzz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

class FileManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.csv_dir = os.path.join(base_dir, "csv")
        os.makedirs(self.csv_dir, exist_ok=True)
        self.result_path = os.path.join(self.csv_dir, "result.csv")
        self.etv_path = os.path.join(self.csv_dir, "all_estimate_transfer_fee.csv")
        self.output_path = os.path.join(self.csv_dir, "ml_estimated_values_linear.csv")

    def read_csv(self, file_path):
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy tệp - {e}")
            return None

    def save_csv(self, df, file_path):
        df.to_csv(file_path, index=False)
        print(f"Giá trị ước tính của các cầu thủ đã được lưu vào '{file_path}'")

class PlayerNameMatcher:
    @staticmethod
    def simplify_name(name):
        if not isinstance(name, str):
            return ""
        parts = name.strip().split()
        return " ".join(parts[:2]) if len(parts) >= 2 else name

    @staticmethod
    def match_player_name(name, options, min_score=90):
        if not isinstance(name, str):
            return None, None
        simplified_name = PlayerNameMatcher.simplify_name(name).lower()
        simplified_options = [PlayerNameMatcher.simplify_name(opt).lower() for opt in options if isinstance(opt, str)]
        match = process.extractOne(
            simplified_name,
            simplified_options,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=min_score
        )
        if match is not None:
            matched_idx = simplified_options.index(match[0])
            return options[matched_idx], match[1]
        return None, None

class ValuationConverter:
    @staticmethod
    def convert_valuation(val_text):
        if pd.isna(val_text) or val_text in ["N/A", ""]:
            return np.nan
        try:
            val_text = re.sub(r'[€£]', '', val_text).strip().upper()
            multiplier = 1000000 if 'M' in val_text else 1000 if 'K' in val_text else 1
            value = float(re.sub(r'[MK]', '', val_text)) * multiplier
            return value
        except (ValueError, TypeError):
            return np.nan

class RoleProcessor:
    def __init__(self, role, config, file_manager, name_matcher, valuation_converter):
        self.role = role
        self.config = config
        self.file_manager = file_manager
        self.name_matcher = name_matcher
        self.valuation_converter = valuation_converter
        self.standard_output_columns = [
            'Player', 'Team', 'Nation', 'Position', 'Actual_Transfer_Value_M', 'Predicted_Transfer_Value_M'
        ]

    def load_data(self):
        stats_data = self.file_manager.read_csv(self.file_manager.result_path)
        valuation_data = self.file_manager.read_csv(self.config['data_path'])
        if stats_data is None or valuation_data is None:
            return None, None
        return stats_data, valuation_data

    def preprocess_data(self, stats_data, valuation_data):
        stats_data['Main_Role'] = stats_data['Position'].astype(str).str.split(r'[,/]').str[0].str.strip()
        stats_data = stats_data[stats_data['Main_Role'].str.upper() == self.config['role_filter'].upper()].copy()
        player_list = valuation_data['Player'].dropna().tolist()

        stats_data['Linked_Name'] = None
        stats_data['Link_Score'] = None
        stats_data['Valuation'] = np.nan

        for idx, row in stats_data.iterrows():
            linked_name, score = self.name_matcher.match_player_name(row['Player'], player_list)
            if linked_name:
                stats_data.at[idx, 'Linked_Name'] = linked_name
                stats_data.at[idx, 'Link_Score'] = score
                linked_row = valuation_data[valuation_data['Player'] == linked_name]
                if not linked_row.empty:
                    val_value = self.valuation_converter.convert_valuation(linked_row['Price'].iloc[0])
                    stats_data.at[idx, 'Valuation'] = val_value

        filtered_data = stats_data[stats_data['Linked_Name'].notna()].copy()
        filtered_data = filtered_data.drop_duplicates(subset='Linked_Name')
        unmatched_players = stats_data[stats_data['Linked_Name'].isna()]['Player'].dropna().tolist()

        if unmatched_players:
            print(f"Cầu thủ {self.role} không khớp: {len(unmatched_players)} cầu thủ không được khớp.")
            print(unmatched_players)

        return filtered_data, unmatched_players

    def prepare_features(self, filtered_data):
        attributes = self.config['attributes']
        target_col = 'Valuation'

        for col in attributes:
            if col in ['Team', 'Nation']:
                filtered_data[col] = filtered_data[col].fillna('Unknown')
            else:
                filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')
                median_val = filtered_data[col].median()
                filtered_data[col] = filtered_data[col].fillna(median_val if not pd.isna(median_val) else 0)

        numeric_attrs = [col for col in attributes if col not in ['Team', 'Nation']]
        for col in numeric_attrs:
            filtered_data[col] = np.log1p(filtered_data[col].clip(lower=0))

        for col in self.config['key_attributes']:
            if col in filtered_data.columns:
                filtered_data[col] = filtered_data[col] * 2.0
        if 'Minutes' in filtered_data.columns:
            filtered_data['Minutes'] = filtered_data['Minutes'] * 1.5
        if 'Age' in filtered_data.columns:
            filtered_data['Age'] = filtered_data['Age'] * 0.5

        return filtered_data, attributes, target_col, numeric_attrs

    def train_model(self, ml_data, attributes, target_col, numeric_attrs):
        if ml_data.empty:
            print(f"Lỗi: Không có dữ liệu Valuation hợp lệ cho {self.role}.")
            return None, None, None

        X = ml_data[attributes]
        y = ml_data[target_col]

        if len(ml_data) > 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            print(f"Cảnh báo: Không đủ dữ liệu cho {self.role} để chia tập huấn luyện/kiểm tra.")
            X_train, y_train = X, y
            X_test, y_test = X, y

        categorical_attrs = [col for col in attributes if col in ['Team', 'Nation']]
        data_transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_attrs),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_attrs)
            ])

        model_pipeline = Pipeline([
            ('transformer', data_transformer),
            ('model', LinearRegression())
        ])

        model_pipeline.fit(X_train, y_train)
        return model_pipeline, X_test, y_test

    def generate_output(self, filtered_data, model_pipeline, attributes):
        filtered_data['Estimated_Value'] = model_pipeline.predict(filtered_data[attributes])
        filtered_data['Estimated_Value'] = filtered_data['Estimated_Value'].clip(lower=100_000, upper=200_000_000)
        filtered_data['Predicted_Transfer_Value_M'] = (filtered_data['Estimated_Value'] / 1_000_000).round(2)
        filtered_data['Actual_Transfer_Value_M'] = (filtered_data['Valuation'] / 1_000_000).round(2)

        for col in self.standard_output_columns:
            if col not in filtered_data.columns:
                filtered_data[col] = np.nan if col in ['Actual_Transfer_Value_M', 'Predicted_Transfer_Value_M'] else ''

        filtered_data['Position'] = self.role
        output_data = filtered_data[self.standard_output_columns].copy()

        numeric_attrs_no_age = [col for col in attributes if col != 'Age' and col not in ['Team', 'Nation']]
        for col in numeric_attrs_no_age:
            if col in output_data.columns:
                output_data[col] = np.expm1(output_data[col]).round(2)
        if 'Age' in output_data.columns:
            output_data['Age'] = np.expm1(output_data['Age']).round(0)
            median_age = output_data['Age'].median()
            output_data['Age'] = output_data['Age'].fillna(median_age).astype(int)

        return output_data

    def process(self):
        print(f"\nĐang xử lý {self.role}...")
        stats_data, valuation_data = self.load_data()
        if stats_data is None or valuation_data is None:
            return None, []

        filtered_data, unmatched_players = self.preprocess_data(stats_data, valuation_data)
        filtered_data, attributes, target_col, numeric_attrs = self.prepare_features(filtered_data)
        ml_data = filtered_data.dropna(subset=[target_col]).copy()

        model_pipeline, X_test, y_test = self.train_model(ml_data, attributes, target_col, numeric_attrs)
        if model_pipeline is None:
            return None, unmatched_players

        output_data = self.generate_output(filtered_data, model_pipeline, attributes)
        return output_data, unmatched_players

class ValuationAnalyzer:
    def __init__(self, base_dir, roles_config):
        self.file_manager = FileManager(base_dir)
        self.name_matcher = PlayerNameMatcher()
        self.valuation_converter = ValuationConverter()
        self.roles_config = roles_config

    def run(self):
        combined_outputs = []
        unmatched_records = []

        for role, config in self.roles_config.items():
            processor = RoleProcessor(role, config, self.file_manager, self.name_matcher, self.valuation_converter)
            output, unmatched = processor.process()
            if output is not None:
                combined_outputs.append(output)
            if unmatched:
                unmatched_records.extend([(role, player) for player in unmatched])

        if combined_outputs:
            final_output = pd.concat(combined_outputs, ignore_index=True)
            final_output = final_output.sort_values(by='Predicted_Transfer_Value_M', ascending=False)
            self.file_manager.save_csv(final_output, self.file_manager.output_path)

# Role configurations (same as original)
roles_config = {
    'Goalkeeper': {
        'role_filter': 'GK',
        'data_path': r"C:\Users\luong\Desktop\baitaplon trr\csv\all_estimate_transfer_fee.csv",
        'attributes': [
            'Save%', 'CS%', 'GA90', 'Minutes', 'Age', 'PK Save%', 'Team', 'Nation'
        ],
        'key_attributes': ['Save%', 'CS%', 'PK Save%']
    },
    'Defender': {
        'role_filter': 'DF',
        'data_path': r"C:\Users\luong\Desktop\baitaplon trr\csv\all_estimate_transfer_fee.csv",
        'attributes': [
            'Tkl', 'TklW', 'Int', 'Blocks', 'Recov', 'Minutes', 'Team', 'Age', 'Nation', 'Aerl Won%',
            'Aerl Won', 'Cmp', 'Cmp%', 'PrgP', 'LongCmp%', 'Carries', 'Touches', 'Dis', 'Mis'
        ],
        'key_attributes': ['Tkl', 'TklW', 'Int', 'Blocks', 'Aerl Won%', 'Aerl Won', 'Recov']
    },
    'Midfielder': {
        'role_filter': 'MF',
        'data_path': r"C:\Users\luong\Desktop\baitaplon trr\csv\all_estimate_transfer_fee.csv",
        'attributes': [
            'Cmp%', 'KP', 'PPA', 'PrgP', 'Tkl', 'Ast', 'SCA', 'Touches', 'Minutes', 'Team', 'Age', 'Nation',
            'Pass into 1_3', 'xAG', 'Carries 1_3', 'ProDist', 'Rec', 'Mis', 'Dis'
        ],
        'key_attributes': ['KP', 'PPA', 'PrgP', 'SCA', 'xAG', 'Pass into 1_3', 'Carries 1_3']
    },
    'Forward': {
        'role_filter': 'FW',
        'data_path': r"C:\Users\luong\Desktop\baitaplon trr\csv\all_estimate_transfer_fee.csv",
        'attributes': [
            'Gls', 'Ast', 'Gls per 90', 'xG per 90', 'SoT%', 'G per Sh', 'SCA90', 'GCA90',
            'PrgC', 'Carries 1_3', 'Aerl Won%', 'Team', 'Age', 'Minutes'
        ],
        'key_attributes': ['Gls', 'Ast', 'Gls per 90', 'xG per 90', 'SCA90', 'GCA90']
    }
}

if __name__ == "__main__":
    base_dir = r"C:\Users\luong\Desktop\baitaplon trr"
    analyzer = ValuationAnalyzer(base_dir, roles_config)
    analyzer.run()