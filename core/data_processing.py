"""
PJM Data Processing Pipeline
Handles data loading, cleaning, and feature engineering for hourly energy data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Try to import holidays library
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    print("Warning: 'holidays' library not installed. Holiday features will be disabled.")


class PJMDataProcessor:
    """
    A class to handle PJM energy data processing for ML models.
    
    Attributes:
        data_path (str): Path to the data directory
        raw_data (pd.DataFrame): Raw loaded data
        processed_data (pd.DataFrame): Data after feature engineering
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to raw CSV file. If None, uses config default.
        """
        if data_path is None:
            self.data_path = str(config.DATA_DIR / "raw" / config.RAW_DATA_FILE)
        else:
            self.data_path = data_path
            
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = []
        self.target_column = config.LOAD_COL
        
        # US Holidays
        if HOLIDAYS_AVAILABLE:
            self.us_holidays = holidays.US()
        
    def load_data(self, filename: str = None) -> pd.DataFrame:
        """
        Load PJM data from CSV file.
        
        Args:
            filename: Name of the CSV file to load. If None, uses self.data_path directly.
            
        Returns:
            DataFrame with loaded data
        """
        if filename is None:
            file_path = self.data_path
        else:
            file_path = os.path.join(os.path.dirname(self.data_path), filename)
        
        print(f"Loading data from: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Parse datetime
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Set datetime as index
        df = df.set_index('Datetime')
        
        # Sort by datetime
        df = df.sort_index()
        
        self.raw_data = df
        
        print(f"[OK] Loaded {len(df):,} records")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        print(f"   Columns: {list(df.columns)}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Clean the data: handle missing values, duplicates, outliers.
        
        Args:
            df: DataFrame to clean. If None, uses self.raw_data
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.raw_data.copy()
        else:
            df = df.copy()
            
        print("\nCleaning data...")
        
        original_len = len(df)
        
        # 1. Remove duplicates
        duplicates = df.index.duplicated(keep='first')
        if duplicates.sum() > 0:
            print(f"   Removed {duplicates.sum()} duplicate timestamps")
            df = df[~duplicates]
        
        # 2. Handle missing values
        missing = df[self.target_column].isna().sum()
        if missing > 0:
            print(f"   Found {missing} missing values")
            # Interpolate missing values (linear interpolation)
            df[self.target_column] = df[self.target_column].interpolate(method='linear')
            print(f"   Interpolated missing values")
        
        # 3. Handle outliers using IQR method
        Q1 = df[self.target_column].quantile(0.01)
        Q3 = df[self.target_column].quantile(0.99)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (df[self.target_column] < lower_bound) | (df[self.target_column] > upper_bound)
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            print(f"   Found {outlier_count} outliers ({outlier_count/len(df)*100:.2f}%)")
            # Cap outliers instead of removing
            df.loc[df[self.target_column] < lower_bound, self.target_column] = lower_bound
            df.loc[df[self.target_column] > upper_bound, self.target_column] = upper_bound
            print(f"   Capped outliers to [{lower_bound:.0f}, {upper_bound:.0f}]")
        
        # 4. Check for gaps in time series
        expected_freq = pd.Timedelta(hours=1)
        time_diff = df.index.to_series().diff()
        gaps = time_diff[time_diff > expected_freq]
        
        if len(gaps) > 0:
            print(f"   Found {len(gaps)} gaps in time series")
            # Reindex to fill gaps
            full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
            df = df.reindex(full_idx)
            df[self.target_column] = df[self.target_column].interpolate(method='linear')
            print(f"   Filled gaps with interpolation")
        
        print(f"[OK] Cleaning complete: {original_len:,} -> {len(df):,} records")
        
        return df
    
    def create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create calendar-based features.
        
        Args:
            df: DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with calendar features added
        """
        print("\nCreating calendar features...")
        
        df = df.copy()
        
        # Basic time features
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek  # Monday=0, Sunday=6
        df['dayofmonth'] = df.index.day
        df['dayofyear'] = df.index.dayofyear
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['weekofyear'] = df.index.isocalendar().week.astype(int)
        
        # Binary features
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        
        # Time of day categories
        df['hour_category'] = pd.cut(
            df['hour'], 
            bins=[-1, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        
        # Peak hours (typically 6 AM - 10 PM for electricity)
        df['is_peak_hour'] = ((df['hour'] >= 6) & (df['hour'] <= 22)).astype(int)
        
        # Business hours (9 AM - 5 PM on weekdays)
        df['is_business_hour'] = (
            (df['hour'] >= 9) & 
            (df['hour'] <= 17) & 
            (df['dayofweek'] < 5)
        ).astype(int)
        
        print(f"   Created {len([c for c in df.columns if c not in [self.target_column]])} calendar features")
        
        return df
    
    def create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create holiday-related features.
        
        Args:
            df: DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with holiday features added
        """
        if not HOLIDAYS_AVAILABLE:
            print("[WARN] Skipping holiday features (holidays library not installed)")
            df['is_holiday'] = 0
            df['is_holiday_eve'] = 0
            df['days_to_holiday'] = 0
            return df
            
        print("\nCreating holiday features...")
        
        df = df.copy()
        
        # Is holiday
        df['is_holiday'] = df.index.map(lambda x: x.date() in self.us_holidays).astype(int)
        
        # Is day before holiday (holiday eve)
        df['is_holiday_eve'] = df.index.map(
            lambda x: (x.date() + timedelta(days=1)) in self.us_holidays
        ).astype(int)
        
        # Is day after holiday
        df['is_holiday_after'] = df.index.map(
            lambda x: (x.date() - timedelta(days=1)) in self.us_holidays
        ).astype(int)
        
        # Days to next holiday (capped at 30)
        def days_to_next_holiday(date):
            for i in range(30):
                check_date = date.date() + timedelta(days=i)
                if check_date in self.us_holidays:
                    return i
            return 30
        
        df['days_to_holiday'] = df.index.map(days_to_next_holiday)
        
        holiday_count = df['is_holiday'].sum()
        print(f"   Found {holiday_count} holiday hours ({holiday_count/24:.0f} holiday days)")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for time series forecasting.
        
        Args:
            df: DataFrame with target column
            
        Returns:
            DataFrame with lag features added
        """
        print("\nCreating lag features...")
        
        df = df.copy()
        target = self.target_column
        
        # Lag features
        lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]  # 168 = 1 week
        
        for lag in lag_hours:
            df[f'lag_{lag}h'] = df[target].shift(lag)
        
        # Same hour yesterday
        df['lag_24h_same_hour'] = df[target].shift(24)
        
        # Same hour last week
        df['lag_168h_same_hour'] = df[target].shift(168)
        
        # Same hour, same day last week
        df['lag_168h_same_day'] = df[target].shift(168)
        
        print(f"   Created {len(lag_hours) + 3} lag features")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window statistics features.
        
        Args:
            df: DataFrame with target column
            
        Returns:
            DataFrame with rolling features added
        """
        print("\nCreating rolling features...")
        
        df = df.copy()
        target = self.target_column
        
        # Rolling windows
        windows = [3, 6, 12, 24, 48, 168]  # hours
        
        for window in windows:
            # Rolling mean
            df[f'rolling_mean_{window}h'] = df[target].shift(1).rolling(window=window).mean()
            
            # Rolling std
            df[f'rolling_std_{window}h'] = df[target].shift(1).rolling(window=window).std()
            
            # Rolling min
            df[f'rolling_min_{window}h'] = df[target].shift(1).rolling(window=window).min()
            
            # Rolling max
            df[f'rolling_max_{window}h'] = df[target].shift(1).rolling(window=window).max()
        
        # Expanding mean (historical average up to that point)
        df['expanding_mean'] = df[target].shift(1).expanding().mean()
        
        # Difference features
        df['diff_1h'] = df[target].diff(1)
        df['diff_24h'] = df[target].diff(24)
        df['diff_168h'] = df[target].diff(168)
        
        # Percent change
        df['pct_change_1h'] = df[target].pct_change(1)
        df['pct_change_24h'] = df[target].pct_change(24)
        
        feature_count = len(windows) * 4 + 6
        print(f"   Created {feature_count} rolling/diff features")
        
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cyclical encoding for periodic features.
        
        Args:
            df: DataFrame with time features
            
        Returns:
            DataFrame with cyclical features added
        """
        print("\nCreating cyclical features...")
        
        df = df.copy()
        
        # Hour of day (0-23 → sine/cosine)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (0-6 → sine/cosine)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Day of month (1-31 → sine/cosine)
        df['dayofmonth_sin'] = np.sin(2 * np.pi * df['dayofmonth'] / 31)
        df['dayofmonth_cos'] = np.cos(2 * np.pi * df['dayofmonth'] / 31)
        
        # Month of year (1-12 → sine/cosine)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Day of year (1-365 → sine/cosine)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        
        print(f"   Created 10 cyclical features")
        
        return df
    
    def process_all_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            df: DataFrame to process. If None, loads and cleans data first.
            
        Returns:
            DataFrame with all features
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        if df is None:
            if self.raw_data is None:
                self.load_data()
            df = self.clean_data()
        
        # Apply all feature engineering steps
        df = self.create_calendar_features(df)
        df = self.create_holiday_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_cyclical_features(df)
        
        # Handle NaN values from lag/rolling features
        original_len = len(df)
        
        # Drop rows with NaN in target column
        df = df.dropna(subset=[self.target_column])
        
        # Fill remaining NaN in features with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        dropped = original_len - len(df)
        print(f"\nDropped {dropped} rows with missing target")
        print(f"Final dataset: {len(df):,} records")
        
        # Store processed data
        self.processed_data = df
        
        # Store feature columns (all except target and categorical)
        categorical_cols = ['hour_category']
        self.feature_columns = [
            col for col in df.columns 
            if col != self.target_column and col not in categorical_cols
        ]
        
        print(f"\n[OK] FEATURE ENGINEERING COMPLETE")
        print(f"   Total features: {len(self.feature_columns)}")
        
        return df
    
    def prepare_train_test_split(
        self, 
        df: pd.DataFrame = None,
        test_size: float = 0.2,
        split_date: str = None
    ) -> tuple:
        """
        Split data into training and test sets (time-based split).
        
        Args:
            df: DataFrame to split. If None, uses self.processed_data
            test_size: Proportion of data to use for testing (if split_date not provided)
            split_date: Specific date to split on (format: 'YYYY-MM-DD')
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if df is None:
            df = self.processed_data
            
        if df is None:
            raise ValueError("No data available. Run process_all_features() first.")
        
        print("\n" + "="*60)
        print("TRAIN/TEST SPLIT")
        print("="*60)
        
        # Determine split point
        if split_date:
            split_point = pd.to_datetime(split_date)
        else:
            split_idx = int(len(df) * (1 - test_size))
            split_point = df.index[split_idx]
        
        # Split data
        train = df[df.index < split_point]
        test = df[df.index >= split_point]
        
        # Prepare X and y
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        
        X_train = train[feature_cols]
        X_test = test[feature_cols]
        y_train = train[self.target_column]
        y_test = test[self.target_column]
        
        print(f"   Split date: {split_point}")
        print(f"   Training set: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
        print(f"      Date range: {train.index.min()} to {train.index.max()}")
        print(f"   Test set: {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
        print(f"      Date range: {test.index.min()} to {test.index.max()}")
        print(f"   Features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance_df(self, feature_names: list, importances: np.array) -> pd.DataFrame:
        """
        Create a DataFrame of feature importances.
        
        Args:
            feature_names: List of feature names
            importances: Array of importance values
            
        Returns:
            DataFrame sorted by importance
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_processed_data(self, output_path: str = None):
        """
        Save processed data to CSV.
        
        Args:
            output_path: Full path to output file. If None, uses config default.
        """
        if self.processed_data is None:
            raise ValueError("No processed data to save. Run process_all_features() first.")
        
        if output_path is None:
            output_path = config.PROCESSED_DIR / config.PROCESSED_DATA_FILE
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.processed_data.to_csv(output_path)
        print(f"Saved processed data to: {output_path}")
        
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the data.
        
        Returns:
            Dictionary with summary statistics
        """
        df = self.processed_data if self.processed_data is not None else self.raw_data
        
        if df is None:
            return {}
            
        return {
            'total_records': len(df),
            'date_range': {
                'start': str(df.index.min()),
                'end': str(df.index.max())
            },
            'target_stats': {
                'min': df[self.target_column].min(),
                'max': df[self.target_column].max(),
                'mean': df[self.target_column].mean(),
                'std': df[self.target_column].std(),
                'median': df[self.target_column].median()
            },
            'feature_count': len(self.feature_columns),
            'missing_values': df.isna().sum().sum()
        }


# Convenience function for quick data preparation
def prepare_pjm_data(
    data_path: str = None,
    filename: str = 'PJME_hourly.csv',
    test_size: float = 0.2,
    split_date: str = None
) -> tuple:
    """
    Quick function to load and prepare PJM data for ML models.
    
    Args:
        data_path: Path to data directory
        filename: CSV file to load
        test_size: Proportion for test set
        split_date: Optional specific split date
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, processor)
    """
    processor = PJMDataProcessor(data_path)
    processor.load_data(filename)
    processor.process_all_features()
    
    X_train, X_test, y_train, y_test = processor.prepare_train_test_split(
        test_size=test_size,
        split_date=split_date
    )
    
    return X_train, X_test, y_train, y_test, processor


# Example usage
if __name__ == "__main__":
    # Run data processing pipeline
    processor = PJMDataProcessor()
    
    # Load data
    raw_data = processor.load_data()
    
    # Process all features
    processed_data = processor.process_all_features()
    
    # Save processed data
    output_path = config.PROCESSED_DIR / config.PROCESSED_DATA_FILE
    os.makedirs(output_path.parent, exist_ok=True)
    processed_data.to_csv(output_path)
    print(f"\nProcessed data saved to: {output_path}")
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test = processor.prepare_train_test_split(
        test_size=0.2,
        split_date='2017-01-01'
    )
    
    # Print summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    summary = processor.get_data_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Save processed data
    processor.save_processed_data()

