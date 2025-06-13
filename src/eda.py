# scripts/eda_engine.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


class InsuranceEDA:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None

    def load_data(self, delimiter: str = ','):
        print("[INFO] Loading data...")
        self.df = pd.read_csv(self.data_path, delimiter=delimiter)
        print("[INFO] Data loaded successfully with shape:", self.df.shape)

    def data_summary(self):
        print("\n[SUMMARY] Descriptive statistics:")
        print(self.df.describe(include='all'))
        print("\n[SUMMARY] Data types:")
        print(self.df.dtypes)

    def check_missing_values(self):
        print("\n[CHECK] Missing values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])
    
    def drop_column(self, column_name):
        """Drops a column from the DataFrame if it exists."""
        if column_name in self.df.columns:
            self.df.drop(columns=column_name, inplace=True)
            print(f"successfuly dropped {column_name} column")
        else:
            print(f"Column '{column_name}' does not exist in the DataFrame.")
   
    def univariate_analysis(self):
        print("\n[UNIVARIATE] Plotting histograms for numerical columns...")
        num_cols = self.df.select_dtypes(include=['number']).columns
        for col in num_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()

        print("\n[UNIVARIATE] Plotting bar plots for categorical columns...")
        cat_cols = self.df.select_dtypes(include='object').nunique()
        for col in cat_cols[cat_cols < 20].index:  # limit to low cardinality
            plt.figure(figsize=(6, 4))
            self.df[col].value_counts().plot(kind='bar')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()


    def bivariate_analysis(self):
        print("\n[BIVARIATE] Correlation Matrix for numeric features:")
        num_df = self.df.select_dtypes(include='number')
        corr = num_df.corr(numeric_only=True)
        print(corr)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

        if 'PostalCode' in self.df.columns and 'TotalPremium' in self.df.columns and 'TotalClaims' in self.df.columns:
            print("\n[RELATIONSHIP] TotalPremium vs TotalClaims by PostalCode")
            grouped = self.df.groupby('PostalCode')[['TotalPremium', 'TotalClaims']].mean().reset_index()
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=grouped, x='TotalPremium', y='TotalClaims')
            plt.title("Average Total Premium vs Claims per PostalCode")
            plt.tight_layout()
            plt.show()

    def loss_ratio_analysis(self, groupby_cols=["Province", "VehicleType", "Gender"]):
        print("\n[LOSS RATIO] Grouped Loss Ratios:")
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium'].replace(0, 1)

            for col in groupby_cols:
                if col in self.df.columns:
                    grouped = self.df.groupby(col)['LossRatio'].mean().sort_values(ascending=False)
                    print(f"\nLoss Ratio by {col}:\n", grouped)

                    plt.figure(figsize=(10, 6))
                    grouped.plot(kind='bar', color='orange')
                    plt.title(f"Average Loss Ratio by {col}")
                    plt.ylabel("Loss Ratio")
                    plt.xlabel(col)
                    plt.tight_layout()
                    plt.show()

    def time_trend_analysis(self):
        if 'TransactionMonth' in self.df.columns:
            print("\n[TIME TREND] Claims and Premium over Time")
            self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'], errors='coerce')
            time_grouped = self.df.groupby(self.df['TransactionMonth'].dt.to_period('M')).agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum'
            }).reset_index()

            time_grouped['TransactionMonth'] = time_grouped['TransactionMonth'].astype(str)

            plt.figure(figsize=(10, 5))
            plt.plot(time_grouped['TransactionMonth'], time_grouped['TotalPremium'], label='Total Premium')
            plt.plot(time_grouped['TransactionMonth'], time_grouped['TotalClaims'], label='Total Claims')
            plt.xticks(rotation=45)
            plt.title("Monthly Total Premium vs Claims")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def vehicle_risk_analysis(self):
        print("\n[VEHICLE RISK] Analyzing Claim by Vehicle Make/Model")
        if 'make' in self.df.columns and 'Model' in self.df.columns and 'TotalClaims' in self.df.columns:
            grouped = self.df.groupby(['make', 'Model'])['TotalClaims'].sum().reset_index()
            top = grouped.sort_values(by='TotalClaims', ascending=False).head(10)
            bottom = grouped.sort_values(by='TotalClaims', ascending=True).head(10)

            print("\nTop 10 Models by Total Claims:\n", top)
            print("\nBottom 10 Models by Total Claims:\n", bottom)

            plt.figure(figsize=(8, 5))
            sns.barplot(data=top, x='TotalClaims', y='Model', hue='make')
            plt.title("Top 10 Models by Total Claims")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(8, 5))
            sns.barplot(data=bottom, x='TotalClaims', y='Model', hue='make')
            plt.title("Bottom 10 Models by Total Claims")
            plt.tight_layout()
            plt.show()

    def detect_outliers(self):
        print("\n[OUTLIERS] Using box plots for outlier detection...")
        num_cols = ['TotalClaims', 'TotalPremium', 'CustomValueEstimate']
        for col in num_cols:
            if col in self.df.columns:
                plt.figure(figsize=(6, 4))
                sns.boxplot(x=self.df[col])
                plt.title(f"Boxplot for {col}")
                plt.tight_layout()
                plt.show()
    
