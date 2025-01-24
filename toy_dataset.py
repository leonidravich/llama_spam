import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetProcessor:
    def __init__(self, dataset_dir='datasets', test_size=0.3, eval_test_ratio=0.5, random_state=42):
        """
        Initialize the DatasetProcessor with dataset parameters.

        Args:
            dataset_dir (str): Directory path containing the dataset file.
            test_size (float): Proportion of the data to include in the test set.
            eval_test_ratio (float): Ratio to split test set further into evaluation and test.
            random_state (int): Random seed for reproducibility.
        """
        self.dataset_path = f"{dataset_dir}/emails.csv"
        self.test_size = test_size
        self.eval_test_ratio = eval_test_ratio
        self.random_state = random_state
        self.dataframe = None

    def preprocess_dataset(self):
        """
        Preprocess the dataset by loading, renaming columns, removing duplicates,
        and assigning labels based on 'spam' values.

        Returns:
            pd.DataFrame: The preprocessed dataset.
        """
        dataframe = pd.read_csv(self.dataset_path)
        dataframe.columns = ["text", "spam"]
        dataframe = dataframe.drop_duplicates()
        dataframe['label'] = dataframe['spam'].apply(lambda x: 'spam' if x == 1 else 'ham')
        self.dataframe = dataframe

    def split_data(self):
        """
        Splits the preprocessed dataframe into training, evaluation, and testing datasets.

        Returns:
            tuple: Split data (X_train, X_eval, X_test, y_train, y_eval, y_test)
        """
        if self.dataframe is None:
            raise ValueError("Data not preprocessed. Please preprocess the dataset first.")

        X = self.dataframe['text']
        y = self.dataframe['label']

        # Initial split into train and temp (for eval and test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y
        )

        # Further split temp into evaluation and test sets
        X_eval, X_test, y_eval, y_test = train_test_split(
            X_temp, y_temp, test_size=self.eval_test_ratio,
            random_state=self.random_state, stratify=y_temp
        )

        return X_train, X_eval, X_test, y_train, y_eval, y_test

def test_dataset_processor():
    print("Testing spam toy dataset Processor:")
    # Initialize the processor
    processor = DatasetProcessor()

    # Preprocess the dataset
    processor.preprocess_dataset()

    # Split into train, eval, and test datasets
    X_train, X_eval, X_test, y_train, y_eval, y_test = processor.split_data()

    print("Training Data:")
    print(y_train.value_counts(normalize=False))
    print(y_train.value_counts(normalize=True))

    # Print statistics for eval data
    print("Evaluation Data:")
    print(y_eval.value_counts(normalize=False))
    print(y_eval.value_counts(normalize=True))

    # Print statistics for test data
    print("Test Data:")
    print(y_test.value_counts(normalize=False))
    print(y_test.value_counts(normalize=True))

if __name__ == "__main__":
    test_dataset_processor()