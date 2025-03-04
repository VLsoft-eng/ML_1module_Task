import logging
import os
import pickle

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from clearml import Task

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/log_file.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VlClassifier:
    def __init__(self, model_path: str = 'data/models/model.pkl'):
        self.model_path = model_path
        self.model = None
        self.categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
        self.numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        self.task = None

    def init_clearml_task(self, task_name: str):
        """Инициализация задачи ClearML с принудительным закрытием предыдущей"""
        if self.task is not None:
            logger.info(f"Closing existing task: {self.task.task_id}")
            self.task.close()  # Закрываем текущую задачу, если она существует
            self.task = None  # Сбрасываем ссылку
        try:
            self.task = Task.init(
                project_name="Spaceship Titanic",
                task_name=task_name,
                output_uri=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize ClearML task: {str(e)}")
            self.task = None  # Есл
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()

            df[['Deck', 'Cabin_num', 'Side']] = df['Cabin'].str.split('/', expand=True)
            df['Cabin_num'] = pd.to_numeric(df['Cabin_num'], errors='coerce')

            df['Group'] = df['PassengerId'].str.split('_').str[0]
            df['GroupSize'] = df.groupby('Group')['Group'].transform('count')

            for num_feat in self.numeric_features:
                df[num_feat] = df[num_feat].fillna(df[num_feat].median())
            for cat_feat in self.categorical_features:
                df[cat_feat] = df[cat_feat].fillna('Unknown')

            df['TotalSpend'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)

            features = df.drop(columns=['PassengerId', 'Cabin', 'Name', 'Group'], errors='ignore')
            return features
        except Exception as e:
            logger.error(f"Error during preprocessing data: {str(e)}")
            raise

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: np.ndarray) -> float:
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'verbose': False,
            'random_seed': 42
        }
        model = CatBoostClassifier(**params)

        cat_features = [col for col in X.columns if X[col].dtype == 'object']

        split_idx = int(0.8 * len(y))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        model.fit(train_pool)
        accuracy = model.score(val_pool)

        if self.task:
            self.task.get_logger().report_scalar("trial_params", "iterations", value=params['iterations'],
                                                 iteration=trial.number)
        return accuracy

    def train(self, train_path: str, n_trials: int = 20):
        try:
            self.init_clearml_task("Spaceship_Training")

            df_raw = pd.read_csv(train_path)
            logger.info(f"Uploaded train dataset with size: {df_raw.shape}")

            features = self.preprocess_data(df_raw)
            X = features.drop('Transported', axis=1)
            y = features['Transported'].astype(int)

            if self.task:
                final_train_df = X.copy()
                final_train_df['Transported'] = y
                self.task.upload_artifact(
                    "final_train_dataset",
                    artifact_object=final_train_df,
                    metadata={"description": "Final preprocessed training dataset"}
                )
                logger.info("Final preprocessed training dataset uploaded successfully")

            logger.info("Parameters optimization started")
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self._objective(trial, X, y), n_trials=n_trials)

            best_params = study.best_params
            logger.info(f"Best params: {best_params}")
            if self.task:
                self.task.connect_configuration(best_params)

            self.model = CatBoostClassifier(**best_params, random_seed=42)
            cat_features = [col for col in X.columns if X[col].dtype == 'object']
            self.model.fit(X, y, cat_features=cat_features)

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)

            if os.path.exists(self.model_path):
                logger.info(f"Model saved in {self.model_path}")
                if self.task:
                    self.task.upload_artifact(
                        "best_model",
                        artifact_object=self.model_path,
                        metadata={
                            "description": "Best performing CatBoost model",
                            "accuracy": self.model.score(X, y),
                            "parameters": best_params
                        }
                    )
                    logger.info("Best model uploaded in ClearML successfully")
            else:
                logger.error(f"Error with saving model in {self.model_path}")
                raise FileNotFoundError(f"Error with saving model in {self.model_path}")

            accuracy = self.model.score(X, y)
            logger.info(f"Accuracy on training data: {accuracy:.4f}")
            if self.task:
                self.task.get_logger().report_single_value("final_accuracy", accuracy)
                logger.info("Closing ClearML session")
                self.task.close()

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, test_path: str):
        try:
            self.init_clearml_task("Spaceship_Prediction")
            logger.info("ClearML task initialized")

            if not os.path.exists(self.model_path):
                logger.error(f"Model from {self.model_path} not found!")
                raise FileNotFoundError(f"Model from {self.model_path} not found!")
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model successfully uploaded from {self.model_path}")

            df_raw = pd.read_csv(test_path)
            passenger_ids = df_raw['PassengerId']
            logger.info(f"Uploaded test dataset with size: {df_raw.shape}")

            features = self.preprocess_data(df_raw)

            if self.task:
                self.task.upload_artifact(
                    "final_test_dataset",
                    artifact_object=features,
                    metadata={"description": "Final preprocessed test dataset"}
                )
                logger.info("Final test dataset uploaded in ClearML as 'final_test_dataset'")

            predictions = self.model.predict(features)
            logger.info("Predictions completed")

            results = pd.DataFrame({
                'PassengerId': passenger_ids,
                'Transported': predictions.astype(bool)
            })
            os.makedirs('data', exist_ok=True)
            results.to_csv('data/results.csv', index=False)
            logger.info("Predictions saved in data/results.csv")

            if self.task:
                self.task.upload_artifact("predictions", artifact_object=results)
                logger.info("Prediction results uploaded in ClearML successfully")
                self.task.close()

        except Exception as e:
            logger.error(f"Error during prediction {str(e)}")
            raise