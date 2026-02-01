import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import joblib
warnings.filterwarnings('ignore')

class StudentPerformanceAnalyzer:
    def __init__(self, file_path=None, df=None):
        """Initialize the analyzer with data path or DataFrame"""
        if file_path:
            # Load with semicolon delimiter as in notebook
            self.df = pd.read_csv(file_path, delimiter=';')
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError("Either file_path or df must be provided")
        
        self.scaler = StandardScaler()
        self.kmeans = None
        self.rf_model = None
        self.rf_params = None
        
    def preprocess_data(self):
        """Preprocess the student data - EXACTLY as in notebook"""
        df = self.df.copy()
        
        df = df.drop_duplicates()
        
        cols_to_drop = ['school','address','Mjob','Fjob','reason','guardian']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        cols = ['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime',
                'goout','Dalc','Walc','health','absences','G1','G2','G3']
        
        exclude_from_clip = ['failures', 'traveltime', 'studytime', 'Medu', 'Fedu']
        
        for col in cols:
            if col in df.columns:
                if col not in exclude_from_clip:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower, upper)
        
        encoder = LabelEncoder()
        for col in df.select_dtypes('object').columns:
            df[col] = encoder.fit_transform(df[col])
        
        df['performance'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
        
        self.df = df
        return self.df
    
    def get_academic_summary(self):
        """Generate comprehensive academic statistics"""
        stats = {
            'total_students': len(self.df),
            'pass_count': len(self.df[self.df['performance'] == 1]),
            'fail_count': len(self.df[self.df['performance'] == 0]),
            'pass_rate': (len(self.df[self.df['performance'] == 1]) / len(self.df)) * 100,
            'avg_final_grade': self.df['G3'].mean(),
            'avg_studytime': self.df['studytime'].mean(),
            'avg_absences': self.df['absences'].mean(),
            'avg_failures': self.df['failures'].mean(),
            'avg_G1': self.df['G1'].mean(),
            'avg_G2': self.df['G2'].mean(),
            'high_achievers': len(self.df[self.df['G3'] >= 15]),
            'at_risk': len(self.df[self.df['G3'] < 10]),
            'perfect_attendance': len(self.df[self.df['absences'] == 0])
        }
        
        return stats
    
    def perform_clustering(self, n_clusters=2):
        """Perform K-means clustering EXACTLY as in notebook"""
        X_clustering = self.df.drop(['G1', 'G2', 'G3'], axis=1, errors='ignore')
        
        X_scaled = self.scaler.fit_transform(X_clustering)
        
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        
        self.kmeans = KMeans(n_clusters=n_clusters, init='k-means++', 
                            random_state=42, n_init=10)
        self.df['cluster_id'] = self.kmeans.fit_predict(X_scaled)
        
        score = silhouette_score(X_scaled, self.df['cluster_id'])
        
        cluster_impact = self.df.groupby('cluster_id')['G3'].mean()
        self.df['cluster_weight'] = self.df['cluster_id'].map(cluster_impact)
        
        cluster_centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        
        return {
            'wcss': wcss,
            'n_clusters': n_clusters,
            'silhouette_score': score,
            'cluster_labels': self.df['cluster_id'].values,
            'cluster_centers': cluster_centers,
            'cluster_impact': cluster_impact.to_dict()
        }
    
    def train_hybrid_model(self):
        """Train RandomForest hybrid model EXACTLY as in notebook"""
        X_hybrid = self.df.drop(['G3', 'performance', 'G1'], axis=1, errors='ignore')
        y = self.df['performance']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_hybrid, y, test_size=0.2, random_state=42, stratify=y
        )
        
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [None, 10, 20],
            'class_weight': ['balanced'],
            'min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid, cv=5, scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
        
        self.rf_model = grid_search.best_estimator_
        self.rf_params = grid_search.best_params_
        
        y_pred = self.rf_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        feature_importances = pd.Series(
            self.rf_model.feature_importances_, 
            index=X_hybrid.columns
        ).sort_values(ascending=False)
        
        return {
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'classification_report': report,
            'feature_importances': feature_importances.to_dict(),
            'y_true': y_test.values,
            'y_pred': y_pred,
            'feature_names': X_hybrid.columns.tolist(),
            'X_test': X_test,
            'y_test': y_test,
            'X_train': X_train,
            'y_train': y_train
        }
    
    def compare_models(self):
        """Compare traditional vs hybrid models EXACTLY as in notebook"""
        # Best parameters from notebook (EXACTLY as in notebook)
        best_params = {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 5,
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        # 1. Traditional model (without cluster)
        X_no_cluster = self.df.drop(['G3', 'performance', 'G1', 'cluster_id', 'cluster_weight'], 
                                   axis=1, errors='ignore')
        y = self.df['performance']
        X_train_nc, X_test_nc, y_train_nc, y_test_nc = train_test_split(
            X_no_cluster, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 2. Hybrid model (with cluster ID and cluster weight)
        X_hybrid = self.df.drop(['G3', 'performance', 'G1'], axis=1, errors='ignore')
        X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
            X_hybrid, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 3. Train both models
        model_no_cluster = RandomForestClassifier(**best_params)
        model_hybrid = RandomForestClassifier(**best_params)
        
        model_no_cluster.fit(X_train_nc, y_train_nc)
        model_hybrid.fit(X_train_h, y_train_h)
        
        # 4. Calculate accuracies (EXACTLY as in notebook)
        acc_nc = accuracy_score(y_test_nc, model_no_cluster.predict(X_test_nc))
        acc_h = accuracy_score(y_test_h, model_hybrid.predict(X_test_h))
        
        # 5. Calculate improvement (EXACTLY as in notebook)
        improvement = ((acc_h - acc_nc) / acc_nc) * 100 if acc_nc > 0 else 0
        
        return {
            'accuracy_without_cluster': acc_nc,
            'accuracy_with_cluster': acc_h,
            'improvement_percent': improvement,
            'model_comparison': {
                'traditional': model_no_cluster,
                'hybrid': model_hybrid
            }
        }
    
    def get_cluster_statistics(self):
        """Get detailed statistics for each cluster"""
        if 'cluster_id' not in self.df.columns:
            return None
        
        # Aggregate statistics by cluster
        agg_functions = {
            'G3': ['mean', 'std', 'min', 'max', 'count'],
            'G2': 'mean',
            'G1': 'mean',
            'studytime': 'mean',
            'absences': 'mean',
            'failures': 'mean',
            'performance': 'mean',
            'cluster_weight': 'mean'
        }
        
        # Select only existing columns
        existing_cols = [col for col in agg_functions.keys() if col in self.df.columns]
        agg_functions = {col: agg_functions[col] for col in existing_cols}
        
        cluster_stats = self.df.groupby('cluster_id').agg(agg_functions).round(2)
        
        # Flatten multi-index columns
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
        
        # Add student count per cluster
        cluster_stats['student_count'] = self.df['cluster_id'].value_counts().sort_index()
        
        # Reset index for better display
        cluster_stats = cluster_stats.reset_index()
        
        return cluster_stats
    
    def get_cluster_profiles(self):
        """Generate descriptive profiles for each cluster"""
        if 'cluster_id' not in self.df.columns:
            return None
        
        profiles = {}
        for cluster in sorted(self.df['cluster_id'].unique()):
            cluster_data = self.df[self.df['cluster_id'] == cluster]
            
            # Count performance in cluster
            pass_count = cluster_data['performance'].sum() if 'performance' in cluster_data.columns else 0
            
            profile = {
                'size': len(cluster_data),
                'avg_grade': cluster_data['G3'].mean() if 'G3' in cluster_data.columns else 0,
                'avg_G2': cluster_data['G2'].mean() if 'G2' in cluster_data.columns else 0,
                'avg_G1': cluster_data['G1'].mean() if 'G1' in cluster_data.columns else 0,
                'avg_studytime': cluster_data['studytime'].mean() if 'studytime' in cluster_data.columns else 0,
                'avg_absences': cluster_data['absences'].mean() if 'absences' in cluster_data.columns else 0,
                'avg_failures': cluster_data['failures'].mean() if 'failures' in cluster_data.columns else 0,
                'avg_cluster_weight': cluster_data['cluster_weight'].mean() if 'cluster_weight' in cluster_data.columns else 0,
                'pass_rate': (pass_count / len(cluster_data)) * 100 if len(cluster_data) > 0 else 0,
                'description': self._generate_cluster_description(cluster_data)
            }
            
            profiles[f'Cluster {cluster}'] = profile
        
        return profiles
    
    def _generate_cluster_description(self, cluster_data):
        """Generate human-readable description for a cluster"""
        # Performance level
        avg_grade = cluster_data['G3'].mean() if 'G3' in cluster_data.columns else 0
        if avg_grade >= 15:
            performance = "High Achievers"
            performance_emoji = "🏆"
        elif avg_grade >= 10:
            performance = "Average Performers"
            performance_emoji = "📊"
        else:
            performance = "At-Risk Students"
            performance_emoji = "⚠️"
        
        # Study habits
        avg_studytime = cluster_data['studytime'].mean() if 'studytime' in cluster_data.columns else 0
        if avg_studytime >= 3:
            study_habit = "Dedicated Studiers"
            study_emoji = "📚"
        elif avg_studytime >= 2:
            study_habit = "Regular Studiers"
            study_emoji = "📖"
        else:
            study_habit = "Minimal Studiers"
            study_emoji = "📝"
        
        # Attendance
        avg_absences = cluster_data['absences'].mean() if 'absences' in cluster_data.columns else 0
        if avg_absences >= 10:
            attendance = "Poor Attendance"
            attendance_emoji = "❌"
        elif avg_absences >= 5:
            attendance = "Frequent Absences"
            attendance_emoji = "⚠️"
        else:
            attendance = "Good Attendance"
            attendance_emoji = "✅"
        
        # Academic history
        avg_failures = cluster_data['failures'].mean() if 'failures' in cluster_data.columns else 0
        if avg_failures >= 1:
            failure_status = "History of Failures"
            failure_emoji = "📉"
        else:
            failure_status = "No Previous Failures"
            failure_emoji = "📈"
        
        return f"{performance_emoji} {performance} | {study_emoji} {study_habit} | {attendance_emoji} {attendance} | {failure_emoji} {failure_status}"
    
    def get_cluster_composition(self):
        """Get detailed composition of each cluster"""
        if 'cluster_id' not in self.df.columns:
            return None
        
        composition = {}
        for cluster in sorted(self.df['cluster_id'].unique()):
            cluster_data = self.df[self.df['cluster_id'] == cluster]
            
            # Performance composition
            pass_count = cluster_data['performance'].sum()
            fail_count = len(cluster_data) - pass_count
            
            # Study time composition
            high_study = len(cluster_data[cluster_data['studytime'] >= 3])
            medium_study = len(cluster_data[(cluster_data['studytime'] >= 2) & (cluster_data['studytime'] < 3)])
            low_study = len(cluster_data[cluster_data['studytime'] < 2])
            
            # Absence composition
            high_absence = len(cluster_data[cluster_data['absences'] >= 10])
            medium_absence = len(cluster_data[(cluster_data['absences'] >= 5) & (cluster_data['absences'] < 10)])
            low_absence = len(cluster_data[cluster_data['absences'] < 5])
            
            composition[f'Cluster {cluster}'] = {
                'total_students': len(cluster_data),
                'pass_students': pass_count,
                'fail_students': fail_count,
                'high_study': high_study,
                'medium_study': medium_study,
                'low_study': low_study,
                'high_absence': high_absence,
                'medium_absence': medium_absence,
                'low_absence': low_absence
            }
        
        return composition
    
    def get_correlation_matrix(self):
        """Get correlation matrix for all features"""
        return self.df.corr(numeric_only=True)
    
    def save_model(self, path='student_performance_model.pkl'):
        """Save the trained model"""
        if self.rf_model:
            joblib.dump({
                'model': self.rf_model,
                'scaler': self.scaler,
                'params': self.rf_params
            }, path)
            return True
        return False
    
    def load_model(self, path='student_performance_model.pkl'):
        """Load a trained model"""
        saved_data = joblib.load(path)
        self.rf_model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.rf_params = saved_data['params']
        return self.rf_model

    def validate_data_structure(self):
        """Validate that the data has required columns from notebook"""
        required_columns = ['G1', 'G2', 'G3', 'studytime', 'absences', 'failures']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        return True, "Data structure is valid"

    # ============ PREDICTION FUNCTIONS ADDED FROM NOTEBOOK ============
    
    def predict_student_performance(self, student_features):
        """Predict performance for a single student EXACTLY as in notebook"""
        if self.rf_model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Convert to DataFrame
        if isinstance(student_features, dict):
            student_df = pd.DataFrame([student_features])
        else:
            student_df = student_features.copy()
        
        # 2. Ensure all required features are present
        # The model expects the same features as during training
        required_features = self.rf_model.feature_names_in_
        
        # Check for missing features
        missing_features = set(required_features) - set(student_df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # 3. Reorder columns to match training data
        student_df = student_df[required_features]
        
        # 4. Make prediction (EXACTLY as in notebook)
        prediction = self.rf_model.predict(student_df)
        prediction_proba = self.rf_model.predict_proba(student_df)
        
        # 5. Return results in notebook format
        return {
            'prediction': int(prediction[0]),  # 1 for pass, 0 for fail
            'probability': float(prediction_proba[0][1]),  # Probability of passing
            'confidence': float(max(prediction_proba[0])),  # Confidence score
            'predicted_class': 'Pass' if prediction[0] == 1 else 'Fail',
            'features_used': required_features.tolist()
        }
    
    def batch_predict(self, students_data):
        """Predict performance for multiple students EXACTLY as in notebook"""
        if self.rf_model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Convert to DataFrame
        if isinstance(students_data, list):
            students_df = pd.DataFrame(students_data)
        else:
            students_df = students_data.copy()
        
        # Get required features
        required_features = self.rf_model.feature_names_in_
        
        # Check for missing features
        missing_features = set(required_features) - set(students_df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Reorder columns
        students_df = students_df[required_features]
        
        # Make predictions (EXACTLY as in notebook)
        predictions = self.rf_model.predict(students_df)
        predictions_proba = self.rf_model.predict_proba(students_df)
        
        # Prepare results in notebook format
        results = []
        for i in range(len(predictions)):
            results.append({
                'student_id': i,
                'prediction': int(predictions[i]),
                'probability_pass': float(predictions_proba[i][1]),
                'probability_fail': float(predictions_proba[i][0]),
                'confidence': float(max(predictions_proba[i])),
                'predicted_class': 'Pass' if predictions[i] == 1 else 'Fail',
                'at_risk': predictions[i] == 0
            })
        
        # Add to DataFrame for easy analysis
        results_df = pd.DataFrame(results)
        
        # Calculate statistics (matching notebook analysis)
        stats = {
            'total_students': len(results_df),
            'predicted_pass': results_df['prediction'].sum(),
            'predicted_fail': len(results_df) - results_df['prediction'].sum(),
            'pass_rate': (results_df['prediction'].sum() / len(results_df)) * 100,
            'avg_confidence': results_df['confidence'].mean(),
            'high_risk_students': len(results_df[results_df['probability_pass'] < 0.3]),
            'moderate_risk_students': len(results_df[(results_df['probability_pass'] >= 0.3) & (results_df['probability_pass'] < 0.5)]),
            'low_risk_students': len(results_df[results_df['probability_pass'] >= 0.5])
        }
        
        return {
            'predictions': results_df,
            'statistics': stats,
            'model_metrics': {
                'model_type': 'RandomForestClassifier',
                'best_params': self.rf_params,
                'features_used': required_features.tolist()
            }
        }
    
    def get_prediction_explanation(self, student_features, top_n=5):
        """Explain prediction using feature importance EXACTLY as in notebook"""
        if self.rf_model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Get prediction first
        prediction_result = self.predict_student_performance(student_features)
        
        # Get feature importance (EXACTLY as in notebook)
        feature_importance = pd.Series(
            self.rf_model.feature_importances_,
            index=self.rf_model.feature_names_in_
        ).sort_values(ascending=False)
        
        # Get top features contributing to the decision
        top_features = feature_importance.head(top_n)
        
        # Get student's actual values for these features
        if isinstance(student_features, dict):
            student_vals = student_features
        else:
            student_vals = student_features.iloc[0].to_dict()
        
        explanation = {
            'prediction': prediction_result['predicted_class'],
            'confidence': prediction_result['confidence'],
            'top_factors': []
        }
        
        for feature in top_features.index:
            value = student_vals.get(feature, 'N/A')
            importance = feature_importance[feature]
            
            # Determine if this feature contributes positively or negatively
            # Based on the direction of impact (simplified approach)
            if feature in ['G2', 'G1', 'studytime', 'cluster_weight']:
                impact = 'Positive' if float(value) > self.df[feature].median() else 'Negative'
            elif feature in ['absences', 'failures']:
                impact = 'Negative' if float(value) > self.df[feature].median() else 'Positive'
            else:
                impact = 'Neutral'
            
            explanation['top_factors'].append({
                'feature': feature,
                'value': value,
                'importance': importance,
                'impact': impact,
                'description': self._get_feature_description(feature, value)
            })
        
        return explanation
    
    def _get_feature_description(self, feature, value):
        """Generate human-readable description for a feature value"""
        descriptions = {
            'G2': f"Mid-term grade: {value}/20",
            'G1': f"First period grade: {value}/20",
            'G3': f"Final grade: {value}/20",
            'studytime': f"Study time: {value} hours per week",
            'absences': f"Number of absences: {value}",
            'failures': f"Past class failures: {value}",
            'cluster_weight': f"Cluster performance impact: {value:.2f}",
            'cluster_id': f"Behavioral cluster: {value}",
            'age': f"Age: {value} years",
            'Medu': f"Mother's education level: {value}",
            'Fedu': f"Father's education level: {value}",
            'traveltime': f"Travel time to school: {value}",
            'famrel': f"Family relationship quality: {value}",
            'freetime': f"Free time after school: {value}",
            'goout': f"Going out with friends: {value}",
            'Dalc': f"Workday alcohol consumption: {value}",
            'Walc': f"Weekend alcohol consumption: {value}",
            'health': f"Health status: {value}"
        }
        
        return descriptions.get(feature, f"{feature}: {value}")
    
    def generate_recommendations(self, student_features):
        """Generate personalized recommendations based on prediction EXACTLY as notebook logic"""
        # Get prediction and explanation
        prediction = self.predict_student_performance(student_features)
        explanation = self.get_prediction_explanation(student_features)
        
        recommendations = {
            'prediction_summary': prediction['predicted_class'],
            'risk_level': 'High' if prediction['probability'] < 0.3 else 
                          'Moderate' if prediction['probability'] < 0.5 else 'Low',
            'key_areas': [],
            'action_items': [],
            'monitoring_suggestions': []
        }
        
        # Analyze top factors to generate recommendations
        for factor in explanation['top_factors'][:3]:  # Top 3 factors
            feature = factor['feature']
            value = factor['value']
            impact = factor['impact']
            
            if impact == 'Negative':
                if feature == 'studytime':
                    recommendations['key_areas'].append('Study Habits')
                    recommendations['action_items'].append('Increase weekly study time by 2-3 hours')
                elif feature == 'absences':
                    recommendations['key_areas'].append('Attendance')
                    recommendations['action_items'].append('Improve attendance rate and reduce absences')
                elif feature == 'failures':
                    recommendations['key_areas'].append('Academic History')
                    recommendations['action_items'].append('Seek additional tutoring for subjects with previous failures')
                elif feature in ['G1', 'G2']:
                    recommendations['key_areas'].append('Academic Performance')
                    recommendations['action_items'].append('Focus on improving mid-term performance')
        
        # Add cluster-specific recommendations if cluster_id is available
        if 'cluster_id' in student_features:
            cluster_id = student_features['cluster_id']
            cluster_data = self.df[self.df['cluster_id'] == cluster_id]
            
            if len(cluster_data) > 0:
                cluster_avg = cluster_data['G3'].mean()
                if cluster_avg < 10:
                    recommendations['action_items'].append('Join cluster-specific intervention program')
                    recommendations['monitoring_suggestions'].append('Weekly check-ins with academic advisor')
        
        # Ensure unique recommendations
        recommendations['key_areas'] = list(set(recommendations['key_areas']))
        recommendations['action_items'] = list(set(recommendations['action_items']))
        
        return recommendations