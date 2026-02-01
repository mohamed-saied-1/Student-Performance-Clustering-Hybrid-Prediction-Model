import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class StudentVisualizations:
    def __init__(self):
        # Complete dark theme color palette
        self.dark_theme = {
            'background': '#0F172A',
            'paper_bg': '#1E293B',
            'plot_bg': '#1E293B',
            'text': '#E2E8F0',
            'text_secondary': '#94A3B8',
            'grid': 'rgba(148, 163, 184, 0.1)',
            'border': 'rgba(148, 163, 184, 0.2)',
            'primary': '#3B82F6',
            'secondary': '#8B5CF6',
            'success': '#10B981',
            'warning': '#F59E0B',
            'danger': '#EF4444',
            'info': '#06B6D4'
        }
        
        # Cluster colors optimized for dark theme
        self.cluster_colors = [
            '#FF6B6B', '#4ECDC4', '#FFD166',  # Red, Teal, Yellow
            '#06D6A0', '#118AB2', '#EF476F',  # Green, Blue, Pink
            '#7209B7', '#F72585', '#4361EE'   # Purple, Magenta, Indigo
        ]
        
        # Chart templates for dark theme
        self.template = "plotly_dark"
        
    def _apply_dark_theme(self, fig, height=None):
        """Apply dark theme to any plotly figure"""
        # Make height more flexible
        if height is None:
            height = 400  # Default height
        
        fig.update_layout(
            template=self.template,
            plot_bgcolor=self.dark_theme['plot_bg'],
            paper_bgcolor=self.dark_theme['paper_bg'],
            font=dict(
                family="Inter, Arial, sans-serif",
                size=11,  # Reduced from 12 for better fit
                color=self.dark_theme['text']
            ),
            height=height,
            title=dict(
                font=dict(
                    size=16,  # Reduced from 18
                    color=self.dark_theme['text'],
                    family="Inter, Arial, sans-serif"
                ),
                x=0.5,
                xanchor='center',
                y=0.95
            ),
            hoverlabel=dict(
                bgcolor=self.dark_theme['paper_bg'],
                font_size=11,  # Reduced from 12
                font_color=self.dark_theme['text'],
                font_family="Inter, Arial, sans-serif"
            ),
            # Increased margins for better text spacing
            margin=dict(t=70, b=70, l=70, r=70, pad=10),
            autosize=True  # Add autosize for responsiveness
        )
        
        # Update axes for dark theme
        fig.update_xaxes(
            gridcolor=self.dark_theme['grid'],
            linecolor=self.dark_theme['border'],
            tickfont=dict(
                color=self.dark_theme['text_secondary'],
                family="Inter, Arial, sans-serif",
                size=10  # Reduced tick font size
            ),
            title_font=dict(
                color=self.dark_theme['text'],
                family="Inter, Arial, sans-serif",
                size=13  # Reduced from 14
            ),
            showgrid=True,
            zeroline=False
        )
        fig.update_yaxes(
            gridcolor=self.dark_theme['grid'],
            linecolor=self.dark_theme['border'],
            tickfont=dict(
                color=self.dark_theme['text_secondary'],
                family="Inter, Arial, sans-serif",
                size=10  # Reduced tick font size
            ),
            title_font=dict(
                color=self.dark_theme['text'],
                family="Inter, Arial, sans-serif",
                size=13  # Reduced from 14
            ),
            showgrid=True,
            zeroline=False
        )
        
        return fig
    
    def create_grade_distribution(self, df):
        """Create G3 grade distribution histogram with dark theme"""
        fig = px.histogram(
            df,
            x='G3',
            nbins=20,
            title='📊 Final Grade (G3) Distribution',
            color_discrete_sequence=[self.dark_theme['primary']],
            opacity=0.85,
            labels={'G3': 'Final Grade', 'count': 'Number of Students'}
        )
        
        # Add passing threshold line
        fig.add_vline(
            x=10, 
            line_dash="dash", 
            line_color=self.dark_theme['danger'],
            line_width=2,
            annotation_text="Passing Threshold (10)",
            annotation_position="top right",
            annotation_font=dict(
                color=self.dark_theme['text'],
                size=11,  # Reduced
                family="Inter, Arial, sans-serif"
            ),
            annotation_bgcolor=self.dark_theme['paper_bg'],
            annotation_bordercolor=self.dark_theme['border']
        )
        
        # Add mean line
        mean_val = df['G3'].mean()
        fig.add_vline(
            x=mean_val, 
            line_dash="dot", 
            line_color=self.dark_theme['success'],
            line_width=2,
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="top left",
            annotation_font=dict(
                color=self.dark_theme['text'],
                size=11,  # Reduced
                family="Inter, Arial, sans-serif"
            ),
            annotation_bgcolor=self.dark_theme['paper_bg'],
            annotation_bordercolor=self.dark_theme['border']
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig, height=450)
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix):
        """Create correlation heatmap matching notebook style"""
        # Ensure we have numeric data only
        corr_numeric = correlation_matrix.select_dtypes(include=[np.number])
        
        fig = px.imshow(
            corr_numeric,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="🔗 Feature Correlation Matrix",
            labels=dict(color="Correlation"),
            width=800,
            height=800
        )
        
        # Customize colorbar
        fig.update_coloraxes(
            colorbar=dict(
                title="Correlation",
                title_font=dict(size=13, color=self.dark_theme['text']),
                tickfont=dict(size=11, color=self.dark_theme['text_secondary'])
            )
        )
        
        # Apply dark theme with custom height
        fig = self._apply_dark_theme(fig)
        fig.update_layout(height=700)
        
        return fig
    
    def create_feature_importance_chart(self, feature_importances):
        """Create feature importance bar chart for hybrid model"""
        # Convert to DataFrame for easier plotting
        if isinstance(feature_importances, dict):
            df_importances = pd.DataFrame({
                'feature': list(feature_importances.keys()),
                'importance': list(feature_importances.values())
            })
        else:
            df_importances = feature_importances
            
        # Sort by importance and get top 15
        df_importances = df_importances.sort_values('importance', ascending=True)
        df_top = df_importances.tail(15)
        
        # Create color gradient
        colors = px.colors.sequential.Teal
        
        fig = px.bar(
            df_top,
            y='feature',
            x='importance',
            orientation='h',
            title='🏆 Top 15 Feature Importances',
            color='importance',
            color_continuous_scale=colors,
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
            # Removed text_auto to prevent overlap
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=True,
            coloraxis_colorbar=dict(
                title="Importance",
                title_font=dict(size=12, color=self.dark_theme['text']),
                tickfont=dict(size=10, color=self.dark_theme['text_secondary'])
            ),
            height=500,
            # Add more margin for text
            margin=dict(t=80, b=80, l=120, r=100)
        )
        
        # Improved text visibility - position inside bars
        fig.update_traces(
            texttemplate='%{x:.3f}',
            textposition='inside',  # Changed from 'outside' to 'inside'
            insidetextanchor='middle',
            textfont=dict(
                color='white',
                size=9,  # Smaller font
                family="Inter, Arial, sans-serif"
            ),
            marker_line_width=0
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig, height=500)
        
        return fig
    
    def create_model_comparison_chart(self, model_results):
        """Create bar chart comparing traditional vs hybrid models"""
        models = ['Traditional Model', 'Hybrid Model (Cluster)']
        accuracies = [model_results['accuracy_without_cluster'], 
                     model_results['accuracy_with_cluster']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=accuracies,
                marker_color=[self.dark_theme['warning'], self.dark_theme['success']],
                marker_line_color='white',
                marker_line_width=1.5,
                opacity=0.85,
                text=[f"{acc:.4f}" for acc in accuracies],
                textposition='auto',  # Changed to 'auto' for better positioning
                textfont=dict(
                    color='white',
                    size=12,  # Reduced from 14
                    family="Inter, Arial, sans-serif"
                ),
                hoverinfo='text+y',
                hovertext=[f"Accuracy: {acc:.4f}" for acc in accuracies]
            )
        ])
        
        fig.update_layout(
            title='⚖️ Model Comparison: Traditional vs Hybrid',
            yaxis_title='Accuracy Score',
            yaxis=dict(
                range=[0, 1.0],
                tickformat='.2f'
            ),
            showlegend=False,
            # Add more margin
            margin=dict(t=80, b=80, l=80, r=80)
        )
        
        # Add improvement annotation
        improvement = model_results['improvement_percent']
        if improvement > 0:
            arrow_color = self.dark_theme['success']
            symbol = "↑"
        else:
            arrow_color = self.dark_theme['danger']
            symbol = "↓"
        
        fig.add_annotation(
            x=1, y=max(accuracies) + 0.05,
            text=f"{symbol} {abs(improvement):.2f}%",
            showarrow=False,
            font=dict(
                size=13,  # Reduced
                color=arrow_color,
                family="Inter, Arial, sans-serif"
            ),
            bgcolor=self.dark_theme['paper_bg'],
            bordercolor=self.dark_theme['border'],
            borderwidth=1,
            borderpad=4
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig, height=450)
        
        return fig
    
    def create_confusion_matrix_heatmap(self, y_true, y_pred, labels=['Fail', 'Pass']):
        """Create confusion matrix visualization"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create annotated heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Teal',
            showscale=True,
            text=cm,
            texttemplate='%{text}',
            textfont=dict(
                size=13,  # Reduced from 14
                color='white',
                family="Inter, Arial, sans-serif"
            ),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='📊 Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            height=400
        )
        
        # Update colorbar
        fig.update_coloraxes(
            colorbar=dict(
                title="Count",
                title_font=dict(size=12, color=self.dark_theme['text']),
                tickfont=dict(size=10, color=self.dark_theme['text_secondary'])
            )
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig, height=400)
        
        return fig
    
    def create_classification_report_table(self, report_dict):
        """Create table for classification report"""
        # Convert report to DataFrame
        report_df = pd.DataFrame(report_dict).transpose().round(3)
        
        # Create table figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Class'] + list(report_df.columns),
                fill_color=self.dark_theme['paper_bg'],
                align=['left', 'center'],
                font=dict(
                    color=self.dark_theme['text'],
                    size=12,  # Reduced
                    family="Inter, Arial, sans-serif"
                ),
                height=35
            ),
            cells=dict(
                values=[report_df.index] + [report_df[col] for col in report_df.columns],
                fill_color=self.dark_theme['plot_bg'],
                align=['left', 'center'],
                font=dict(
                    color=self.dark_theme['text_secondary'],
                    size=11,  # Reduced
                    family="Inter, Arial, sans-serif"
                ),
                height=30
            )
        )])
        
        fig.update_layout(
            title='📋 Classification Report',
            height=350,
            margin=dict(t=60, b=20, l=20, r=20)
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig, height=350)
        
        return fig
    
    def create_feature_distributions(self, df, features=None):
        """Create distribution plots for all features (dark theme)"""
        # If no specific features provided, use a default set or all numeric columns
        if features is None:
            # Try to get default features if they exist
            default_features = ['studytime', 'failures', 'absences', 'G1', 'G2', 'G3']
            available_features = [f for f in default_features if f in df.columns]
            
            if available_features:
                features = available_features
            else:
                # Use all numeric columns, but limit to first 6
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                features = numeric_cols[:6]
        
        n_cols = 3
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=[f'📈 {feature}' for feature in features],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        for i, feature in enumerate(features):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=df[feature],
                    name=feature,
                    marker_color=self.dark_theme['primary'],
                    marker_line_color='white',
                    marker_line_width=1,
                    opacity=0.8,
                    nbinsx=20,
                    hovertemplate=f"<b>{feature}</b><br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>"
                ),
                row=row, col=col
            )
            
            # Add mean line
            mean_val = df[feature].mean()
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color=self.dark_theme['danger'],
                line_width=2,
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="top right",
                annotation_font=dict(
                    size=10,  # Reduced
                    color=self.dark_theme['text'],
                    family="Inter, Arial, sans-serif"
                ),
                annotation_bgcolor='rgba(30, 41, 59, 0.8)',
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="📊 Feature Distributions",
            showlegend=False,
            height=300 * n_rows
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig, height=300 * n_rows)
        
        # Update subplot titles for dark theme
        fig.update_annotations(
            font=dict(
                size=13,  # Reduced
                color=self.dark_theme['text'],
                family="Inter, Arial, sans-serif"
            )
        )
        
        return fig
    
    def create_elbow_method_plot(self, wcss, selected_clusters=None):
        """Create elbow method plot for optimal clusters (dark theme)"""
        fig = go.Figure()
        
        # Add WCSS line
        fig.add_trace(go.Scatter(
            x=list(range(1, len(wcss) + 1)),
            y=wcss,
            mode='lines+markers',
            name='WCSS',
            line=dict(color=self.dark_theme['primary'], width=3),
            marker=dict(
                size=10,
                color=self.dark_theme['primary'],
                line=dict(color='white', width=1.5)
            ),
            hovertemplate="<b>Clusters: %{x}</b><br>WCSS: %{y:.2f}<extra></extra>"
        ))
        
        # Highlight selected clusters if provided
        if selected_clusters is not None:
            fig.add_vline(
                x=selected_clusters, 
                line_dash="dash", 
                line_color=self.dark_theme['danger'],
                line_width=2.5,
                annotation_text=f"Selected: {selected_clusters} clusters",
                annotation_position="top right",
                annotation_font=dict(
                    color=self.dark_theme['text'],
                    size=11,  # Reduced
                    family="Inter, Arial, sans-serif"
                ),
                annotation_bgcolor=self.dark_theme['paper_bg'],
                annotation_bordercolor=self.dark_theme['border']
            )
        
        fig.update_layout(
            title='📊 Elbow Method for Optimal Number of Clusters',
            xaxis_title='Number of Clusters',
            yaxis_title='Within-Cluster Sum of Squares (WCSS)',
            hovermode='x unified'
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig, height=500)
        
        # Configure x-axis
        fig.update_xaxes(
            tickmode='linear',
            tick0=1,
            dtick=1
        )
        
        return fig
    
    def create_cluster_scatter(self, df, x_col, y_col, cluster_col='cluster_id'):
        """Create scatter plot colored by clusters (dark theme)"""
        # Ensure cluster_id is treated as categorical
        df_display = df.copy()
        df_display[cluster_col] = df_display[cluster_col].astype(str)
        
        fig = px.scatter(
            df_display,
            x=x_col,
            y=y_col,
            color=cluster_col,
            title=f'📍 {x_col} vs {y_col} by Cluster',
            labels={cluster_col: 'Cluster'},
            color_discrete_sequence=self.cluster_colors,
            opacity=0.75,
            hover_data=['G3', 'studytime', 'absences', 'failures', 'performance']
        )
        
        # Improve marker appearance
        fig.update_traces(
            marker=dict(
                size=10,
                line=dict(width=1.5, color='white'),
                opacity=0.8
            ),
            selector=dict(mode='markers')
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend=dict(
                title="Cluster",
                title_font=dict(
                    color=self.dark_theme['text'],
                    size=12,
                    family="Inter, Arial, sans-serif"
                ),
                font=dict(
                    color=self.dark_theme['text_secondary'],
                    size=11,
                    family="Inter, Arial, sans-serif"
                ),
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(30, 41, 59, 0.7)',
                bordercolor=self.dark_theme['border'],
                borderwidth=1
            ),
            # Add more margin for better spacing
            margin=dict(t=80, b=80, l=80, r=100),
            autosize=True
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig, height=550)
        
        return fig

    def create_3d_cluster_scatter(self, df, x_col, y_col, z_col, cluster_col='cluster_id'):
        """Create 3D scatter plot colored by clusters (dark theme)"""
        # Ensure cluster_id is treated as categorical
        df_display = df.copy()
        df_display[cluster_col] = df_display[cluster_col].astype(str)
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df_display,
            x=x_col,
            y=y_col,
            z=z_col,
            color=cluster_col,
            title=f'📍 3D Cluster Visualization: {x_col} vs {y_col} vs {z_col}',
            labels={cluster_col: 'Cluster'},
            color_discrete_sequence=self.cluster_colors,
            opacity=0.8,
            hover_data=['G3', 'studytime', 'absences', 'failures', 'performance']
        )
        
        # Improve marker appearance
        fig.update_traces(
            marker=dict(
                size=5,
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            selector=dict(mode='markers')
        )
        
        # Update layout for 3D
        fig.update_layout(
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                xaxis=dict(
                    backgroundcolor=self.dark_theme['plot_bg'],
                    gridcolor=self.dark_theme['grid'],
                    showbackground=True,
                    zerolinecolor=self.dark_theme['border']
                ),
                yaxis=dict(
                    backgroundcolor=self.dark_theme['plot_bg'],
                    gridcolor=self.dark_theme['grid'],
                    showbackground=True,
                    zerolinecolor=self.dark_theme['border']
                ),
                zaxis=dict(
                    backgroundcolor=self.dark_theme['plot_bg'],
                    gridcolor=self.dark_theme['grid'],
                    showbackground=True,
                    zerolinecolor=self.dark_theme['border']
                )
            ),
            legend=dict(
                title="Cluster",
                title_font=dict(
                    color=self.dark_theme['text'],
                    size=12,
                    family="Inter, Arial, sans-serif"
                ),
                font=dict(
                    color=self.dark_theme['text_secondary'],
                    size=11,
                    family="Inter, Arial, sans-serif"
                ),
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(30, 41, 59, 0.7)',
                bordercolor=self.dark_theme['border'],
                borderwidth=1
            ),
            margin=dict(t=80, b=80, l=80, r=100)
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig, height=600)
        
        return fig
    
    def create_cluster_radar_chart(self, cluster_profiles):
        """Create radar chart comparing cluster profiles (dark theme) with better spacing"""
        if not cluster_profiles:
            return None
        
        categories = ['Grade (G3)', 'Study Time', 'Attendance', 'No Failures']
        
        fig = go.Figure()
        
        for idx, (cluster_name, profile) in enumerate(cluster_profiles.items()):
            # Normalize values to 0-100 scale
            values = [
                (profile['avg_grade'] / 20) * 100,           # Grade (0-20 → 0-100)
                (profile['avg_studytime'] / 4) * 100,        # Study time (1-4 → 0-100)
                100 - min(profile['avg_absences'], 50) * 2,  # Attendance (0-50 → 100-0)
                100 - (profile['avg_failures'] / 4) * 100    # No failures (0-4 → 100-0)
            ]
            
            color = self.cluster_colors[idx % len(self.cluster_colors)]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=cluster_name,
                line=dict(color=color, width=2.5),
                fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.25,)}',
                hovertemplate=f"<b>{cluster_name}</b><br>%{{theta}}: %{{r:.1f}}<extra></extra>"
            ))
        
        # Updated layout with better spacing
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(
                        size=9,  # Reduced further for better fit
                        color=self.dark_theme['text_secondary'],
                        family="Inter, Arial, sans-serif"
                    ),
                    gridcolor=self.dark_theme['grid'],
                    linecolor=self.dark_theme['border']
                ),
                angularaxis=dict(
                    tickfont=dict(
                        size=10,  # Reduced for better fit
                        color=self.dark_theme['text_secondary'],
                        family="Inter, Arial, sans-serif"
                    ),
                    gridcolor=self.dark_theme['grid'],
                    linecolor=self.dark_theme['border']
                ),
                bgcolor=self.dark_theme['paper_bg'],
                # Add more space around the radar
                domain=dict(
                    x=[0.1, 0.9],  # Leave space on left and right
                    y=[0.15, 0.85]  # Leave space on top and bottom
                )
            ),
            title='📈 Cluster Profile Comparison',
            showlegend=True,
            legend=dict(
                title="Clusters",
                title_font=dict(
                    color=self.dark_theme['text'],
                    size=11,  # Reduced
                    family="Inter, Arial, sans-serif"
                ),
                font=dict(
                    color=self.dark_theme['text_secondary'],
                    size=10,  # Reduced
                    family="Inter, Arial, sans-serif"
                ),
                orientation="h",  # Horizontal orientation
                yanchor="bottom",
                y=-0.1,  # Move legend below the chart
                xanchor="center",
                x=0.5,
                bgcolor='rgba(30, 41, 59, 0.7)',
                bordercolor=self.dark_theme['border'],
                borderwidth=1
            ),
            # Add more margin to prevent overlap
            margin=dict(t=50, b=120, l=50, r=50)  # Increased bottom margin for legend
        )
        
        # Apply dark theme with adjusted height
        fig = self._apply_dark_theme(fig, height=550)
        
        return fig
    
    def create_box_plot_comparison(self, df, feature, group_by='cluster_id'):
        """Create box plot comparing feature across clusters (dark theme) with better spacing"""
        # Ensure cluster_id is treated as categorical
        df_display = df.copy()
        df_display[group_by] = df_display[group_by].astype(str)
        
        fig = px.box(
            df_display,
            x=group_by,
            y=feature,
            color=group_by,
            title=f'📦 {feature} Distribution by {group_by}',
            points='all',
            notched=True,
            color_discrete_sequence=self.cluster_colors
        )
        
        fig.update_layout(
            xaxis_title=group_by,
            yaxis_title=feature,
            showlegend=False,
            hovermode='x unified',
            # Add more margin for better text spacing
            margin=dict(t=80, b=80, l=80, r=80),
            xaxis=dict(
                tickfont=dict(size=10)  # Reduce tick font size
            ),
            yaxis=dict(
                tickfont=dict(size=10)  # Reduce tick font size
            )
        )
        
        # Reduce point size to prevent overlap
        fig.update_traces(
            marker=dict(size=4, opacity=0.7),
            jitter=0.3  # Add jitter to spread points
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig, height=500)  # Increased height
        
        return fig
        
    def create_cluster_composition_chart(self, cluster_composition):
        """Create stacked bar chart showing cluster composition with better spacing"""
        clusters = list(cluster_composition.keys())
        
        # Prepare data for stacked bar chart
        study_data = {
            '📚 High Study': [cluster_composition[cluster]['high_study'] for cluster in clusters],
            '📖 Medium Study': [cluster_composition[cluster]['medium_study'] for cluster in clusters],
            '📝 Low Study': [cluster_composition[cluster]['low_study'] for cluster in clusters]
        }
        
        # Create a more compact color scheme
        colors = [self.dark_theme['success'], self.dark_theme['warning'], self.dark_theme['danger']]
        
        fig = go.Figure()
        
        # Add stacked bars with improved settings
        fig.add_trace(go.Bar(
            name='📚 High Study',
            x=clusters,
            y=study_data['📚 High Study'],
            marker_color=colors[0],
            marker_line_color='white',
            marker_line_width=0.5,
            hovertemplate='<b>%{x}</b><br>High Study: %{y} students<extra></extra>',
            text=study_data['📚 High Study'],
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=10, color='white')
        ))
        
        fig.add_trace(go.Bar(
            name='📖 Medium Study',
            x=clusters,
            y=study_data['📖 Medium Study'],
            marker_color=colors[1],
            marker_line_color='white',
            marker_line_width=0.5,
            hovertemplate='<b>%{x}</b><br>Medium Study: %{y} students<extra></extra>',
            text=study_data['📖 Medium Study'],
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=10, color='white')
        ))
        
        fig.add_trace(go.Bar(
            name='📝 Low Study',
            x=clusters,
            y=study_data['📝 Low Study'],
            marker_color=colors[2],
            marker_line_color='white',
            marker_line_width=0.5,
            hovertemplate='<b>%{x}</b><br>Low Study: %{y} students<extra></extra>',
            text=study_data['📝 Low Study'],
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=10, color='white')
        ))
        
        # Calculate total students per cluster for better x-axis labeling
        totals = []
        for cluster in clusters:
            total = (cluster_composition[cluster]['high_study'] + 
                    cluster_composition[cluster]['medium_study'] + 
                    cluster_composition[cluster]['low_study'])
            totals.append(total)
        
        # Update layout with better spacing
        fig.update_layout(
            title='📊 Cluster Composition by Study Time',
            xaxis_title='Cluster',
            yaxis_title='Number of Students',
            barmode='stack',
            showlegend=True,
            legend=dict(
                # title="Study Time Level",
                title_font=dict(
                    color=self.dark_theme['text'],
                    size=11,  # Reduced size
                    family="Inter, Arial, sans-serif"
                ),
                font=dict(
                    color=self.dark_theme['text_secondary'],
                    size=10,  # Reduced size
                    family="Inter, Arial, sans-serif"
                ),
                orientation="h",  # Horizontal legend
                yanchor="bottom",
                y=1.02,  # Position above the chart
                xanchor="center",
                x=0.5,
                bgcolor='rgba(30, 41, 59, 0.7)',
                bordercolor=self.dark_theme['border'],
                borderwidth=1,
                traceorder='normal',
                itemwidth=30  # Fixed width for legend items
            ),
            # Add more margin for better spacing
            margin=dict(t=100, b=80, l=80, r=80),
            # Add padding to prevent text overlap
            bargap=0.2,  # Space between bars
            bargroupgap=0.1,  # Space between bar groups
            # Add annotations for total students
            annotations=[
                dict(
                    x=cluster_idx,
                    y=totals[cluster_idx] + max(totals) * 0.05,  # Position above bars
                    text=f"Total: {totals[cluster_idx]}",
                    showarrow=False,
                    font=dict(size=10, color=self.dark_theme['text_secondary']),
                    xref="x",
                    yref="y"
                ) for cluster_idx in range(len(clusters))
            ],
            xaxis=dict(
                tickfont=dict(size=11),
                title_font=dict(size=12)
            ),
            yaxis=dict(
                tickfont=dict(size=10),
                title_font=dict(size=12),
                gridcolor=self.dark_theme['grid']
            )
        )
        
        # Apply dark theme with increased height
        fig = self._apply_dark_theme(fig, height=500)
        
        # Override some theme settings for better text visibility
        fig.update_layout(
            plot_bgcolor=self.dark_theme['plot_bg'],
            paper_bgcolor=self.dark_theme['paper_bg'],
        )
        
        return fig
    
    # ============ NEW PREDICTION VISUALIZATION METHODS ============
    
    def create_prediction_probability_chart(self, predictions_df):
        """Create visualization of prediction probabilities"""
        fig = px.histogram(
            predictions_df,
            x='probability_pass',
            nbins=20,
            title='📊 Prediction Probability Distribution',
            color_discrete_sequence=[self.dark_theme['primary']],
            opacity=0.85,
            labels={'probability_pass': 'Pass Probability', 'count': 'Number of Students'}
        )
        
        # Add risk threshold lines
        fig.add_vline(
            x=0.3, 
            line_dash="dash", 
            line_color=self.dark_theme['danger'],
            line_width=2,
            annotation_text="High Risk (<30%)",
            annotation_position="top right",
            annotation_font=dict(
                color=self.dark_theme['text'],
                size=10,
                family="Inter, Arial, sans-serif"
            ),
            annotation_bgcolor=self.dark_theme['paper_bg'],
            annotation_bordercolor=self.dark_theme['border']
        )
        
        fig.add_vline(
            x=0.5, 
            line_dash="dash", 
            line_color=self.dark_theme['warning'],
            line_width=2,
            annotation_text="Moderate Risk (30-50%)",
            annotation_position="top",
            annotation_font=dict(
                color=self.dark_theme['text'],
                size=10,
                family="Inter, Arial, sans-serif"
            ),
            annotation_bgcolor=self.dark_theme['paper_bg'],
            annotation_bordercolor=self.dark_theme['border']
        )
        
        fig.add_vline(
            x=0.7, 
            line_dash="dash", 
            line_color=self.dark_theme['success'],
            line_width=2,
            annotation_text="Low Risk (>50%)",
            annotation_position="top left",
            annotation_font=dict(
                color=self.dark_theme['text'],
                size=10,
                family="Inter, Arial, sans-serif"
            ),
            annotation_bgcolor=self.dark_theme['paper_bg'],
            annotation_bordercolor=self.dark_theme['border']
        )
        
        fig = self._apply_dark_theme(fig, height=450)
        return fig
    
    def create_prediction_scatter(self, predictions_df, x_feature='probability_pass', y_feature='confidence'):
        """Create scatter plot of predictions"""
        fig = px.scatter(
            predictions_df,
            x=x_feature,
            y=y_feature,
            color='predicted_class',
            title='📍 Prediction Confidence vs Probability',
            color_discrete_map={
                'Pass': self.dark_theme['success'],
                'Fail': self.dark_theme['danger']
            },
            hover_data=['student_id', 'predicted_class', 'probability_pass', 'confidence'],
            opacity=0.7
        )
        
        fig.update_layout(
            xaxis_title='Pass Probability',
            yaxis_title='Confidence Score',
            legend=dict(
                title="Prediction",
                title_font=dict(
                    color=self.dark_theme['text'],
                    size=12,
                    family="Inter, Arial, sans-serif"
                ),
                font=dict(
                    color=self.dark_theme['text_secondary'],
                    size=11,
                    family="Inter, Arial, sans-serif"
                )
            )
        )
        
        fig = self._apply_dark_theme(fig, height=500)
        return fig
    
    # ============ NEW DATA OVERVIEW VISUALIZATION METHODS ============
    
    def create_missing_values_chart(self, df, sample_rows=50):
        """Create missing values heatmap for data overview"""
        # Get sample of data for heatmap
        sample_df = df.head(sample_rows)
        
        # Create binary matrix (1 for missing, 0 for present)
        missing_matrix = sample_df.isnull().astype(int).values.T
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=missing_matrix,
            x=list(range(sample_rows)),
            y=list(sample_df.columns),
            colorscale='Reds',
            showscale=True,
            zmin=0,
            zmax=1,
            hovertemplate='<b>%{y}</b><br>Row %{x}<br>Missing: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Missing Data Heatmap (First {sample_rows} Rows)',
            xaxis_title='Row Index',
            yaxis_title='Features',
            height=400
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig, height=400)
        
        return fig
    
    def create_data_types_chart(self, df):
        """Create bar chart showing distribution of data types"""
        # Count data types
        dtype_counts = df.dtypes.value_counts()
        dtype_df = pd.DataFrame({
            'Data Type': dtype_counts.index.astype(str),
            'Count': dtype_counts.values
        })
        
        # Create bar chart
        fig = px.bar(
            dtype_df,
            x='Data Type',
            y='Count',
            title='📊 Data Type Distribution',
            color='Count',
            color_continuous_scale='Teal',
            text='Count'
        )
        
        fig.update_traces(
            textposition='inside',  # Changed from 'outside'
            insidetextanchor='middle',
            textfont=dict(
                color='white',
                size=11,  # Reduced
                family="Inter, Arial, sans-serif"
            ),
            marker_line_color='white',
            marker_line_width=1
        )
        
        fig.update_layout(
            xaxis_title='Data Type',
            yaxis_title='Number of Columns'
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig, height=400)
        
        return fig
    
    def create_outlier_analysis_chart(self, df, columns=None, max_cols=10):
        """Create visualization for outlier analysis"""
        if columns is None:
            # Use numeric columns, limit to max_cols
            columns = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
        
        # Calculate IQR and outliers for each column
        outlier_data = []
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_percentage = (len(outliers) / len(df)) * 100
            
            outlier_data.append({
                'Feature': col,
                'Outliers': len(outliers),
                'Outlier %': outlier_percentage,
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound
            })
        
        outlier_df = pd.DataFrame(outlier_data)
        
        # Create bar chart
        fig = px.bar(
            outlier_df,
            x='Feature',
            y='Outlier %',
            title='⚠️ Outlier Percentage by Feature',
            color='Outlier %',
            color_continuous_scale='YlOrRd',
            text='Outlier %',
            hover_data=['Outliers', 'Lower Bound', 'Upper Bound']
        )
        
        fig.update_traces(
            texttemplate='%{y:.1f}%',
            textposition='inside',  # Changed from 'outside'
            insidetextanchor='middle',
            textfont=dict(
                color='white',
                size=10,
                family="Inter, Arial, sans-serif"
            )
        )
        
        fig.update_layout(
            xaxis_title='Feature',
            yaxis_title='Outlier Percentage (%)',
            height=500
        )
        
        # Add threshold lines
        fig.add_hline(
            y=5,
            line_dash="dash",
            line_color=self.dark_theme['warning'],
            annotation_text="5% Threshold",
            annotation_position="right",
            annotation_font=dict(
                color=self.dark_theme['warning'],
                size=10
            )
        )
        
        fig.add_hline(
            y=10,
            line_dash="dash",
            line_color=self.dark_theme['danger'],
            annotation_text="10% Threshold",
            annotation_position="right",
            annotation_font=dict(
                color=self.dark_theme['danger'],
                size=10
            )
        )
        
        # Apply dark theme
        fig = self._apply_dark_theme(fig, height=500)
        
        return fig