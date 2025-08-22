# traffic_accident_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TrafficAccidentAnalyzer:
    def __init__(self):
        self.df = None
        self.gdf = None
        
    def generate_sample_data(self, n_samples=5000):
        """Generate sample traffic accident data"""
        print("Generating sample traffic accident data...")
        
        # Sample data parameters
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
        
        road_types = ['Highway', 'Local Street', 'Intersection', 'Freeway', 
                     'Residential', 'Commercial', 'Rural Road', 'Bridge']
        
        weather_conditions = ['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy', 
                             'Windy', 'Ice', 'Sleet', 'Heavy Rain']
        
        light_conditions = ['Daylight', 'Dark - Lighted', 'Dark - Not Lighted', 
                          'Dawn', 'Dusk', 'Dark - Unknown Lighting']
        
        accident_types = ['Rear-end', 'Head-on', 'Side-swipe', 'Angle', 
                         'Single Vehicle', 'Pedestrian', 'Bicycle', 'Parked Vehicle']
        
        severity_levels = ['Property Damage', 'Minor Injury', 'Serious Injury', 'Fatal']
        
        # Generate sample data
        data = []
        np.random.seed(42)
        
        for i in range(n_samples):
            # Generate random location coordinates within US bounds
            lat = np.random.uniform(24.0, 50.0)
            lon = np.random.uniform(-125.0, -66.0)
            city = np.random.choice(cities)
            
            # Generate timestamp (last 2 years)
            days_ago = np.random.randint(0, 730)
            hours = np.random.randint(0, 24)
            minutes = np.random.randint(0, 60)
            timestamp = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes)
            
            # Generate time-based patterns
            hour_of_day = timestamp.hour
            day_of_week = timestamp.weekday()
            month = timestamp.month
            
            # Generate correlated features
            if month in [12, 1, 2]:  # Winter months
                weather = np.random.choice(['Snow', 'Ice', 'Clear', 'Cloudy'], p=[0.3, 0.2, 0.3, 0.2])
            elif month in [6, 7, 8]:  # Summer months
                weather = np.random.choice(['Clear', 'Rain', 'Cloudy'], p=[0.6, 0.2, 0.2])
            else:
                weather = np.random.choice(weather_conditions)
            
            # Time-based patterns
            if 6 <= hour_of_day <= 9 or 16 <= hour_of_day <= 19:  # Rush hours
                road_type = np.random.choice(['Highway', 'Freeway', 'Intersection'], p=[0.4, 0.3, 0.3])
                severity = np.random.choice(severity_levels, p=[0.3, 0.4, 0.2, 0.1])
            else:
                road_type = np.random.choice(road_types)
                severity = np.random.choice(severity_levels, p=[0.5, 0.3, 0.15, 0.05])
            
            # Weather impact on accident type
            if weather in ['Rain', 'Snow', 'Ice']:
                accident_type = np.random.choice(['Rear-end', 'Single Vehicle', 'Angle'], p=[0.5, 0.3, 0.2])
            else:
                accident_type = np.random.choice(accident_types)
            
            # Light conditions based on time
            if 6 <= hour_of_day <= 18:
                light_condition = 'Daylight'
            else:
                light_condition = np.random.choice(['Dark - Lighted', 'Dark - Not Lighted', 'Dark - Unknown Lighting'])
            
            data.append({
                'accident_id': i + 1,
                'timestamp': timestamp,
                'city': city,
                'latitude': lat,
                'longitude': lon,
                'road_type': road_type,
                'weather_condition': weather,
                'light_condition': light_condition,
                'accident_type': accident_type,
                'severity': severity,
                'num_vehicles': np.random.randint(1, 5),
                'num_injuries': np.random.poisson(0.5),
                'num_fatalities': np.random.poisson(0.05),
                'speed_limit': np.random.choice([25, 35, 45, 55, 65, 70]),
                'road_condition': np.random.choice(['Dry', 'Wet', 'Snowy', 'Icy']),
                'visibility': np.random.uniform(0.1, 10.0)  # miles
            })
        
        self.df = pd.DataFrame(data)
        
        # Create datetime features
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['year'] = self.df['timestamp'].dt.year
        
        return self.df
    
    def create_geodataframe(self):
        """Convert DataFrame to GeoDataFrame"""
        geometry = [Point(xy) for xy in zip(self.df['longitude'], self.df['latitude'])]
        self.gdf = gpd.GeoDataFrame(self.df, geometry=geometry, crs="EPSG:4326")
        return self.gdf
    
    def analyze_time_patterns(self):
        """Analyze temporal patterns in accidents"""
        print("Analyzing temporal patterns...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Accidents by Hour of Day',
                'Accidents by Day of Week',
                'Accidents by Month',
                'Accidents by Severity Over Time'
            )
        )
        
        # Hourly distribution
        hourly_counts = self.df['hour'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=hourly_counts.index, y=hourly_counts.values, name='By Hour'),
            row=1, col=1
        )
        
        # Daily distribution
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_counts = self.df['day_of_week'].value_counts().sort_index()
        daily_counts.index = day_names
        fig.add_trace(
            go.Bar(x=daily_counts.index, y=daily_counts.values, name='By Day'),
            row=1, col=2
        )
        
        # Monthly distribution
        monthly_counts = self.df['month'].value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig.add_trace(
            go.Bar(x=month_names, y=monthly_counts.values, name='By Month'),
            row=2, col=1
        )
        
        # Severity over time
        severity_time = self.df.groupby(['month', 'severity']).size().unstack()
        for severity in severity_time.columns:
            fig.add_trace(
                go.Scatter(x=month_names, y=severity_time[severity], 
                          name=severity, mode='lines+markers'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, width=1000, 
                         title_text="Temporal Patterns of Traffic Accidents")
        fig.show()
    
    def analyze_weather_road_conditions(self):
        """Analyze impact of weather and road conditions"""
        print("Analyzing weather and road conditions...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Accidents by Weather Condition',
                'Accidents by Road Condition',
                'Weather vs Accident Severity',
                'Road Type vs Accident Type'
            )
        )
        
        # Weather conditions
        weather_counts = self.df['weather_condition'].value_counts()
        fig.add_trace(
            go.Bar(x=weather_counts.index, y=weather_counts.values, name='Weather'),
            row=1, col=1
        )
        
        # Road conditions
        road_counts = self.df['road_condition'].value_counts()
        fig.add_trace(
            go.Bar(x=road_counts.index, y=road_counts.values, name='Road Condition'),
            row=1, col=2
        )
        
        # Weather vs Severity
        weather_severity = pd.crosstab(self.df['weather_condition'], self.df['severity'])
        for severity in weather_severity.columns:
            fig.add_trace(
                go.Bar(x=weather_severity.index, y=weather_severity[severity], 
                      name=severity, showlegend=False),
                row=2, col=1
            )
        
        # Road Type vs Accident Type
        road_accident = pd.crosstab(self.df['road_type'], self.df['accident_type'])
        for acc_type in road_accident.columns:
            fig.add_trace(
                go.Bar(x=road_accident.index, y=road_accident[acc_type], 
                      name=acc_type, showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(height=800, width=1000, 
                         title_text="Weather and Road Condition Analysis")
        fig.show()
    
    def create_heatmap(self):
        """Create accident heatmap"""
        print("Creating accident heatmap...")
        
        # Create base map
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        
        # Prepare heatmap data
        heat_data = [[row['latitude'], row['longitude']] for index, row in self.df.iterrows()]
        
        # Add heatmap
        HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)
        
        # Save map
        m.save('accident_heatmap.html')
        print("Heatmap saved as 'accident_heatmap.html'")
        
        return m
    
    def analyze_severity_factors(self):
        """Analyze factors contributing to accident severity"""
        print("Analyzing severity factors...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Factors Contributing to Accident Severity', fontsize=16)
        
        # Weather vs Severity
        weather_severity = pd.crosstab(self.df['weather_condition'], self.df['severity'], normalize='index')
        weather_severity.plot(kind='bar', stacked=True, ax=axes[0, 0])
        axes[0, 0].set_title('Weather Impact on Severity')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Road Type vs Severity
        road_severity = pd.crosstab(self.df['road_type'], self.df['severity'], normalize='index')
        road_severity.plot(kind='bar', stacked=True, ax=axes[0, 1])
        axes[0, 1].set_title('Road Type Impact on Severity')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Time of Day vs Severity
        hour_severity = pd.crosstab(self.df['hour'], self.df['severity'], normalize='index')
        hour_severity.plot(kind='area', stacked=True, ax=axes[0, 2])
        axes[0, 2].set_title('Time of Day Impact on Severity')
        
        # Light Condition vs Severity
        light_severity = pd.crosstab(self.df['light_condition'], self.df['severity'], normalize='index')
        light_severity.plot(kind='bar', stacked=True, ax=axes[1, 0])
        axes[1, 0].set_title('Light Condition Impact on Severity')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Speed Limit vs Severity
        speed_severity = self.df.groupby('speed_limit')['severity'].value_counts(normalize=True).unstack()
        speed_severity.plot(kind='bar', stacked=True, ax=axes[1, 1])
        axes[1, 1].set_title('Speed Limit Impact on Severity')
        
        # Accident Type vs Severity
        accident_severity = pd.crosstab(self.df['accident_type'], self.df['severity'], normalize='index')
        accident_severity.plot(kind='bar', stacked=True, ax=axes[1, 2])
        axes[1, 2].set_title('Accident Type Impact on Severity')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def create_correlation_analysis(self):
        """Create correlation analysis between factors"""
        print("Performing correlation analysis...")
        
        # Convert categorical variables to numerical for correlation
        numeric_df = self.df.copy()
        
        # Encode categorical variables
        severity_map = {'Property Damage': 1, 'Minor Injury': 2, 'Serious Injury': 3, 'Fatal': 4}
        numeric_df['severity_numeric'] = numeric_df['severity'].map(severity_map)
        
        # Select numerical features for correlation
        numerical_features = ['hour', 'day_of_week', 'month', 'num_vehicles', 
                             'num_injuries', 'num_fatalities', 'speed_limit', 
                             'visibility', 'severity_numeric']
        
        correlation_matrix = numeric_df[numerical_features].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': .8})
        plt.title('Correlation Matrix of Accident Factors')
        plt.tight_layout()
        plt.show()
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n" + "="*70)
        print("TRAFFIC ACCIDENT ANALYSIS INSIGHTS REPORT")
        print("="*70)
        
        total_accidents = len(self.df)
        fatal_accidents = len(self.df[self.df['severity'] == 'Fatal'])
        injury_accidents = len(self.df[self.df['num_injuries'] > 0])
        
        print(f"\nOverall Statistics:")
        print(f"Total Accidents Analyzed: {total_accidents:,}")
        print(f"Fatal Accidents: {fatal_accidents} ({fatal_accidents/total_accidents:.1%})")
        print(f"Injury Accidents: {injury_accidents} ({injury_accidents/total_accidents:.1%})")
        
        # Most dangerous times
        print(f"\nTemporal Patterns:")
        peak_hour = self.df['hour'].value_counts().idxmax()
        peak_day = self.df['day_of_week'].value_counts().idxmax()
        peak_month = self.df['month'].value_counts().idxmax()
        
        print(f"Peak Accident Hour: {peak_hour}:00")
        print(f"Peak Accident Day: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][peak_day]}")
        print(f"Peak Accident Month: {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][peak_month-1]}")
        
        # Most dangerous conditions
        print(f"\nHigh-Risk Conditions:")
        # Add severity numeric mapping for analysis
        severity_map = {'Property Damage': 1, 'Minor Injury': 2, 'Serious Injury': 3, 'Fatal': 4}
        self.df['severity_numeric'] = self.df['severity'].map(severity_map)
        
        worst_weather = self.df.groupby('weather_condition')['severity_numeric'].mean().idxmax()
        worst_road = self.df.groupby('road_condition')['severity_numeric'].mean().idxmax()
        worst_light = self.df.groupby('light_condition')['severity_numeric'].mean().idxmax()
        
        print(f"Most Dangerous Weather: {worst_weather}")
        print(f"Most Dangerous Road Condition: {worst_road}")
        print(f"Most Dangerous Light Condition: {worst_light}")
        
        # High-risk locations
        print(f"\nGeographic Patterns:")
        city_counts = self.df['city'].value_counts()
        print(f"City with Most Accidents: {city_counts.index[0]} ({city_counts.iloc[0]} accidents)")
        
        # Severity factors
        print(f"\nSeverity Factors:")
        high_severity = self.df[self.df['severity'].isin(['Serious Injury', 'Fatal'])]
        if len(high_severity) > 0:
            common_factors = {
                'weather': high_severity['weather_condition'].value_counts().index[0],
                'road_type': high_severity['road_type'].value_counts().index[0],
                'time': f"{high_severity['hour'].value_counts().index[0]}:00"
            }
            print(f"Common factors in severe accidents: {common_factors}")
        
        print("="*70)

def main():
    """Main function to run the traffic accident analysis"""
    print("Traffic Accident Data Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = TrafficAccidentAnalyzer()
    
    # Generate sample data
    df = analyzer.generate_sample_data(3000)
    print(f"Generated {len(df)} sample accident records")
    
    # Create GeoDataFrame
    gdf = analyzer.create_geodataframe()
    
    # Show sample data
    print("\nSample Data:")
    print(df[['timestamp', 'city', 'road_type', 'weather_condition', 'severity']].head())
    
    # Perform analyses
    analyzer.analyze_time_patterns()
    analyzer.analyze_weather_road_conditions()
    analyzer.analyze_severity_factors()
    analyzer.create_correlation_analysis()
    
    # Generate heatmap
    analyzer.create_heatmap()
    
    # Generate insights report
    analyzer.generate_insights_report()
    
    print("\nAnalysis complete! Check the generated visualizations and heatmap.")

if __name__ == "__main__":
    main()