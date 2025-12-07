import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="Edge Device Monitoring", layout="wide")

def load_telemetry_data():
    """Load all telemetry data from devices."""
    telemetry_dir = Path("data/telemetry")
    
    st.sidebar.info(f"Looking for telemetry in: {telemetry_dir.absolute()}")
    
    if not telemetry_dir.exists():
        st.sidebar.error(f"Directory doesn't exist: {telemetry_dir.absolute()}")
        return pd.DataFrame()
    
    all_data = []
    files = list(telemetry_dir.glob("*_telemetry.jsonl"))
    st.sidebar.info(f"Found {len(files)} telemetry files")
    
    for telemetry_file in files:
        try:
            with open(telemetry_file) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        st.sidebar.warning(f"Bad JSON in {telemetry_file.name} line {line_num}")
                        continue
        except Exception as e:
            st.sidebar.error(f"Error reading {telemetry_file.name}: {e}")
            continue
    
    st.sidebar.success(f"Loaded {len(all_data)} total records")
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def load_device_stats():
    """Load device statistics."""
    stats_dir = Path("data/device_stats")
    
    if not stats_dir.exists():
        st.sidebar.warning(f"Stats directory doesn't exist: {stats_dir.absolute()}")
        return []
    
    stats = []
    files = list(stats_dir.glob("*_stats.json"))
    st.sidebar.info(f"Found {len(files)} stats files")
    
    for stats_file in files:
        try:
            with open(stats_file) as f:
                stats.append(json.load(f))
        except Exception as e:
            st.sidebar.error(f"Error reading {stats_file.name}: {e}")
            continue
    
    return stats

def main():
    st.title("🤖 Edge Device Fleet Monitoring")
    st.markdown("Real-time monitoring of edge device inference and performance")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 5)
    
    if st.sidebar.button("Refresh Now"):
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Debug Info**")
    
    # Load data
    df = load_telemetry_data()
    device_stats = load_device_stats()
    
    if df.empty:
        st.warning("⚠️ No telemetry data available!")
        st.info("Run the edge simulator to generate data:")
        st.code("python src/deployment/edge_simulator.py --num-devices 3 --duration 60")
        
        # Show what we have
        st.markdown("### Checking data directories...")
        telemetry_dir = Path("data/telemetry")
        if telemetry_dir.exists():
            files = list(telemetry_dir.glob("*"))
            st.write(f"Files in {telemetry_dir}: {len(files)}")
            if files:
                st.write("Files found:", [f.name for f in files[:5]])
        else:
            st.error(f"Telemetry directory doesn't exist!")
        return
    
    # Overview metrics
    st.header("📊 Fleet Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Active Devices", df['device_id'].nunique())
    
    with col2:
        st.metric("Total Inferences", len(df))
    
    with col3:
        accuracy = (df['correct'].sum() / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Accuracy", f"{accuracy:.1f}%")
    
    with col4:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    with col5:
        avg_latency = df['inference_time_ms'].mean()
        st.metric("Avg Latency", f"{avg_latency:.2f} ms")
    
    # Time series charts
    st.header("📈 Performance Metrics Over Time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Inference count over time
        df_time = df.set_index('timestamp').resample('10S').size().reset_index(name='count')
        fig = px.line(df_time, x='timestamp', y='count', 
                     title='Inferences per 10 seconds',
                     labels={'count': 'Number of Inferences', 'timestamp': 'Time'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average confidence over time
        df_conf = df.set_index('timestamp')['confidence'].resample('10S').mean().reset_index()
        fig = px.line(df_conf, x='timestamp', y='confidence',
                     title='Average Confidence Over Time',
                     labels={'confidence': 'Confidence', 'timestamp': 'Time'})
        fig.add_hline(y=0.25, line_dash="dash", line_color="red", 
                     annotation_text="Min Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    # Per-device metrics
    st.header("🔧 Per-Device Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy by device
        device_accuracy = df.groupby('device_id')['correct'].mean() * 100
        fig = px.bar(device_accuracy, 
                    title='Accuracy by Device',
                    labels={'value': 'Accuracy (%)', 'device_id': 'Device'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Inference time by device
        device_latency = df.groupby('device_id')['inference_time_ms'].mean()
        fig = px.bar(device_latency,
                    title='Average Inference Time by Device',
                    labels={'value': 'Time (ms)', 'device_id': 'Device'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Device statistics table
    if device_stats:
        st.header("📋 Device Statistics")
        
        stats_df = pd.DataFrame(device_stats)
        
        # Select and rename columns if they exist
        display_cols = []
        col_mapping = {
            'device_id': 'Device ID',
            'total_inferences': 'Inferences',
            'positive_detections': 'Detections',
            'data_transmissions': 'Transmissions',
            'avg_confidence': 'Avg Confidence',
            'avg_inference_time_ms': 'Avg Latency (ms)',
            'positive_rate': 'Detection Rate'
        }
        
        for old_col, new_col in col_mapping.items():
            if old_col in stats_df.columns:
                display_cols.append(old_col)
        
        if display_cols:
            stats_display = stats_df[display_cols].copy()
            stats_display = stats_display.rename(columns=col_mapping)
            st.dataframe(stats_display, use_container_width=True)
        else:
            st.warning("Stats data has unexpected format")
            st.json(device_stats[0])
    
    # Recent telemetry
    st.header("📡 Recent Telemetry")
    
    recent_df = df.sort_values('timestamp', ascending=False).head(20)
    display_cols = ['timestamp', 'device_id', 'prediction', 'true_label', 
                    'confidence', 'inference_time_ms', 'correct']
    
    st.dataframe(recent_df[display_cols], use_container_width=True)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()