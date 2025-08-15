import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
from io import BytesIO
import hashlib

def whittaker_smooth(y, lmbd, d=2):
    m = len(y)
    E = eye(m, format='csc')
    D = E[1:, :] - E[:-1, :]
    for i in range(d - 1):
        D = D[1:, :] - D[:-1, :]
    W = diags(np.ones(m), 0, shape=(m, m))
    A = csc_matrix(W) + lmbd * D.T @ D
    z = spsolve(A, y)
    return z

# THÊM CACHING CHO VIỆC ĐỌC FILE EXCEL
@st.cache_data(show_spinner="Loading Excel file...")
def cached_read_excel(file_bytes, file_name, sheet_name=None):
    """
    Cache Excel file reading to avoid re-reading on parameter changes.
    Uses file content hash as cache key.
    """
    # Create a file-like object from bytes
    file_buffer = BytesIO(file_bytes)
    
    try:
        if sheet_name:
            df = pd.read_excel(file_buffer, sheet_name=sheet_name, engine='openpyxl')
        else:
            df = pd.read_excel(file_buffer, engine='openpyxl')
        
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner="Getting Excel sheet names...")
def cached_get_sheet_names(file_bytes, file_name):
    """Cache sheet names extraction."""
    file_buffer = BytesIO(file_bytes)
    try:
        excel_file = pd.ExcelFile(file_buffer)
        return excel_file.sheet_names, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def cached_convert_afm_data(file_bytes, file_name, sheet_name=None):
    """
    Cache the entire AFM data conversion process.
    This prevents re-processing on parameter changes.
    """
    df, error = cached_read_excel(file_bytes, file_name, sheet_name)
    
    if df is None:
        return None, error
    
    # Check if required columns exist
    required_columns = ['Time (s)', 'Load (N)', 'Displacement (mm)']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        available_columns = list(df.columns)
        return None, f"Missing columns: {missing_columns}. Available columns: {available_columns}"
    
    # Process the data
    processed_df = pd.DataFrame({
        'Time': df['Time (s)'],
        'Force': df['Load (N)'],
        'Indentation': df['Displacement (mm)'] / 1000  # Convert mm to m
    })
    
    return processed_df, None

def convert_afm_data_to_dataframe(uploaded_file, sheet_name=None):
    """
    Wrapper function that uses caching for Excel files.
    """
    try:
        # For Excel files, use caching
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            # Get file bytes for caching
            file_bytes = uploaded_file.read()
            # Reset file pointer for potential future reads
            uploaded_file.seek(0)
            
            return cached_convert_afm_data(file_bytes, uploaded_file.name, sheet_name)
        
        # For other file types, process normally (they're usually small)
        else:
            df = pd.read_csv(uploaded_file, delimiter=r'\s+', header=None, engine='python')
            if df.shape[1] < 3:
                return None, "File must have at least 3 columns"
            
            df.columns = ['Time', 'Force', 'Indentation']
            return df, None
            
    except Exception as e:
        return None, str(e)

def get_excel_sheet_names(uploaded_file):
    """Get all sheet names from an Excel file with caching."""
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            sheet_names, error = cached_get_sheet_names(file_bytes, uploaded_file.name)
            return sheet_names if sheet_names else None
        else:
            return None
    except Exception as e:
        return None

def create_download_data(results_df, file_format='csv'):
    """
    Create downloadable data in specified format.
    
    Args:
        results_df (pd.DataFrame): The results dataframe
        file_format (str): 'csv' or 'excel'
    
    Returns:
        bytes: File data as bytes
        str: MIME type
        str: File extension
    """
    if file_format == 'csv':
        # Create CSV data
        csv_data = results_df.to_csv(index=False)
        return csv_data.encode('utf-8'), 'text/csv', 'csv'
    
    elif file_format == 'excel':
        # Create Excel data with multiple sheets
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Main results sheet
            results_df.to_excel(writer, sheet_name='Rheological_Results', index=False)
            
            # Create summary sheet with analysis parameters

            g_prime_col = "G'"
            g_double_prime_col = "G''"

            summary_data = {
                'Parameter': [
                    'Analysis Date',
                    'Total Data Points',
                    'Frequency Range (rad/s)',
                    'G\' Range (Pa)', 
                    'G\'\' Range (Pa)',
                    'tan(δ) Range',
                    'File Format'
                ],

                'Value': [
                    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(results_df),
                    f"{results_df['Frequency'].min():.2e} - {results_df['Frequency'].max():.2e}",
                    f"{results_df[g_prime_col].min():.2e} - {results_df[g_prime_col].max():.2e}",
                    f"{results_df[g_double_prime_col].min():.2e} - {results_df[g_double_prime_col].max():.2e}",
                    f"{results_df['tan(δ)'].min():.3f} - {results_df['tan(δ)'].max():.3f}",
                    'i-Rheo Analysis'
            ]

            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Analysis_Summary', index=False)
            
            # Create log-spaced data sheet for easier plotting
            log_freq = np.logspace(np.log10(results_df['Frequency'].min()), 
                                 np.log10(results_df['Frequency'].max()), 50)
            
            # Interpolate results to log-spaced frequencies
            from scipy.interpolate import interp1d
            
            try:
                g_prime_interp = interp1d(results_df['Frequency'], results_df['G\''], 
                                        kind='linear', bounds_error=False, fill_value='extrapolate')
                g_double_prime_interp = interp1d(results_df['Frequency'], results_df['G\'\''], 
                                               kind='linear', bounds_error=False, fill_value='extrapolate')
                tan_delta_interp = interp1d(results_df['Frequency'], results_df['tan(δ)'], 
                                          kind='linear', bounds_error=False, fill_value='extrapolate')
                
                log_results_df = pd.DataFrame({
                    'Frequency': log_freq,
                    'G\'': g_prime_interp(log_freq),
                    'G\'\'': g_double_prime_interp(log_freq),
                    'tan(δ)': tan_delta_interp(log_freq)
                })
                
                log_results_df.to_excel(writer, sheet_name='Log_Spaced_Data', index=False)
            except:
                pass  # Skip if interpolation fails
        
        buffer.seek(0)
        return buffer.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'xlsx'

# --- Physics Formulas (no changes) ---
def calculate_lambda_factor(probe_type, indentation, **kwargs):
    poisson_ratio = kwargs.get('poisson_ratio', 0.5)
    indentation = np.asarray(indentation)
    indentation_safe = np.maximum(indentation, 0)
    if probe_type == "Cylindrical (Flat punch)":
        radius = kwargs.get('probe_radius', 1e-6)
        return (4 * radius / (1 - poisson_ratio)) * indentation_safe
    elif probe_type == "Spherical":
        radius = kwargs.get('probe_radius', 2.41e-6)
        return (8/3) * np.sqrt(radius) * (indentation_safe**1.5) / (1 - poisson_ratio)
    elif probe_type == "Conical":
        angle_deg = kwargs.get('cone_angle', 20.0)
        angle_rad = np.radians(angle_deg)
        return (4 * np.tan(angle_rad) / (np.pi * (1 - poisson_ratio))) * (indentation_safe**2)
    elif probe_type == "Four-sided pyramid":
        angle_deg = kwargs.get('pyramid_angle', 22.0)
        angle_rad = np.radians(angle_deg)
        return (np.tan(angle_rad) / np.sqrt(2)) * (indentation_safe**2)

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("i-Rheo_Indentation")

# THÊM THÔNG BÁO VỀ CACHE
if st.sidebar.button("🗑️ Clear Cache", help="Clear cached data if you're having issues"):
    st.cache_data.clear()
    st.success("Cache cleared! Please re-upload your file.")
    st.rerun()

# File uploader with Excel support
uploaded_file = st.file_uploader(
    "Choose a data file (.txt, .dat, .csv, .xlsx, .xls)", 
    type=["txt", "dat", "csv", "xlsx", "xls"]
)

# Excel file handling
sheet_name = None
if uploaded_file and uploaded_file.name.endswith(('.xlsx', '.xls')):
    st.subheader("📊 Excel File Processing")
    
    # THÊM LOADING INDICATOR CHO EXCEL
    with st.spinner("📂 Reading Excel file structure..."):
        sheet_names = get_excel_sheet_names(uploaded_file)
    
    if sheet_names:
        st.info(f"📋 Found {len(sheet_names)} sheet(s) in the Excel file: {', '.join(sheet_names)}")
        
        # Sheet selection
        sheet_name = st.selectbox(
            "Select the sheet containing your AFM data:",
            options=sheet_names,
            index=1 if len(sheet_names) > 1 else 0,  # Thêm dòng này
            help="Choose the sheet that contains columns: 'Time (s)', 'Load (N)', 'Displacement (mm)'"
        )
        
        # Preview option với caching
        if st.checkbox("🔍 Preview selected sheet", help="Show first few rows to verify data"):
            try:
                # Sử dụng cached function để preview
                file_bytes = uploaded_file.read()
                uploaded_file.seek(0)
                
                preview_df, error = cached_read_excel(file_bytes, uploaded_file.name, sheet_name)
                if preview_df is not None:
                    st.write("**First 5 rows of selected sheet:**")
                    st.dataframe(preview_df.head())
                    st.write(f"**Total columns:** {len(preview_df.columns)}")
                    st.write(f"**Column names:** {list(preview_df.columns)}")
                else:
                    st.error(f"Error previewing sheet: {error}")
            except Exception as e:
                st.error(f"Error previewing sheet: {e}")
    else:
        st.error("❌ Could not read sheet names from the Excel file. Please check if the file is corrupted.")
        st.stop()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Parameters")
    st.subheader("1. Probe Type")
    probe_type = st.selectbox("Select Probe Geometry", ["Spherical", "Cylindrical (Flat punch)", "Conical", "Four-sided pyramid"])
    probe_params = {'poisson_ratio': st.number_input("Poisson Ratio (ν)", value=0.5, step=0.01)}
    if probe_type == "Cylindrical (Flat punch)": probe_params['probe_radius'] = st.number_input("Cylinder Radius (m)", value=1e-6, format="%.2e")
    elif probe_type == "Spherical": probe_params['probe_radius'] = st.number_input("Sphere Radius (m)", value=2.41e-6, format="%.2e")
    elif probe_type == "Conical": probe_params['cone_angle'] = st.number_input("Cone Half-angle θ (°)", value=20.0)
    elif probe_type == "Four-sided pyramid": probe_params['pyramid_angle'] = st.number_input("Pyramid Face-to-axis Angle θ (°)", value=22.0)
    st.divider()
    st.subheader("2. Analysis Window Finder")
    quiet_period_factor = st.slider("Quiet Period Factor", 0.1, 1.0, 0.8, 0.05)
    noise_multiplier = st.number_input("Noise Multiplier", min_value=1.0, value=5.0, step=0.5)
    st.divider()
    st.subheader("3. Smoothing (Whittaker)")
    use_smoothing = st.checkbox("Apply Whittaker Smoother", value=True)
    lmbd = st.number_input("Smoothness (λ)", min_value=1.0, value=100.0, step=10.0, format="%e", disabled=not use_smoothing,
                           help="Controls the smoothness. Larger values = more smoothing. Try values like 1, 10, 100, 1000...")
    st.divider()
    st.subheader("4. Numerical Parameters")
    interpolation_kind = st.selectbox("Interpolation Method", ['linear', 'cubic'], index=0)
    num_interp = st.number_input("Interpolated Points", value=1000)
    num_plot = st.number_input("Plotting Points", value=200)

# --- Main Data Processing Block ---
if uploaded_file:
    try:
        # Handle different file types với progress bar
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            if not sheet_name:
                st.warning("⚠️ Please select a sheet from the Excel file above.")
                st.stop()
            
            # LOADING CHO EXCEL VỚI PROGRESS
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Processing Excel file...")
            progress_bar.progress(20)
            
            data, error_msg = convert_afm_data_to_dataframe(uploaded_file, sheet_name)
            progress_bar.progress(60)
            
            if data is None:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error processing Excel file: {error_msg}")
                st.write("**Expected columns:** Time (s), Load (N), Displacement (mm)")
                st.stop()
            else:
                progress_bar.progress(100)
                status_text.text(f"✅ Successfully converted Excel data! Found {len(data)} data points.")
                progress_bar.empty()
                status_text.empty()
        else:
            # Handle TXT, DAT, CSV files (original logic)
            data = pd.read_csv(uploaded_file, delimiter=r'\s+', header=None, engine='python')
            if data.shape[1] < 3: 
                st.error("File must have at least 3 columns."); 
                st.stop()
            
            data.columns = ['Time', 'Force', 'Indentation']
        
        # Common data processing for all file types
        data = data.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
        
        if len(data) == 0:
            st.error("❌ No valid numeric data found in the file.")
            st.stop()
        indentation_sign = "Negative (δ < 0)"
        column1, column2, column3 = data['Time'].values, data['Force'].values, data['Indentation'].values
        column3_processed = np.abs(column3) if indentation_sign == "Negative (δ < 0)" else column3

        # Automatic analysis window finder logic
        peakindex = np.argmax(column2)
        diff_values = np.diff(column3_processed)
        idx_fastest_indent = np.argmin(diff_values)
        quiet_period_end_idx = int(idx_fastest_indent * quiet_period_factor)
        if quiet_period_end_idx < 20: quiet_period_end_idx = min(20, len(diff_values))
        noise_level = np.std(diff_values[:quiet_period_end_idx]) if quiet_period_end_idx > 0 else 1e-12
        STATIONARY_THRESHOLD = noise_level * noise_multiplier
        endindex = -1
        for i in range(peakindex, len(diff_values)):
            if np.abs(diff_values[i]) > STATIONARY_THRESHOLD:
                endindex = i; break
        if endindex == -1: endindex = len(column1) - 1
        if peakindex >= endindex: st.error(f"Analysis Error: Could not find a valid relaxation phase. Adjust 'Noise Multiplier'."); st.stop()

        # Store automatic detection results
        auto_start_idx = peakindex
        auto_end_idx = endindex

        # --- NEW: Manual Analysis Range Control ---
        st.subheader("📊 Analysis Range Control")
        
        # Create columns for the controls
        col_auto, col_manual = st.columns([1, 2])
        
        with col_auto:
            st.info(f"**Auto-detected range:**\n- Start: Index {auto_start_idx} (t={column1[auto_start_idx]:.4f}s)\n- End: Index {auto_end_idx} (t={column1[auto_end_idx]:.4f}s)\n- Points: {auto_end_idx - auto_start_idx + 1}")
        
        with col_manual:
            st.write("**Manual Adjustment:**")
            
            # Calculate extended bounds for sliders (much larger range)
            total_points = len(column1)
            
            # For start index: allow much larger backward extension
            backward_extension = min(auto_start_idx, total_points // 10)  # Up to 10% of total data backward
            forward_extension = min(total_points - auto_start_idx, total_points // 20)  # Up to 5% forward
            
            min_start = max(0, auto_start_idx - backward_extension)
            max_start = min(total_points - 10, auto_start_idx + forward_extension)
            
            # For end index: allow much larger forward extension
            backward_end_extension = min(auto_end_idx - auto_start_idx, total_points // 20)  # Up to 5% backward from auto end
            forward_end_extension = min(total_points - auto_end_idx, total_points // 5)  # Up to 20% of total data forward
            
            min_end = max(auto_start_idx + 10, auto_end_idx - backward_end_extension)
            max_end = min(total_points - 1, auto_end_idx + forward_end_extension)
            
            # Add extension range selector
            st.write("**Extension Range Control:**")
            extension_factor = st.selectbox(
                "Choose extension range",
                options=[1, 2, 5, 10, 20],
                index=2,
                help="Multiplier for how much you can extend the analysis range"
            )
            
            # Recalculate bounds with extension factor
            min_start = max(0, auto_start_idx - backward_extension * extension_factor)
            max_start = min(total_points - 10, auto_start_idx + forward_extension * extension_factor)
            min_end = max(auto_start_idx + 10, auto_end_idx - backward_end_extension * extension_factor)
            max_end = min(total_points - 1, auto_end_idx + forward_end_extension * extension_factor)
            
        # Create two main columns for balanced layout
        col_left_info, col_right_adjustment = st.columns([1, 1])

        with col_left_info:
            st.write("**📊 Current Analysis Range:**")
            st.write(f"- **Auto Start:** {auto_start_idx:,} (t={column1[auto_start_idx]:.4f}s)")
            st.write(f"- **Auto End:** {auto_end_idx:,} (t={column1[auto_end_idx]:.4f}s)")
            st.write(f"- **Auto Points:** {auto_end_idx - auto_start_idx + 1:,}")
            
            st.write("**🎯 Available Ranges:**")
            st.write(f"- **Start Range:** {min_start:,} to {max_start:,}")
            st.write(f"- **End Range:** {min_end:,} to {max_end:,}")
            
            # Extension range info
            st.write(f"**📏 Extension Factor:** {extension_factor}x")

        with col_right_adjustment:
            st.write("**🎯 Quick Offset Adjustment:**")
            
            # Offset inputs in sub-columns
            subcol1, subcol2 = st.columns(2)
            
            with subcol1:
                start_offset = st.number_input(
                    "Start Offset",
                    value=0,
                    step=1,
                    help="Add/subtract from auto start index"
                )
                calculated_start = auto_start_idx + start_offset
                calculated_start = max(min_start, min(max_start, calculated_start))
                
            with subcol2:
                end_offset = st.number_input(
                    "End Offset", 
                    value=0,
                    step=1,
                    help="Add/subtract from auto end index"
                )
                calculated_end = auto_end_idx + end_offset
                min_allowed_end = max(min_end, calculated_start + 10)
                calculated_end = max(min_allowed_end, min(max_end, calculated_end))
            
            # Show calculations
            st.write("**➜ Calculated Values:**")
            st.write(f"- Start: {auto_start_idx:,} + ({start_offset:+d}) = **{calculated_start:,}**")
            st.write(f"- End: {auto_end_idx:,} + ({end_offset:+d}) = **{calculated_end:,}**")

        # Method selection (full width)
        adjustment_method = st.radio(
            "**⚙️ Adjustment Method:**",
            options=["Use Offset Values", "Use Manual Sliders"],
            index=0,
            horizontal=True,
            help="Choose how to adjust the analysis range"
        )

        # Create columns based on adjustment method
        if adjustment_method == "Use Offset Values":
            # Use calculated values from offset inputs
            manual_start_idx = calculated_start
            manual_end_idx = calculated_end
            
            # Create columns for status display
            col_status_left, col_status_right = st.columns([1, 1])
            
            with col_status_left:
                # Show validation status
                if manual_start_idx == auto_start_idx + start_offset and manual_end_idx == auto_end_idx + end_offset:
                    st.success("✅ Offset values applied successfully!")
                else:
                    st.warning("⚠️ Some values adjusted to stay within bounds")
            
            with col_status_right:
                # Show quick stats
                selected_points = manual_end_idx - manual_start_idx + 1
                time_range = column1[manual_end_idx] - column1[manual_start_idx]
                st.info(f"📊 **{selected_points:,}** points over **{time_range:.4f}s**")
                
        else:
            # Original slider interface with balanced layout
            st.write("**🎚️ Slider Adjustment:**")
            
            # Create columns for sliders
            col_slider_left, col_slider_right = st.columns([1, 1])
            
            with col_slider_left:
                # Sliders for manual adjustment with step size
                step_size = max(1, (max_start - min_start) // 1000)
                
                manual_start_idx = st.slider(
                    "Start Index",
                    min_value=min_start,
                    max_value=max_start,
                    value=auto_start_idx,
                    step=step_size,
                    help=f"Adjust starting point. Step: {step_size:,}"
                )
                
                st.write(f"**Selected Start:** {manual_start_idx:,}")
                st.write(f"**Time:** {column1[manual_start_idx]:.4f}s")
                
            with col_slider_right:
                # Update end slider bounds based on start selection
                adjusted_min_end = max(min_end, manual_start_idx + 10)
                end_step_size = max(1, (max_end - adjusted_min_end) // 1000)
                
                manual_end_idx = st.slider(
                    "End Index",
                    min_value=adjusted_min_end,
                    max_value=max_end,
                    value=min(auto_end_idx, max_end),
                    step=end_step_size,
                    help=f"Adjust ending point. Step: {end_step_size:,}"
                )
                
                st.write(f"**Selected End:** {manual_end_idx:,}")
                st.write(f"**Time:** {column1[manual_end_idx]:.4f}s")

        # Final summary section (full width)
        col_summary_left, col_summary_right = st.columns([1, 1])

        with col_summary_left:
            # Display current selection info
            selected_points = manual_end_idx - manual_start_idx + 1
            time_range = column1[manual_end_idx] - column1[manual_start_idx]
            st.success(f"**📊 Final Selection:** {selected_points:,} points over {time_range:.4f}s")

        with col_summary_right:
            # Show extension details
            start_extension = manual_start_idx - auto_start_idx
            end_extension = manual_end_idx - auto_end_idx
            
            if start_extension != 0 or end_extension != 0:
                extension_info = []
                if start_extension < 0:
                    extension_info.append(f"⬅️ Start moved back {abs(start_extension):,}")
                elif start_extension > 0:
                    extension_info.append(f"➡️ Start moved forward {start_extension:,}")
                    
                if end_extension > 0:
                    extension_info.append(f"➡️ End extended {end_extension:,}")
                elif end_extension < 0:
                    extension_info.append(f"⬅️ End shortened {abs(end_extension):,}")
                
                if extension_info:
                    st.info("**🔄 Changes:** " + " | ".join(extension_info))
            else:
                st.info("**🔄 Changes:** Using auto-detected range")

        # Show time information for the selected indices
        st.write(f"**⏰ Time Range:** {column1[manual_start_idx]:.4f}s to {column1[manual_end_idx]:.4f}s")

        # Use manual selection for analysis
        analysed_slice = slice(manual_start_idx, manual_end_idx + 1)
        
        analysis_df = pd.DataFrame({
            'Time': column1[analysed_slice],
            'Force': column2[analysed_slice],
            'Indentation': column3_processed[analysed_slice]
        }).drop_duplicates(subset='Time', keep='first')
        
        if len(analysis_df) <= 10: st.error(f"Analysis Error: Window has only {len(analysis_df)} unique points."); st.stop()
        
        # --- Apply Whittaker Smoother ---
        if use_smoothing:
            st.info(f"Applying Whittaker Smoother with λ = {lmbd:.1e}")
            analysis_df['Force'] = whittaker_smooth(analysis_df['Force'].values, lmbd=lmbd)
        
        ref_time_idx = max(0, manual_start_idx - 1)
        analysis_df['Time'] = analysis_df['Time'] - column1[ref_time_idx]
        
        timerescaled1_clean, peakrescaled_clean, gn_clean = analysis_df['Time'].values, analysis_df['Force'].values, analysis_df['Indentation'].values
        
        gnl = calculate_lambda_factor(probe_type, gn_clean, **probe_params)
        if use_smoothing: gnl = whittaker_smooth(gnl, lmbd=lmbd)
            
        t1, tN = timerescaled1_clean[0], timerescaled1_clean[-1]
        if t1 <= 0 or tN <= t1: st.error(f"Invalid time range for analysis: t1={t1:.4f}, tN={tN:.4f}."); st.stop()
            
        g1f = peakrescaled_clean[0]
        g1n = calculate_lambda_factor(probe_type, column3_processed[manual_start_idx], **probe_params)

        freqrange = np.logspace(np.log10(1/tN), np.log10(1/t1), int(num_plot))
        xf = np.logspace(np.log10(t1), np.log10(tN), int(num_interp))
        
        yf_interp = interp1d(timerescaled1_clean, peakrescaled_clean, kind=interpolation_kind, bounds_error=False, fill_value='extrapolate')
        yn_interp = interp1d(timerescaled1_clean, gnl, kind=interpolation_kind, bounds_error=False, fill_value='extrapolate')
        yf, yn = yf_interp(xf), yn_interp(xf)

        Af, An = g1f, g1n
        safe_slice_len = min(20, len(yf) - 1)
        Bn = (yn[-1] - yn[-safe_slice_len - 1]) / safe_slice_len * t1
        Bf = (yf[-1] - yf[-safe_slice_len - 1]) / safe_slice_len * t1

        ggF, ggn = np.zeros(int(num_plot), dtype=complex), np.zeros(int(num_plot), dtype=complex)
        for ww, w in enumerate(freqrange):
            expTerm = np.exp(-1j * w * xf[1:]) - np.exp(-1j * w * xf[:-1])
            diff_yf = (yf[1:] - yf[:-1]) / (xf[1:] - xf[:-1])
            ggF[ww] = (1j*w*Af + (1-np.exp(-1j*w*t1))*((g1f-Af)/t1) + Bf*np.exp(-1j*w*tN) + np.sum(diff_yf*expTerm))
            diff_yn = (yn[1:] - yn[:-1]) / (xf[1:] - xf[:-1])
            ggn[ww] = (1j*w*An + (1-np.exp(-1j*w*t1))*((g1n-An)/t1) + Bn*np.exp(-1j*w*tN) + np.sum(diff_yn*expTerm))

        with np.errstate(divide='ignore', invalid='ignore'):
            G_complex = ggF / ggn
            G_prime, G_double_prime = np.real(G_complex), np.abs(np.imag(G_complex))
            tan_delta = G_double_prime / G_prime
        
        # Filter valid results
        valid_mask = np.isfinite(G_prime) & np.isfinite(G_double_prime) & (G_prime > 0) & (G_double_prime > 0)
        
        freqs_valid = freqrange[valid_mask]
        G_prime_valid = G_prime[valid_mask]
        G_double_prime_valid = G_double_prime[valid_mask]
        tan_delta_valid = tan_delta[valid_mask]
        
        results_df = pd.DataFrame({
            'Frequency': freqs_valid, 
            "G'": G_prime_valid, 
            "G''": G_double_prime_valid, 
            "tan(δ)": tan_delta_valid
        })
        
# --- Plotting with Plotly ---
        st.success(f"Analysis complete! Found a relaxation phase with {len(analysis_df)} unique data points.")
        
        # --- 1. RAW DATA VISUALIZATION FIRST ---
        st.subheader("📊 Raw Data Visualization")
        col3, col4 = st.columns(2)

        with col3:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=data['Time'], y=data['Force'], mode='lines',
                                    name='Raw Data', line=dict(color='lightblue', width=2)))
            auto_slice = slice(auto_start_idx, auto_end_idx + 1)
            fig3.add_trace(go.Scatter(x=data['Time'][auto_slice], y=data['Force'][auto_slice], mode='lines',
                                    name='Auto-detected', line=dict(color='orange', width=3, dash='dash')))
            fig3.add_trace(go.Scatter(x=data['Time'][analysed_slice], y=data['Force'][analysed_slice], mode='lines',
                                    name='Selected Range', line=dict(color='#FF0000', width=4)))
            fig3.update_layout(
                title_text="<b>Force vs. Time</b>",
                title_x=0.45,
                title_font=dict(size=22, color='black'),
                font=dict(size=18, color='black'),
                xaxis_title="<b>Time [s]</b>",
                yaxis_title="<b>Force [N]</b>",
                xaxis_title_font=dict(color='black'),
                yaxis_title_font=dict(color='black'),
                legend=dict(yanchor="bottom", y=1.05, xanchor="right", x=1),
                plot_bgcolor='white'
            )
            fig3.update_xaxes(showgrid=True, showline=True, linewidth=2, linecolor='black', mirror=True,
                            tickfont=dict(size=20, color='black'))
            fig3.update_yaxes(showgrid=True, showline=True, linewidth=2, linecolor='black', mirror=True,
                            tickfont=dict(size=20, color='black'))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=data['Time'], y=data['Indentation'], mode='lines',
                                    name='Raw Data', line=dict(color='lightblue', width=2)))
            fig4.add_trace(go.Scatter(x=data['Time'][auto_slice], y=data['Indentation'][auto_slice], mode='lines',
                                    name='Auto-detected', line=dict(color='orange', width=3, dash='dash')))
            fig4.add_trace(go.Scatter(x=data['Time'][analysed_slice], y=data['Indentation'][analysed_slice], mode='lines',
                                    name='Selected Range', line=dict(color='#FF0000', width=4)))
            fig4.update_layout(
                title_text="<b>Indentation vs. Time</b>",
                title_x=0.3,
                title_font=dict(size=22, color='black'),
                font=dict(size=18, color='black'),
                xaxis_title="<b>Time [s]</b>",
                yaxis_title="<b>Indentation [m]</b>",
                xaxis_title_font=dict(color='black'),
                yaxis_title_font=dict(color='black'),
                legend=dict(yanchor="bottom", y=1.05, xanchor="right", x=1),
                plot_bgcolor='white'
            )
            fig4.update_xaxes(showgrid=True, showline=True, linewidth=2, linecolor='black', mirror=True,
                            tickfont=dict(size=20, color='black'))
            fig4.update_yaxes(showgrid=True, showline=True, linewidth=2, linecolor='black', mirror=True,
                            tickfont=dict(size=20, color='black'))
            st.plotly_chart(fig4, use_container_width=True)

        st.divider()

        # --- 2. RHEOLOGICAL RESULTS VISUALIZATION SECOND ---
        st.subheader("📈 Rheological Results Visualization")
        col1, col2 = st.columns(2)

        def generate_log_ticks(data_range):
            min_val, max_val = np.log10(data_range.min()), np.log10(data_range.max())
            tick_vals = np.power(10, np.arange(np.floor(min_val), np.ceil(max_val) + 1))
            tick_text = [f"10<sup>{int(np.log10(v))}</sup>" for v in tick_vals]
            return tick_vals, tick_text

        # --- Results Plot 1: Rheological Moduli ---
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=results_df['Frequency'], y=results_df["G'"], mode='markers',
                                    marker=dict(symbol='circle-open', color='#0000FF', size=7, line=dict(width=1)), name="G'"))
            fig1.add_trace(go.Scatter(x=results_df['Frequency'], y=results_df["G''"], mode='markers',
                                    marker=dict(symbol='square-open', color='#FF0000', size=7, line=dict(width=1)), name="G''"))

            freq_ticks_vals, freq_ticks_text = generate_log_ticks(results_df['Frequency'])
            moduli_ticks_vals, moduli_ticks_text = generate_log_ticks(pd.concat([results_df["G'"], results_df["G''"]]))

            fig1.update_layout(
                title_text="<b>Rheological Moduli</b>",
                title_x=0.45,
                title_font=dict(size=22, color='black'),
                font=dict(size=18, color='black'),
                xaxis_type="log",
                yaxis_type="log",
                xaxis_title="<b>Frequency [rad/s]</b>",
                yaxis_title="<b>Moduli [Pa]</b>",
                xaxis_title_font=dict(color='black'),
                yaxis_title_font=dict(color='black'),
                legend=dict(yanchor="bottom", y=1.05, xanchor="right", x=1),
                plot_bgcolor='white'
            )
            fig1.update_xaxes(showgrid=True, showline=True, linewidth=2, linecolor='black', mirror=True,
                            tickvals=freq_ticks_vals, ticktext=freq_ticks_text, tickfont=dict(size=20, color='black'))
            fig1.update_yaxes(showgrid=True, showline=True, linewidth=2, linecolor='black', mirror=True,
                            tickvals=moduli_ticks_vals, ticktext=moduli_ticks_text, tickfont=dict(size=20, color='black'))
            st.plotly_chart(fig1, use_container_width=True)

        # --- Results Plot 2: Loss Tangent ---
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=results_df['Frequency'], y=results_df['tan(δ)'], mode='markers',
                                    marker=dict(symbol='diamond-open', size=7, color='#0000FF', line=dict(width=1)), name='tan(δ)'))

            tan_ticks_vals, tan_ticks_text = generate_log_ticks(results_df['tan(δ)'])

            fig2.update_layout(
                title_text="<b>Loss Tangent</b>",
                title_x=0.45,
                title_font=dict(size=22, color='black'),
                font=dict(size=18, color='black'),
                xaxis_type="log",
                yaxis_type="log",
                xaxis_title="<b>Frequency [rad/s]</b>",
                yaxis_title="<b>tan(δ)</b>",
                xaxis_title_font=dict(color='black'),
                yaxis_title_font=dict(color='black'),
                legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
                plot_bgcolor='white'
            )
            fig2.update_xaxes(showgrid=True, showline=True, linewidth=2, linecolor='black', mirror=True,
                            tickvals=freq_ticks_vals, ticktext=freq_ticks_text, tickfont=dict(size=20, color='black'))
            fig2.update_yaxes(showgrid=True, showline=True, linewidth=2, linecolor='black', mirror=True,
                            tickvals=tan_ticks_vals, ticktext=tan_ticks_text, tickfont=dict(size=20, color='black'))
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # --- 3. DOWNLOAD SECTION LAST ---
        st.subheader("💾 Download Results")
        
        # Calculate averages for G' and G''
        g_prime_mean = results_df["G'"].mean()
        g_double_prime_mean = results_df["G''"].mean()

        col_download1, col_download2, col_download3 = st.columns([1, 1, 2])
        
        with col_download1:
            # CSV Download
            csv_data, csv_mime, csv_ext = create_download_data(results_df, 'csv')
            
            # Generate filename based on uploaded file
            base_filename = uploaded_file.name.rsplit('.', 1)[0] if uploaded_file.name else "rheological_results"
            csv_filename = f"{base_filename}_rheological_results.{csv_ext}"
            
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=csv_filename,
                mime=csv_mime,
                help="Download results as CSV file for further analysis"
            )
        
        with col_download2:
            # Excel Download
            excel_data, excel_mime, excel_ext = create_download_data(results_df, 'excel')
            excel_filename = f"{base_filename}_rheological_results.{excel_ext}"
            
            st.download_button(
                label="📊 Download Excel",
                data=excel_data,
                file_name=excel_filename,
                mime=excel_mime,
                help="Download results as Excel file with multiple sheets"
            )
        
        with col_download3:
            # Results preview with averages
            st.write("**Preview of downloadable data:**")
            st.write(f"📈 **{len(results_df)} data points** across frequency range:")
            st.write(f"• Frequency: {results_df['Frequency'].min():.2e} - {results_df['Frequency'].max():.2e} rad/s")

            # Đặt tên cột vào biến để tránh lỗi f-string
            g_prime_col = "G'"
            g_double_prime_col = "G''"

            st.write(f"• G': {results_df[g_prime_col].min():.2e} - {results_df[g_prime_col].max():.2e} Pa")
            st.write(f"• G'': {results_df[g_double_prime_col].min():.2e} - {results_df[g_double_prime_col].max():.2e} Pa")
            
            # Add average values section
            st.write("**📊 Average Values:**")
            st.write(f"• **G' (Average):** {g_prime_mean:.2e} Pa")
            st.write(f"• **G'' (Average):** {g_double_prime_mean:.2e} Pa") 

        # Show first few rows as preview
        with st.expander("🔍 Preview Results Data (first 10 rows)", expanded=False):
            st.dataframe(results_df.head(10), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        # THÊM DEBUG INFO
        if st.checkbox("🐛 Show debug information"):
            st.write("**Debug Information:**")
            st.write(f"- File name: {uploaded_file.name if uploaded_file else 'None'}")
            st.write(f"- File type: {type(uploaded_file)}")
            st.write(f"- Error type: {type(e).__name__}")
            st.write(f"- Error details: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
else:
    st.info("Please upload a data file to begin analysis.")