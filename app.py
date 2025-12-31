import streamlit as st
import rasterio
import numpy as np
import pandas as pd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import shape
import tempfile
import os
import re

st.set_page_config(layout="wide", page_title="GIS Processor")
st.title("🗺️ Interactive GIS Raster-Vector Processor")


def extract_band_number(filename):
    """Extract band number from filename (e.g., B02, B8A, B11)"""
    match = re.search(r'B(\d{1,2}A?)', filename, re.IGNORECASE)
    return match.group(1) if match else None


def get_band_name(band_num):
    """Map Sentinel-2 band numbers to names"""
    mapping = {
        '01': 'Coastal', '02': 'Blue', '03': 'Green', '04': 'Red',
        '05': 'RedEdge1', '06': 'RedEdge2', '07': 'RedEdge3',
        '08': 'NIR', '8A': 'NIR_Narrow', '09': 'WaterVapor',
        '10': 'SWIR_Cirrus', '11': 'SWIR1', '12': 'SWIR2'
    }
    return mapping.get(band_num, f'Band{band_num}')


def calculate_index(bands_dict, index_name):
    """Calculate spectral index from band dictionary"""
    try:
        if index_name == 'NDVI':
            if 'Red' in bands_dict and 'NIR' in bands_dict:
                r, nir = bands_dict['Red'], bands_dict['NIR']
                return np.where((nir + r) != 0, (nir - r) / (nir + r), np.nan)

        elif index_name == 'NDWI':
            if 'Green' in bands_dict and 'NIR' in bands_dict:
                g, nir = bands_dict['Green'], bands_dict['NIR']
                return np.where((g + nir) != 0, (g - nir) / (g + nir), np.nan)

        elif index_name == 'NDSI':
            if 'Green' in bands_dict and 'SWIR1' in bands_dict:
                g, swir = bands_dict['Green'], bands_dict['SWIR1']
                return np.where((g + swir) != 0, (g - swir) / (g + swir), np.nan)

        elif index_name == 'Moisture':
            if 'NIR' in bands_dict and 'SWIR1' in bands_dict:
                nir, swir = bands_dict['NIR'], bands_dict['SWIR1']
                return np.where((nir + swir) != 0, (nir - swir) / (nir + swir), np.nan)

        elif index_name == 'EVI':
            if 'Blue' in bands_dict and 'Red' in bands_dict and 'NIR' in bands_dict:
                b, r, nir = bands_dict['Blue'], bands_dict['Red'], bands_dict['NIR']
                return 2.5 * ((nir - r) / (nir + 6 * r - 7.5 * b + 1))

        elif index_name == 'SAVI':
            if 'Red' in bands_dict and 'NIR' in bands_dict:
                r, nir = bands_dict['Red'], bands_dict['NIR']
                L = 0.5
                return ((nir - r) / (nir + r + L)) * (1 + L)
    except:
        pass
    return None


def zonal_stats(index_array, geometry, transform, shape_tuple):
    """Calculate zonal statistics for polygon"""
    try:
        from rasterio.features import geometry_mask

        mask_arr = geometry_mask([geometry], transform=transform,
                                 invert=True, out_shape=shape_tuple)

        masked_values = index_array[mask_arr]
        valid = masked_values[~np.isnan(masked_values)]

        if len(valid) > 0:
            return {
                'mean': float(valid.mean()),
                'count': int(len(valid)),
                'sum': float(valid.sum()),
                'std': float(valid.std()),
                'min': float(valid.min()),
                'max': float(valid.max()),
                'median': float(np.median(valid))
            }
    except Exception as e:
        st.error(f"Error calculating stats: {e}")
    return None


# Sidebar
with st.sidebar:
    st.header("📊 1. Select Index")

    all_indices = ['NDVI', 'NDWI', 'NDSI', 'Moisture', 'EVI', 'SAVI']
    selected_index = st.selectbox("Choose spectral index:", all_indices)

    index_desc = {
        'NDVI': '🌿 Vegetation health',
        'NDWI': '💧 Water content',
        'NDSI': '❄️ Snow/ice cover',
        'Moisture': '💦 Soil moisture',
        'EVI': '🌱 Enhanced vegetation',
        'SAVI': '🏜️ Soil adjusted vegetation'
    }
    st.info(index_desc.get(selected_index, ''))

    # Show required bands for selected index
    required_bands = {
        'NDVI': ['Red (B04)', 'NIR (B08)'],
        'NDWI': ['Green (B03)', 'NIR (B08)'],
        'NDSI': ['Green (B03)', 'SWIR1 (B11)'],
        'Moisture': ['NIR (B08)', 'SWIR1 (B11)'],
        'EVI': ['Blue (B02)', 'Red (B04)', 'NIR (B08)'],
        'SAVI': ['Red (B04)', 'NIR (B08)']
    }
    with st.expander("Required bands", expanded=True):
        for band in required_bands.get(selected_index, []):
            st.text(f"• {band}")

    st.divider()

    st.header("📁 2. Upload Raster Bands")

    uploaded_files = st.file_uploader(
        "Upload multiple GeoTIFF files",
        type=['tif', 'tiff'],
        accept_multiple_files=True,
        help="Upload bands like B02.tif, B03.tif, B04.tif, B08.tif, B11.tif, B12.tif"
    )

    st.divider()

    st.header("✅ 3. Uploaded Bands")

    bands_info = {}
    if uploaded_files:
        st.success(f"**{len(uploaded_files)} files uploaded**")

        # Process uploaded files
        for file in uploaded_files:
            band_num = extract_band_number(file.name)
            if band_num:
                band_name = get_band_name(band_num)
                bands_info[band_name] = file
                st.text(f"✓ {band_name} ({file.name})")

        # Check if required bands are present
        st.divider()
        can_calculate = False

        if selected_index == 'NDVI' and 'Red' in bands_info and 'NIR' in bands_info:
            can_calculate = True
        elif selected_index == 'NDWI' and 'Green' in bands_info and 'NIR' in bands_info:
            can_calculate = True
        elif selected_index == 'NDSI' and 'Green' in bands_info and 'SWIR1' in bands_info:
            can_calculate = True
        elif selected_index == 'Moisture' and 'NIR' in bands_info and 'SWIR1' in bands_info:
            can_calculate = True
        elif selected_index == 'EVI' and 'Blue' in bands_info and 'Red' in bands_info and 'NIR' in bands_info:
            can_calculate = True
        elif selected_index == 'SAVI' and 'Red' in bands_info and 'NIR' in bands_info:
            can_calculate = True

    else:
        st.info("No files uploaded yet")
        can_calculate = False

# Main area
if uploaded_files and can_calculate and selected_index:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🗺️ Interactive Map - Draw Polygon")

        # Save files temporarily and read bands
        temp_files = {}
        bands_dict = {}
        reference_src = None

        try:
            for band_name, file in bands_info.items():
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
                tmp.write(file.read())
                tmp.close()
                temp_files[band_name] = tmp.name

                with rasterio.open(tmp.name) as src:
                    bands_dict[band_name] = src.read(1).astype(float)
                    if reference_src is None:
                        reference_src = {
                            'bounds': src.bounds,
                            'transform': src.transform,
                            'crs': src.crs,
                            'shape': (src.height, src.width)
                        }

            # Create map
            bounds = reference_src['bounds']
            center_lat = (bounds.bottom + bounds.top) / 2
            center_lon = (bounds.left + bounds.right) / 2

            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

            # Add raster extent
            folium.Rectangle(
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                color='blue',
                fill=True,
                fillOpacity=0.1,
                popup=f'Raster extent ({reference_src["shape"][1]}x{reference_src["shape"][0]})'
            ).add_to(m)

            # Add drawing tools
            draw = Draw(
                export=True,
                draw_options={
                    'polyline': False,
                    'rectangle': True,
                    'circle': False,
                    'marker': False,
                    'circlemarker': False,
                    'polygon': True
                }
            )
            draw.add_to(m)

            # Display map
            map_output = st_folium(m, width=800, height=600, key="map")

            # Process drawn polygon
            if map_output and map_output.get('last_active_drawing'):
                drawing = map_output['last_active_drawing']

                if drawing and 'geometry' in drawing:
                    with col2:
                        st.subheader(f"📈 {selected_index} Statistics")

                        with st.spinner("Calculating..."):
                            # Convert drawing to geometry
                            geom = shape(drawing['geometry'])

                            # Calculate index
                            index_array = calculate_index(bands_dict, selected_index)

                            if index_array is not None:
                                # Reproject geometry if needed
                                from shapely.ops import transform as shapely_transform
                                from pyproj import Transformer

                                transformer = Transformer.from_crs("EPSG:4326", reference_src['crs'], always_xy=True)
                                geom_proj = shapely_transform(transformer.transform, geom)

                                # Calculate stats
                                stats = zonal_stats(
                                    index_array,
                                    geom_proj.__geo_interface__,
                                    reference_src['transform'],
                                    reference_src['shape']
                                )

                                if stats:
                                    st.success("✅ Calculation complete!")

                                    # Display stats table
                                    df = pd.DataFrame([{
                                        'Statistic': k.upper(),
                                        'Value': f"{v:.6f}" if isinstance(v, float) else v
                                    } for k, v in stats.items()])

                                    st.dataframe(df, use_container_width=True, hide_index=True)

                                    # Metrics
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.metric("Mean", f"{stats['mean']:.4f}")
                                        st.metric("Std Dev", f"{stats['std']:.4f}")
                                    with col_b:
                                        st.metric("Pixels", stats['count'])
                                        st.metric("Median", f"{stats['median']:.4f}")

                                    # Download
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        "⬇️ Download CSV",
                                        csv,
                                        f"{selected_index}_stats.csv",
                                        "text/csv",
                                        use_container_width=True
                                    )
                                else:
                                    st.error("No valid pixels in polygon")
                            else:
                                st.error(f"Cannot calculate {selected_index}")
            else:
                with col2:
                    st.info("👈 Draw a polygon on the map")
                    st.markdown("""
                    **How to draw:**
                    1. Click polygon/rectangle icon on map
                    2. Draw your area of interest
                    3. Statistics will appear here
                    """)

        finally:
            # Cleanup temp files
            for tmp_path in temp_files.values():
                try:
                    os.unlink(tmp_path)
                except:
                    pass

else:
    st.info("👈 Upload raster bands in the sidebar to begin")
    st.markdown("""
    ### 📖 Instructions:

    1. **Upload multiple raster files** (e.g., Sentinel-2 bands: B02, B03, B04, B08, B11, B12)
    2. **Select an index** from available options based on uploaded bands
    3. **Draw a polygon** on the interactive map
    4. **View zonal statistics** in the right panel
    5. **Download results** as CSV

    ### 🛰️ Sentinel-2 Band Naming:
    - B02 = Blue
    - B03 = Green
    - B04 = Red
    - B08 = NIR
    - B11 = SWIR1
    - B12 = SWIR2

    ### 📊 Supported Indices:
    - **NDVI**: Vegetation Index (Red + NIR)
    - **NDWI**: Water Index (Green + NIR)
    - **NDSI**: Snow Index (Green + SWIR1)
    - **Moisture**: Soil Moisture (NIR + SWIR1)
    - **EVI**: Enhanced Vegetation (Blue + Red + NIR)
    - **SAVI**: Soil Adjusted Vegetation (Red + NIR)
    """)

st.divider()
st.caption("🛰️ Multi-band GIS Processor | Streamlit + Rasterio + Folium")

### Next to update
# creating readme with instructions
# Displaying the .tif over the map , by default rgb if available, or chosen index if possible
# Changing 'trash' option on the map
# Adding support to select fields from uldk instead of drawing geometry
