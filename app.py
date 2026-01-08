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
from PIL import Image
import base64
from io import BytesIO
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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


def create_index_overlay(index_array, bounds, crs, index_name):
    """Create a colored overlay image for the index"""
    try:
        # Normalize index values to 0-1
        valid_mask = ~np.isnan(index_array)
        if not valid_mask.any():
            return None

        normalized = np.full_like(index_array, np.nan)
        valid_data = index_array[valid_mask]

        # Use percentile clipping for better visualization
        vmin, vmax = np.percentile(valid_data, [2, 98])
        normalized[valid_mask] = np.clip((index_array[valid_mask] - vmin) / (vmax - vmin + 1e-10), 0, 1)

        # Define colormaps for different indices
        colormaps = {
            'NDVI': LinearSegmentedColormap.from_list('ndvi', ['brown', 'yellow', 'green', 'darkgreen']),
            'NDWI': LinearSegmentedColormap.from_list('ndwi', ['brown', 'lightblue', 'blue', 'darkblue']),
            'NDSI': LinearSegmentedColormap.from_list('ndsi', ['brown', 'lightgray', 'white', 'cyan']),
            'Moisture': LinearSegmentedColormap.from_list('moisture', ['red', 'yellow', 'lightblue', 'blue']),
            'EVI': LinearSegmentedColormap.from_list('evi', ['brown', 'yellow', 'green', 'darkgreen']),
            'SAVI': LinearSegmentedColormap.from_list('savi', ['tan', 'yellow', 'lightgreen', 'green'])
        }

        cmap = colormaps.get(index_name, plt.cm.RdYlGn)

        # Apply colormap
        rgba = cmap(normalized)
        rgba[~valid_mask] = [0, 0, 0, 0]  # Transparent for NaN
        rgba[:, :, 3] = np.where(valid_mask, 0.6, 0)  # Set alpha

        # Convert to image
        img = Image.fromarray((rgba * 255).astype(np.uint8), mode='RGBA')

        # Save to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str, [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

    except Exception as e:
        st.error(f"Error creating overlay: {e}")
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

if 'clear_counter' not in st.session_state:
    st.session_state.clear_counter = 0

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
    with st.expander("Required bands"):
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

        if can_calculate:
            st.success(f"✅ Ready to calculate {selected_index}")
        else:
            st.warning(f"⚠️ Missing bands for {selected_index}")

        # Add overlay toggle
        if can_calculate:
            show_overlay = True
    else:
        st.info("No files uploaded yet")
        can_calculate = False
        show_overlay = False

# Main area
if uploaded_files and can_calculate and selected_index:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🗺️ Interactive Map")

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

            # Calculate and add index overlay if enabled
            if show_overlay:
                with st.spinner("Generating index overlay..."):
                    index_array = calculate_index(bands_dict, selected_index)

                    if index_array is not None:
                        overlay_result = create_index_overlay(
                            index_array,
                            bounds,
                            reference_src['crs'],
                            selected_index
                        )

                        if overlay_result:
                            img_str, img_bounds = overlay_result

                            folium.raster_layers.ImageOverlay(
                                image=f"data:image/png;base64,{img_str}",
                                bounds=img_bounds,
                                opacity=0.6,
                                interactive=True,
                                cross_origin=False,
                                zindex=1,
                                name=f"{selected_index} Overlay"
                            ).add_to(m)

                            # Add layer control
                            folium.LayerControl().add_to(m)

            # Add drawing tools with trash functionality
            draw = Draw(
                export=True,
                draw_options={
                    'polyline': False,
                    'rectangle': True,
                    'circle': False,
                    'marker': False,
                    'circlemarker': False,
                    'polygon': True
                },
                edit_options={'edit': False}
            )
            draw.add_to(m)


            # Add custom JavaScript to enable "delete all" functionality
            delete_all_js = """
            <script>
            function deleteAllShapes() {
                var map = window.parent.foliumMap;
                if (map && map.eachLayer) {
                    map.eachLayer(function(layer) {
                        if (layer instanceof L.Path && !(layer instanceof L.Rectangle)) {
                            map.removeLayer(layer);
                        }
                    });
                }
            }
            </script>
            """
            m.get_root().html.add_child(folium.Element(delete_all_js))

            # Display map
            map_output = st_folium(
                m,
                width=800,
                height=600,
                key=f"map_{st.session_state.clear_counter}"
            )
            if st.button("🗑️ Reload (clear shapes)", key="clear_btn"):
                st.session_state.clear_counter += 1  # Zmienia klucz mapy
                st.rerun()  # Przeładowanie


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

                    **Clear shapes:**
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

    1. **Upload multiple raster files** (Filenames must contain band e.g. B01, B02, B11)
    2. **Select an index** from available options based on uploaded bands
    3. **Toggle overlay** to see colored index visualization (optional)
    4. **Draw a polygon** on the interactive map
    5. **View zonal statistics** in the right panel
    6. **Download results** as CSV
    7. **Clear shapes** using trash icon

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

    ### 🎨 Overlay Colors:
    - **NDVI**: Brown → Yellow → Green → Dark Green
    - **NDWI**: Brown → Light Blue → Blue → Dark Blue
    - **NDSI**: Brown → Gray → White → Cyan
    - **Moisture**: Red → Yellow → Light Blue → Blue
    """)

st.divider()
st.caption("🛰️ Multi-band GIS Processor | Streamlit + Rasterio + Folium")