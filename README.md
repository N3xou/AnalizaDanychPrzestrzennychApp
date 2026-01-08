### 📖 Instructions:

    1. **Upload multiple raster files** (Filenames must contain bands in their names e.g. B01, B02, B11)
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
    """
