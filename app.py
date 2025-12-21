import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
from io import BytesIO
import tempfile
import os

st.title("GIS Raster-Vector Processor")

# Upload files
vector_file = st.file_uploader("Upload GeoJSON", type=['geojson'])
raster_file = st.file_uploader("Upload Raster (GeoTIFF)", type=['tif', 'tiff'])

if vector_file and raster_file:
    # Load vector
    gdf = gpd.read_file(vector_file)
    st.write(f"Vector features: {len(gdf)}")

    # Save raster to temp file (rasterio needs file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        tmp.write(raster_file.read())
        tmp_path = tmp.name

    # Load raster
    with rasterio.open(tmp_path) as src:
        raster_data = src.read()
        raster_meta = src.meta

        # Calculate spectral index (NDVI example: (NIR - Red) / (NIR + Red))
        if raster_data.shape[0] >= 2:
            red = raster_data[0].astype(float)
            nir = raster_data[1].astype(float)
            ndvi = np.where((nir + red) != 0, (nir - red) / (nir + red), 0)
            st.write("NDVI calculated")
            st.write(f"NDVI range: {ndvi.min():.3f} to {ndvi.max():.3f}")
        else:
            ndvi = raster_data[0].astype(float)
            st.write("Single band processed")

        # Extract statistics per geometry
        results = []
        for idx, row in gdf.iterrows():
            try:
                geom = [row.geometry.__geo_interface__]
                masked, transform = mask(src, geom, crop=True, all_touched=True)
                masked_data = masked[0] if masked.shape[0] == 1 else masked
                valid = masked_data[masked_data != src.nodata]

                if len(valid) > 0:
                    results.append({
                        'feature_id': idx,
                        'mean': valid.mean(),
                        'std': valid.std(),
                        'min': valid.min(),
                        'max': valid.max()
                    })
            except:
                continue

        if results:
            df = pd.DataFrame(results)
            st.write("Statistics per feature:")
            st.dataframe(df)

            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                "results.csv",
                "text/csv"
            )

    os.unlink(tmp_path)
else:
    st.info("Upload both GeoJSON and GeoTIFF files to process")