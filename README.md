# 🗺️ Raster Index Zonal Statistics Tool

An interactive tool for computing spectral indices from **Sentinel-2 raster bands**, visualizing them on a map, and extracting **zonal statistics** from user-drawn polygons.

---

## 📖 Instructions

1. **Upload multiple raster files**  
   - Filenames **must contain band identifiers** (e.g. `B01`, `B02`, `B11`)
2. **Select an index**  
   - Available options depend on the uploaded bands
3. **Toggle overlay** *(optional)*  
   - Visualize the selected index as a colored map overlay
4. **Draw a polygon**  
   - Use the interactive map tools
5. **View zonal statistics**  
   - Results appear in the right-hand panel
6. **Download results**  
   - Export statistics as a **CSV**
7. **Clear shapes**  
   - Use the 🗑️ trash icon to reset drawings

---

## 📊 Supported Indices

| Index | Description | Required Bands |
|------|------------|----------------|
| **NDVI** | Vegetation Index | Red + NIR |
| **NDWI** | Water Index | Green + NIR |
| **NDSI** | Snow Index | Green + SWIR1 |
| **Moisture** | Soil Moisture Index | NIR + SWIR1 |
| **EVI** | Enhanced Vegetation Index | Blue + Red + NIR |
| **SAVI** | Soil Adjusted Vegetation Index | Red + NIR |

---

## 🛰️ Sentinel-2 Band Naming

| Band | Description |
|------|------------|
| **B02** | Blue |
| **B03** | Green |
| **B04** | Red |
| **B08** | Near Infrared (NIR) |
| **B11** | Shortwave Infrared 1 (SWIR1) |
| **B12** | Shortwave Infrared 2 (SWIR2) |

---

## 🎨 Overlay Color Schemes

| Index | Color Ramp |
|------|-----------|
| **NDVI** | Brown → Yellow → Green → Dark Green |
| **NDWI** | Brown → Light Blue → Blue → Dark Blue |
| **NDSI** | Brown → Gray → White → Cyan |
| **Moisture** | Red → Yellow → Light Blue → Blue |

---

## 📦 Output

- Interactive spectral index visualization
- Polygon-based zonal statistics
- Downloadable CSV results

---
