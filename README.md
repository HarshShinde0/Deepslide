# Landslide Detection and Mapping Using Deep Learning Across Multi-Source Satellite Data and Geographic Regions

[![Paper](https://img.shields.io/badge/Paper-SSRN-blue)](https://dx.doi.org/10.2139/ssrn.5225437)
[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/HarshShinde0/Deepslide)
[![Models](https://img.shields.io/badge/Models-HuggingFace-yellow)](https://huggingface.co/harshinde/DeepSlide_Models)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-orange)](https://huggingface.co/datasets/harshinde/LandSlide4Sense)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace-red)](https://huggingface.co/spaces/harshinde/DeepSlide)

**Authors**: [Harsh Shinde](https://harshshinde0.github.io/), Rahul Burange, Omkar Mutyalwar  
**Institution**: KDKCE

## Abstract

Landslides pose severe threats to infrastructure, economies, and human lives, necessitating accurate detection and predictive mapping across diverse geographic regions. With advancements in deep learning and remote sensing, automated landslide detection has become increasingly effective. This study presents a comprehensive approach integrating multi-source satellite imagery and deep learning models to enhance landslide identification and prediction. We leverage Sentinel-2 multispectral data and ALOS PALSAR-derived slope and Digital Elevation Model (DEM) layers to capture critical environmental features influencing landslide occurrences. Various geospatial analysis techniques are employed to assess the impact of terrain characteristics, vegetation cover, and rainfall on detection accuracy. Additionally, we evaluate the performance of multiple state-of-the-art deep learning segmentation models, including U-Net, DeepLabV3+, and ResNet, to determine their effectiveness in landslide detection. The proposed framework contributes to the development of reliable early warning systems, improved disaster risk management, and sustainable land-use planning. Our findings provide valuable insights into the potential of deep learning and multi-source remote sensing in creating robust, scalable, and transferable landslide prediction models.

**Index Terms**: *Image Processing, Machine Learning, Deep Learning, Computer Vision, Remote Sensing.*

## Our Method

### Study Areas

The selected study areas represent diverse geographic and climatic conditions:

1. **Iburi-Tobu Area of Hokkaido, Japan**
   - Hit by a magnitude 6.6 earthquake on September 6, 2018, triggering over 5600 landslides
   - Landslides were exacerbated by preceding heavy rainfall from Typhoon Jebi
   - Landslide inventories were created using very high-resolution aerial images

2. **Kodagu District of Karnataka, India**
   - Experienced extreme rainfall in August 2018, triggering severe landslides and flash floods
   - Landslides were linked to deforestation, unplanned urbanization, and mining activities
   - Previous studies have applied unsupervised learning techniques for landslide detection

3. **Rasuwa District of Bagmati, Nepal**
   - One of the most landslide-prone regions in the Himalayas
   - Major landslides occurred due to the 2015 Gorkha and Dolakha earthquakes
   - Landslide inventory compiled from GPS field surveys and visual interpretation of high-resolution images

4. **Western Taitung County, Taiwan**
   - Landslides frequently triggered by typhoons and earthquakes
   - Typhoon Morakot (2009) caused extensive landslides, destroying villages and infrastructure
   - Landslide inventory derived from previous studies and Google Earth images

### Sensor Characteristics

#### Sentinel-2
- Provides multi-spectral imagery with 13 bands at spatial resolutions of 10, 20, and 60 meters
- High revisit frequency (2–3 days at mid-latitudes) enables continuous monitoring
- Sentinel-2 data were obtained from Google Earth Engine (GEE), ensuring cloud-free imagery for analysis

#### ALOS PALSAR
- Provides synthetic aperture radar (SAR) data with a 12.5m spatial resolution
- The DEM and slope layers derived from ALOS PALSAR were used to supplement optical imagery

### Benchmark Dataset Statistics and Structure

The Landslide4Sense benchmark dataset includes 128×128 window-size patches, each containing 14 distinct data layers. The first 12 bands consist of multi-spectral data from Sentinel-2, while bands 13 and 14 represent the Digital Elevation Model (DEM) and slope data derived from ALOS PALSAR.

**The 14 layers in the Landslide4Sense dataset:**
- Sentinel-2 band 1: Blue spectral band data
- Sentinel-2 band 2: Green spectral band data
- Sentinel-2 band 3: Red spectral band data
- Sentinel-2 band 4: Near Infrared (NIR) spectral band data
- Sentinel-2 band 5-7: Shortwave Infrared (SWIR) spectral band data
- Sentinel-2 band 8: NIR spectral band data
- Sentinel-2 band 9: Water Vapour (WV) spectral band data
- Sentinel-2 band 10: Cirrus (CI) spectral band data
- Sentinel-2 band 11-12: SWIR spectral band data
- Digital Elevation Model (DEM): Elevation information data
- Slope: Slope information data

**Dataset Statistics:**
- 3799 annotated image patches (128 × 128 pixels each, with a resolution of 10 meters per pixel)
- 14 bands per image patch
- Dataset split: 959 patches for training, 2840 patches for testing

## Performance

Our results demonstrate that ResNet34, VGG-16, and EfficientNet-B0 achieved the highest F1 Scores, indicating superior performance in distinguishing landslide-prone areas from non-landslide regions. The ResNet34-based U-Net model attained the best balance between precision and recall, achieving an F1 Score of 0.7470, making it the most reliable among the tested architectures.

### Model Performance Comparison

| Models | F1 Score | Precision | Recall |
|--------|----------|-----------|--------|
| **ResNet34** | **0.7470** | 0.7737 | 0.7267 |
| VGG16 | 0.7357 | 0.7650 | 0.7121 |
| EfficientNet-B0 | 0.7341 | 0.7536 | 0.7221 |
| ResNeXt50_32X4D | 0.7330 | 0.7453 | 0.7247 |
| SeResNet-50 | 0.7328 | 0.7826 | 0.6950 |
| DenseNet121 | 0.7290 | 0.7241 | 0.7400 |
| SeResNeXt50_32x4D | 0.7279 | 0.7249 | 0.7350 |
| InceptionV4 | 0.7246 | 0.7631 | 0.6945 |
| InceptionResNetV2 | 0.7151 | 0.7774 | 0.6692 |
| DeepLabV3+ | 0.7141 | 0.7471 | 0.6897 |
| MobileNetV2 | 0.7119 | 0.7000 | 0.7337 |
| U-Net | 0.7012 | 0.7906 | 0.6338 |
| MiT-B1 | 0.6989 | 0.7574 | 0.6596 |

## Results

We evaluated various deep learning models for landslide detection using the Landslide4Sense dataset with Sentinel-2 imagery and ALOS PALSAR elevation data. The ResNet34-based U-Net achieved the highest F1 Score of 0.7470 with balanced precision (0.7737) and recall (0.7267). VGG16 and EfficientNet-B0 also performed well with F1 Scores of 0.7357 and 0.7341 respectively. Advanced architectures significantly outperformed the classic U-Net (F1: 0.7012), demonstrating the importance of deeper feature extraction mechanisms for complex geospatial data.

## Conclusion

This study demonstrates that hybrid deep learning models with advanced feature extraction significantly outperform traditional U-Net for landslide detection. ResNet34-based U-Net emerged as the most reliable architecture with an F1 Score of 0.7470. Multi-source data integration combining optical imagery with elevation information proves crucial for accurate landslide identification. These findings contribute to developing more reliable disaster risk management and early warning systems for landslide-prone regions.

## Citations

```bibtex
@article{burange2025landslide,
  title={Landslide Detection and Mapping Using Deep Learning Across Multi-Source Satellite Data and Geographic Regions},
  author={Burange, Rahul and Shinde, Harsh and Mutyalwar, Omkar},
  journal={Available at SSRN 5225437},
  year={2025},
  doi={10.2139/ssrn.5225437}
}

@article{burange2025landslide,
  title={Landslide Detection and Mapping Using Deep Learning Across Multi-Source Satellite Data and Geographic Regions},
  author={Burange, Rahul A and Shinde, Harsh K and Mutyalwar, Omkar},
  journal={arXiv preprint arXiv:2507.01123},
  year={2025}
}

@article{burange2025comprehensive,
  title={A Comprehensive Approach to Landslide Detection: Deep Learning and Remote Sensing Integration},
  author={Burange, Rahul and Shinde, Harsh and Mutyalwar, Omkar},
  year={2025},
  publisher={IJARCCE}
}

@article{burange2025exhaustive,
  title={An Exhaustive Review on Deep Learning for Advanced Landslide Detection and Prediction from Multi-Source Satellite Imagery},
  author={Burange, Rahul and Shinde, Harsh and Mutyalwar, Omkar},
  journal={Available at SSRN 5155990},
  year={2025},
  doi={10.2139/ssrn.5155990},
  url={https://ssrn.com/abstract=5155990}
}
```