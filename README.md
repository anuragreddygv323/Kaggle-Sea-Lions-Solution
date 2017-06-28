# Kaggle-Sea-Lions-Solution
NOAA Fisheries Steller Sea Lion Population Count

# 01-Image level regression
- Solves problem as regression task on whole image. Implemented in Keras. 
Modified from https://github.com/mrgloom/Kaggle-Sea-Lions-Solution
  - Additions:
  - Validation dataflow
  - RMSE metric
  - Stacking implementation (to add)
  
# 02-Window level regression
- Image level regression extended to windows extracted from full images.
- Includes code to count sealions in each window, and train network on the enlarged dataset.
