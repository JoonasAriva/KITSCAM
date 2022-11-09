## Data preparation pipeline plan

300 patients in KITS -- do we want to use all data?

Data distribution:

- Data for training classifier: 100 patients
- Data for classifier validation. Use score-cam here. Same data can be used for segmentation model: 150 patients
- Data for segmentation model testing: 50 patients

rules:
- kidneys must occupy at least 1000 pixels 
- no need to use neighboring slices as difference is very small. Take every 5th slice


Current script order:

``prepare_data.py`` --> ``mix_kidney_and_no_kidney.ipynb`` --> ``train_classifier.py`` --> ``scorecam_extraction.py`` 


