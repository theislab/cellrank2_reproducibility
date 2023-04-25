.PHONY: intestinal_epithelium_marker-data_preprocessing sceu_organoid-anndata_generation

# Path to data directory
DATA_DIR := ./data/

INTESTINAL_EPITHELIUM_DATA_URL := https://static-content.springer.com/esm/art%3A10.1038%2Fnature24489/MediaObjects/
SCEU_ORGANOID_DATA_URL := https://ftp.ncbi.nlm.nih.gov/geo/series/GSE128nnn/GSE128365/suppl/

SCEU_ORGANOID_DATA_DIR = $(addsuffix sceu_organoid/, $(DATA_DIR))
SCEU_ORGANOID_RAW_DATA_DIR := $(addsuffix raw/, $(SCEU_ORGANOID_DATA_DIR))

sceu_organoid-download_data:
	mkdir -p $(SCEU_ORGANOID_RAW_DATA_DIR)
	wget -N $(SCEU_ORGANOID_DATA_URL)GSE128365_SupplementaryData_organoids_cell_metadata.csv.gz -P $(SCEU_ORGANOID_RAW_DATA_DIR)
	wget -N $(SCEU_ORGANOID_DATA_URL)GSE128365_SupplementaryData_organoids_labeled_splicedUMI.csv.gz -P $(SCEU_ORGANOID_RAW_DATA_DIR)
	wget -N $(SCEU_ORGANOID_DATA_URL)GSE128365_SupplementaryData_organoids_labeled_unsplicedUMI.csv.gz -P $(SCEU_ORGANOID_RAW_DATA_DIR)
	wget -N $(SCEU_ORGANOID_DATA_URL)GSE128365_SupplementaryData_organoids_unlabeled_splicedUMI.csv.gz -P $(SCEU_ORGANOID_RAW_DATA_DIR)
	wget -N $(SCEU_ORGANOID_DATA_URL)GSE128365_SupplementaryData_organoids_unlabeled_unsplicedUMI.csv.gz -P $(SCEU_ORGANOID_RAW_DATA_DIR)

sceu_organoid-anndata_generation: sceu_organoid-download_data
    ifeq ($(ENFORCE), True)
		python scripts/labeling_kernel/anndata_generation.py --enforce
    else
		python scripts/labeling_kernel/anndata_generation.py
    endif

intestinal_epithelium_marker-download_data:
	mkdir -p $(SCEU_ORGANOID_RAW_DATA_DIR)
	wget -N $(INTESTINAL_EPITHELIUM_DATA_URL)41586_2017_BFnature24489_MOESM3_ESM.xlsx -P $(SCEU_ORGANOID_RAW_DATA_DIR)
	wget -N $(INTESTINAL_EPITHELIUM_DATA_URL)41586_2017_BFnature24489_MOESM4_ESM.xlsx -P $(SCEU_ORGANOID_RAW_DATA_DIR)
	wget -N $(INTESTINAL_EPITHELIUM_DATA_URL)41586_2017_BFnature24489_MOESM7_ESM.xlsx -P $(SCEU_ORGANOID_RAW_DATA_DIR)

# To run locally: make intestinal_epithelium_marker-data_preprocessing LOCATION=LOCAL
intestinal_epithelium_marker-data_preprocessing: intestinal_epithelium_marker-download_data
	python scripts/labeling_kernel/markers.py
