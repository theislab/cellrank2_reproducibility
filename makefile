.PHONY: sceu_organoid-anndata_generation

# Path to data directory
DATA_DIR := ./data/

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
