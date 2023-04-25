import os
import sys

import pandas as pd

sys.path.append("./")
from paths import DATA_DIR  # isort: skip # noqa: E402

# fmt: off
# Taken from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5641633/,
# https://www.molbiolcell.org/doi/10.1091/mbc.E17-03-0204,
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3343025/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6209304/
# https://www.cell.com/cell/pdf/S0092-8674(20)30501-8.pdf
EEC_MARKERS = ["Il6ra", "Tph1", "Cck", "Pax4", "Gck", "Mdk", "Vwa5b2", "Mdk", "Slc18a1", "Chgb", "Scg3"]
# Taken from https://www.sciencedirect.com/science/article/pii/S009286741831643X?via%3Dihub)
EEC_PROGENITOR_MARKERS = ["Myt1", "Sox4", "Rfx6", "Runx1t1"]

# Taken from https://www.mdpi.com/2073-4425/2/1/219,
# https://www.sciencedirect.com/science/article/pii/S0016508509010014
# https://www.sciencedirect.com/science/article/pii/S009286741831643X?via%3Dihub
# https://www.biorxiv.org/content/10.1101/575845v3.full.pdf
GOBLET_MARKERS = [
    "Osbpl2", "Ttc39a", "Mlph", "Dap", "C1galt1c1", "Klf4", "Capn8", "P2rx4", "Ang", "Sec24d", "Zg16", "Rcan3",
    "Gcc2", "Etnk1", "Qsox1", "Plaur", "St3gal1", "Hspa5", "Rasa4", "Ramp1", "Fry", "Syvn1", "Syt7", "Eif2ak3",
    "Dnajc10", "Rassf6", "Zbp1", "Fam114a1", "Sh3pxd2a", "Dnajc3", "Selenom", "C2cd2l", "Adgre5", "Gsn", "AW112010",
    "Tff3", "Bcas1", "Kdelr3", "Guca2a", "Hid1", "Gcnt3", "Spink4", "Liph", "D630039A03Rik", "Sgsm3", "Kcnh3",
    "Cracr2a", "Fxyd3", "Fkbp11", "Bmp8a", "Rap1gap", "Slc12a8", "Tfcp2l1", "Rnase1", "Pck1", "Ggcx", "Scnn1a",
    "Slc22a15", "Tspan1", "Atp2c2", "Rasd2", "Tnfaip8", "Sdf2l1", "Clca1", "Gpr20", "Galnt12", "Creld2", "Creb3l1",
    "Pdia5", "Hspa13", "Gmppb", "Chrm1", "Ica1", "S100a6", "Impad1", "Agr2", "Mia3", "Tor3a", "Kcnk6", "Tspan13",
    "Fhl1", "Grk2", "Scin", "Pacsin1", "Tmed3", "Ern2", "Edem2", "B3gnt5", "Pdia6", "Capn9", "Car8", "Bace2", "Rnf39",
    "Slc17a9", "Mfsd7a", "Slc41a2", "Calr", "Gal3st2", "Rab3d", "Creb3l4", "Nipal2", "Stx17", "Fut4", "S100a14",
    "Selenos", "Ptprr", "Sytl2", "Spats2l", "Galnt5", "Plcb1", "Lrrc26", "Plpp5", "Fcgbp", "Chst4", "Myo5c", "Sidt1",
    "Muc2", "Rab27b", "Hgfac", "Atoh1", "Cracr2b", "Sh3bgrl3", "Aldh3b2", "Spdef", "Foxa3", "Ccl9", "Mllt3", "Sytl4",
    "Gm9994", "Rep15", "Gm1123", "Tpsg1", "Smim14", "Mmp7", "Serp1", "Gfi1", "Tmco3", "Pmm2", "Ift20", "Herpud1",
    "Mcf2l", "Tmem263", "Wars", "Id4", "Pdia3", "Manf", "Tmbim4", "Fgfr3", "Asph", "Atp2a3", "Hepacam2", "Cbfa2t3",
    "Defa17", "Cryptdins", "Mmp7", "Ang4", "Kallikreins", "Muc2"
]
GOBLET_REGULATORS = [
    "miR-210-3p", "miR-222-3p", "Btg2", "Creb3", "Elk3", "Fev", "miR-101c", "miR-125a-5p", "Mllt3", "Prox1", "Spdef",
    "Spib", "Dnmt1", "miR-130a-3p", "Nr0b2", "Ovol2", "Etv6", "Isl1", "Rfx6", "Ehf", "Hes1", "Irf7", "miR-218-5p",
    "miR-677-5p", "Msx1", "Mybl1", "Ascl2", "Pou2f1", "Egr2", "E2f1", "Tcf7", "Gfi1b",
]
GOBLET_AND_PANETH_REGULATORS = [
    "Etv4", "Klf15", "miR-153-3p", "Pax6", "Atoh1", "Fosb", "let-7e-5p", "miR-7a-5p", "Foxa3", "Fosl1", "miR-152-3p",
    "Nkx2-2", "miR-101a-3p", "Myb", "Mitf", "Nr5a2", "Irf1", "Rora", "Myc", "Bhlha15", "Neurod2", "Insm1", "Neurod1",
    "Hoxb4", "Foxa1", "Tead4", "Nr3c1", "Vdr", "Ets1", "Zfp57"
]

# Taken from https://doi.org/10.1038/nature24489
PANETH_MARKERS = [
    "Dkk3", "Gm7325", "Cd244", "Sync", "Samd5", "Dll3", "Defb1", "Pnliprp1", "Hspb8", "Slc30a2", "Fzd9", "Ints6l",
    "Copz2", "Pnliprp2", "Itln1", "Fgfrl1", "Habp2", "Lyz1", "Pla2g2f", "Sntb1", "Reg4", "Ang4", "Klf15", "Clps",
    "Fam46c", "Defa21", "Thbs1", "Mptx2", "Defa24", "Mmp7", "Defa30", "Defa29", "Defa17", "Defa20", "Gm15308",
    "Defa22", "Defa21", "Guca2a", "Gm15315", "Gm21002", "Nupr1", "Gm10104", "Gm1123", "Agr2", "Muc2", "Gm15293",
    "Pnliprp2", "Tspan1", "Itln1", "Pglyrp1", "mt-Atp6", "Guca2b", "Gm15292", "Gm15299", "Defa17" "Clps", "Defa23",
    "Gm14851" "Gm15284", "AY761184", "Rnase1"
]
# fmt: on


if __name__ == "__main__":
    os.makedirs(DATA_DIR / "sceu_organoid" / "processed", exist_ok=True)

    markers = pd.concat(
        [
            pd.read_excel(
                DATA_DIR / "sceu_organoid" / "raw" / "41586_2017_BFnature24489_MOESM3_ESM.xlsx",
                sheet_name="Summary",
                header=5,
            ),
            pd.read_excel(
                DATA_DIR / "sceu_organoid" / "raw" / "41586_2017_BFnature24489_MOESM4_ESM.xlsx",
                sheet_name="Summary",
                header=5,
            ),
            pd.read_excel(
                DATA_DIR / "sceu_organoid" / "raw" / "41586_2017_BFnature24489_MOESM7_ESM.xlsx",
                sheet_name="Summary",
                header=5,
            ).add_prefix("EEC "),
        ]
    )

    goblet_markers = GOBLET_MARKERS + markers["Goblet"].dropna().str.replace("(_).*", "", regex=True).tolist()
    pd.DataFrame({"Gene": goblet_markers}).drop_duplicates().reset_index(drop=True).to_csv(
        DATA_DIR / "sceu_organoid" / "processed" / "goblet_markers.csv"
    )

    goblet_regulators = GOBLET_REGULATORS
    pd.DataFrame({"Gene": list(set(goblet_regulators))}).drop_duplicates().reset_index(drop=True).to_csv(
        DATA_DIR / "sceu_organoid" / "processed" / "goblet_regulators.csv"
    )

    goblet_and_paneth_regulators = GOBLET_AND_PANETH_REGULATORS
    pd.DataFrame({"Gene": list(set(goblet_and_paneth_regulators))}).drop_duplicates().reset_index(drop=True).to_csv(
        DATA_DIR / "sceu_organoid" / "processed" / "goblet_and_paneth_regulators.csv"
    )

    paneth_markers = PANETH_MARKERS + markers["Paneth"].dropna().tolist()
    pd.DataFrame({"Gene": list(set(paneth_markers))}).drop_duplicates().reset_index(drop=True).to_csv(
        DATA_DIR / "sceu_organoid" / "processed" / "paneth_markers.csv"
    )

    eec_markers = (
        EEC_MARKERS
        + pd.melt(
            markers[
                [
                    "Enteroendocrine",
                    "EEC SAKD",
                    "EEC SILA",
                    "EEC SIK",
                    "EEC SIK-P",
                    "EEC SIL-P",
                    "EEC SIN",
                    "EEC EC",
                    "EEC EC Reg4",
                ]
            ]
        )["value"]
        .dropna()
        .tolist()
    )
    pd.DataFrame({"Gene": list(set(eec_markers))}).drop_duplicates().reset_index(drop=True).to_csv(
        DATA_DIR / "sceu_organoid" / "processed" / "eec_markers.csv"
    )

    eec_progenitor_markers = (
        EEC_PROGENITOR_MARKERS
        + pd.melt(
            markers[["EEC Progenitor (early)", "EEC Progenitor (late)", "EEC Progenitor (mid)", "EEC Progenitor (A)"]]
        )["value"]
        .dropna()
        .tolist()
    )
    pd.DataFrame({"Gene": list(set(eec_progenitor_markers))}).drop_duplicates().reset_index(drop=True).to_csv(
        DATA_DIR / "sceu_organoid" / "processed" / "eec_progenitor_markers.csv"
    )

    enterocyte_markers = (
        pd.melt(markers[["Enterocyte", "Enterocyte Mature Distal", "Enterocyte Mature Proximal"]])["value"]
        .dropna()
        .tolist()
    )
    pd.DataFrame({"Gene": list(set(enterocyte_markers))}).drop_duplicates().reset_index(drop=True).to_csv(
        DATA_DIR / "sceu_organoid" / "processed" / "enterocyte_markers.csv"
    )

    enterocyte_progenitor_markers = (
        pd.melt(
            markers[
                [
                    "Enterocyte Immature Distal",
                    "Enterocyte Immature Proximal",
                    "Enterocyte Progenitor",
                    "Enterocyte Progenitor Early",
                    "Enterocyte Progenitor Late",
                    "Enterocyte progenitor (early)",
                    "Enterocyte progenitor (late)",
                ]
            ]
        )["value"]
        .dropna()
        .tolist()
    )
    pd.DataFrame({"Gene": list(set(enterocyte_progenitor_markers))}).drop_duplicates().reset_index(drop=True).to_csv(
        DATA_DIR / "sceu_organoid" / "processed" / "enterocyte_progenitor_markers.csv"
    )
