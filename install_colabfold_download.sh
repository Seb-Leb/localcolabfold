#!/bin/sh

# check whether `wget` and `curl` are installed
type wget || { echo "wget command is not installed. Please install it at first using apt or yum." ; exit 1 ; }
type curl || { echo "curl command is not installed. Please install it at first using apt or yum. " ; exit 1 ; }

GIT_REPO="https://github.com/deepmind/alphafold"
SOURCE_URL="https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar"
CURRENTPATH=`pwd`
COLABFOLDDIR="${CURRENTPATH}/colabfold"
PARAMS_DIR="${COLABFOLDDIR}/alphafold/data/params"
MSATOOLS="${COLABFOLDDIR}/tools"

# download the original alphafold as "${COLABFOLDDIR}"
echo "downloading the original alphafold as ${COLABFOLDDIR}..."
rm -rf ${COLABFOLDDIR}
git clone ${GIT_REPO} ${COLABFOLDDIR}
(cd ${COLABFOLDDIR}; git checkout 1e216f93f06aa04aa699562f504db1d02c3b704c --quiet)

# colabfold patches
echo "Applying several patches to be Alphafold2_advanced..."
cd ${COLABFOLDDIR}
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/colabfold.py
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/pairmsa.py
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/protein.patch
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/config.patch
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/model.patch
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/modules.patch
# GPU relaxation patch
wget -qnc https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/gpurelaxation.patch -O gpurelaxation.patch

# donwload reformat.pl from hh-suite
wget -qnc https://raw.githubusercontent.com/soedinglab/hh-suite/master/scripts/reformat.pl
# Apply multi-chain patch from Lim Heo @huhlim
patch -u alphafold/common/protein.py -i protein.patch
patch -u alphafold/model/model.py -i model.patch
patch -u alphafold/model/modules.py -i modules.patch
patch -u alphafold/model/config.py -i config.patch
cd ..

# Downloading parameter files
echo "Downloading AlphaFold2 trained parameters..."
mkdir -p ${PARAMS_DIR}
curl -fL ${SOURCE_URL} | tar x -C ${PARAMS_DIR}

# Downloading stereo_chemical_props.txt from https://git.scicore.unibas.ch/schwede/openstructure
echo "Downloading stereo_chemical_props.txt..."
wget -q https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
mkdir -p ${COLABFOLDDIR}/alphafold/common
mv stereo_chemical_props.txt ${COLABFOLDDIR}/alphafold/common

# Install Miniconda3 for Linux
echo "Installing Miniconda3 for Linux..."
cd ${COLABFOLDDIR}
wget -q -P . https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ${COLABFOLDDIR}/conda
rm Miniconda3-latest-Linux-x86_64.sh
cd ..
