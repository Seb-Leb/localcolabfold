CURRENTPATH=`pwd`
COLABFOLDDIR="${CURRENTPATH}/colabfold"
PARAMS_DIR="${COLABFOLDDIR}/alphafold/data/params"
MSATOOLS="${COLABFOLDDIR}/tools"

echo "Creating conda environments with python3.7 as ${COLABFOLDDIR}/colabfold-conda"
. "${COLABFOLDDIR}/conda/etc/profile.d/conda.sh"
export PATH="${COLABFOLDDIR}/conda/condabin:${PATH}"
conda create -p $COLABFOLDDIR/colabfold-conda python=3.7 -y
conda activate $COLABFOLDDIR/colabfold-conda
conda update -y conda

echo "Installing conda-forge packages"
conda install -c conda-forge python=3.7 cudnn==8.2.1.32 cudatoolkit==11.1.1 openmm==7.5.1 pdbfixer -y
echo "Installing alphafold dependencies by pip"
python3.7 -m pip install absl-py==0.13.0 biopython==1.79 chex==0.0.7 dm-haiku==0.0.4 dm-tree==0.1.6 immutabledict==2.0.0 jax==0.2.14 ml-collections==0.1.0 numpy==1.19.5 scipy==1.7.0 tensorflow-gpu==2.5.0
python3.7 -m pip install jupyter matplotlib py3Dmol tqdm
python3.7 -m pip install --upgrade jax jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Apply OpenMM patch.
echo "Applying OpenMM patch..."
(cd ${COLABFOLDDIR}/colabfold-conda/lib/python3.7/site-packages/ && patch -p0 < ${COLABFOLDDIR}/docker/openmm.patch)

# Enable GPU-accelerated relaxation.
echo "Enable GPU-accelerated relaxation..."
(cd ${COLABFOLDDIR} && patch -u alphafold/relax/amber_minimize.py -i gpurelaxation.patch)

echo "Downloading runner.py"
(cd ${COLABFOLDDIR} && wget -q "https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/runner.py")

echo "Installation of Alphafold2_advanced finished."
