import subprocess
import tempfile
import scanpy as sc
from pathlib import Path

HERE = Path(__file__).parent
TRANSFER_DIR = '.anndata_transfers_'

def temp_save_adata(adata):
    ''' Create a temporary directory and save a provided adata.
    Returns a tuple with the TemporaryDirectory, name of the saved file
    and a name for the future processed file
    '''

    x = tempfile.TemporaryDirectory(prefix=TRANSFER_DIR, dir='./')
    infile = x.name + '/source_adata.h5ad'
    outfile = x.name + '/corrected_adata.h5ad'
    adata.write(infile, compression='lzf')

    # Need to return the TemporaryDirectory otherwise it leaves the
    # scope and directory gets deleted
    return (x, infile, outfile)


def run_SeuratCCA(adata,
                  batch_key='data_type',
                  reference='10x',
                  anchor_dims=30,
                  integrate_dims=30,
                  k_anchor=5,
                  k_filter=200,
                  k_score=30,
                  use_vargenes=True,
                  debug=False):
    ''' Python wrapper around Routine run_SeuratCCA, which
    performs batch correction on scRNA-Seq data.
    1. Transfers the AnnData object via disk to R (temporary directory)
    2. Runs the R function run_SeuratCCA (passing the arguments)
    by creating a temporary script
    3. Reads the run_SeuratCCA output file and returns to the user

    Parameters
    ----------
    adata
        AnnData object with all batches to be corrrected
    batch_key
        name of the .obs column with the key defining data batches
    reference
        which of the keys should be treated as reference (will not be modified)
    anchor_dims
        FindIntegrationAnchors dims argument
    integrate_dims
        IntegrateData dims argument
    k_anchor
        FindIntegrationAnchors k.anchor argument
    k_filter
        FindIntegrationAnchors k.filter argument
    k_score
        FindIntegrationAnchors k.score argument
    use_vargenes
        bool, whether variable genes should be used
    debug
        bool, if True, prints stdout and stderr from the R subprocess
        and returns a tuple of corrected adata and the
        TemporaryDirectory object to prevent the directory cleanup.

    Returns
    -------
    Batch corrected adata. If debug=True returns a tuple of batch corrected adata
    and TemporaryDirectory object.
    '''

    d, infile, outfile = temp_save_adata(adata)
    print(f'Storing data in "{d.name}" temporary directory')

    if use_vargenes and ('highly_variable' not in adata.var.columns):
        raise Exception('highly_variable genes not found')

    script = f'''
source("{HERE}/subRoutines/Rutils.R")
source("{HERE}/subRoutines/batch_correct.R")
run_SeuratCCA(infile = "{infile}",
              anchor.dims = 1:{anchor_dims},
              integrate.dims = 1:{integrate_dims},
              batch_key = "{batch_key}",
              reference = "{reference}",
              k.anchor = {k_anchor},
              k.filter = {k_filter},
              k.score = {k_score},
              use_vargenes = {str(use_vargenes).upper()},
              outfile = "{outfile}")
    '''
    script_file = d.name + '/run_seuratCCA.R'
    with open(script_file, 'w') as f:
        f.write(script)

    out = subprocess.run(f'R --no-save < {script_file}',
                         check=False,  # Setting to False to allow debug
                         capture_output=True,
                         text=True,
                         shell=True)
    if debug:
        print('out:' + out.stdout)
        print('err:' + out.stderr)

    corrected = sc.read(outfile)

    if debug:
        return corrected, d

    out.check_returncode()
    return corrected
