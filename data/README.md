### File Format:
* pid_list.txt: each protein name a line
* go.txt: each pair of protein name and GO a line
* ppi_mat.npz: adjacency matrix in scipy.sparse.csr_matrix
* ppi_interpro.npz: the intepro binary feature of each protein in ppi_pid_list.txt
* ppi_blastdb: the blastdb of ppi.fasta

Please use the following commands to extract the archive if unzip command doesn't work:

`pip install dtrx`

`dtrx -f data.zip`
