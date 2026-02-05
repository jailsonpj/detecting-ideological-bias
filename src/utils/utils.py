import json
import os
import sys
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
sys.path.insert(0, project_root)
def read_json(filename: str) -> dict:
    """
    Função responsável pela leitura de um
    arquivo JSON.

    Parameters
    ----------
    filename: str
        Nome do arquivo a ser lido

    Returns
    ----------
    Dict
        Dicionário contendo dados lidos do 
        arquivo json.
    """
    f = open(filename)
    data = json.load(f)
    f.close()

    return data